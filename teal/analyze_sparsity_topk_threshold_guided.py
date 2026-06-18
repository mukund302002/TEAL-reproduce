"""
Threshold-Guided TopK experiment — sparsity analysis script.

analyze_sparsity.py se kya alag hai:
  Original : inference time pe SparsifyFn (threshold-based) use hota hai
  Yahan   : inference time pe ThresholdGuidedTopKSparsifyFn use hota hai
             (threshold signal se s2 ko ±1% clip karke TopK lagate hain)

Calibration sparsity wahi rehti hai (histogram CDF se analytical) —
kyunki woh TEAL threshold ka prediction hai, change nahi hoti.

Steps:
  1. Sparse model load + sparsity set  (SparsifyFn thresholds set)
  2. compute_calibration_sparsity      (SparsifyFn.distr + threshold se)
  3. apply_threshold_guided_topk       (SparsifyFns → ThresholdGuidedTopKSparsifyFn)
  4. register_decode_hooks             (ab nayi ThresholdGuidedTopK objects pe)
  5. Generation run
  6. Results table + CSV
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import torch
import argparse
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

if __name__ == "__main__":
    from utils.utils import get_tokenizer, get_sparse_model
    from utils.data import get_dataset
    from teal.model import (
        LlamaSparseForCausalLM, LlamaSparseConfig,
        MistralSparseForCausalLM, MistralSparseConfig,
    )
    from transformers import AutoConfig, AutoModelForCausalLM
    from teal.topk_threshold_guided_utils import apply_threshold_guided_topk_to_sparse_model


# ── analyze_sparsity.py se same weight dict ──────────────────────────────────
WEIGHT_DICT = {
    "Llama-3-8B":  {'q': 1, 'k': 1/4, 'v': 1/4, 'o': 1, 'gate': 3.5,    'up': 3.5,    'down': 3.5},
    "Llama-3-70B": {'q': 1, 'k': 1/8, 'v': 1/8, 'o': 1, 'gate': 3.5,    'up': 3.5,    'down': 3.5},
    "Llama-2-7B":  {'q': 1, 'k': 1/8, 'v': 1/8, 'o': 1, 'gate': 2.6875, 'up': 2.6875, 'down': 2.6875},
    "Llama-2-13B": {'q': 1, 'k': 1/8, 'v': 1/8, 'o': 1, 'gate': 2.7,    'up': 2.7,    'down': 2.7},
    "Llama-2-70B": {'q': 1, 'k': 1/8, 'v': 1/8, 'o': 1, 'gate': 3.5,    'up': 3.5,    'down': 3.5},
    "Mistral-7B":  {'q': 1, 'k': 1/8, 'v': 1/8, 'o': 1, 'gate': 3.5,    'up': 3.5,    'down': 3.5},
}


# ── Dataset configs (analyze_sparsity.py se same) ────────────────────────────
DATASET_CONFIGS = {
    "alpaca": {
        "hf_name":     "tatsu-lab/alpaca",
        "subset":      None,
        "split":       "train",
        "start":       300,
        "text_fields": ["text", "instruction"],
    },
    "wikitext2": {
        "hf_name":     "wikitext",
        "subset":      "wikitext-2-raw-v1",
        "split":       "validation",
        "start":       0,
        "text_fields": ["text"],
    },
    "c4": {
        "hf_name":     "allenai/c4",
        "subset":      "en",
        "split":       "validation",
        "start":       500,
        "text_fields": ["text"],
    },
}


def extract_text(sample, text_fields):
    for field in text_fields:
        val = sample.get(field, "")
        if val and val.strip():
            return val.strip()
    return ""


# ── Step 2: calibration sparsity (analyze_sparsity.py se bilkul same) ────────
def compute_calibration_sparsity(model):
    """
    Har projection ke liye histogram CDF se analytically compute karo:
      cal_sp = CDF(+threshold) - CDF(-threshold)

    Ye SparsifyFn.distr aur SparsifyFn.threshold use karta hai.
    ZARURI: apply_threshold_guided_topk se PEHLE call karo,
            kyunki replace hone ke baad distr available nahi hogi.
    """
    results = {}
    for layer_idx, layer in enumerate(model.model.layers):
        for proj, sfn in layer.mlp.sparse_fns.items():
            t      = torch.tensor(sfn.threshold, dtype=torch.float32)
            cal_sp = (sfn.distr.cdf(t) - sfn.distr.cdf(-t)).item()
            results[(layer_idx, f'mlp.{proj}')] = cal_sp

        for proj, sfn in layer.self_attn.sparse_fns.items():
            t      = torch.tensor(sfn.threshold, dtype=torch.float32)
            cal_sp = (sfn.distr.cdf(t) - sfn.distr.cdf(-t)).item()
            results[(layer_idx, f'attn.{proj}')] = cal_sp

    return results


# ── Step 4: decode hooks (analyze_sparsity.py se same — module type se independent) ──
def register_decode_hooks(model):
    """
    Har sparse_fns module pe forward hook lagao.
    ThresholdGuidedTopKSparsifyFn ka output bhi zeroed tensor hai,
    toh hook bilkul same tarah kaam karta hai.
    """
    stats   = defaultdict(lambda: {'zeros': 0, 'total': 0})
    handles = []

    def make_hook(key):
        def hook(module, input, output):
            if input[0].size(1) == 1:          # decode phase only
                zeros = (output == 0).sum().item()
                total = output.numel()
                stats[key]['zeros'] += zeros
                stats[key]['total'] += total
        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        for proj, sfn in layer.mlp.sparse_fns.items():
            key = f'L{layer_idx}_mlp_{proj}'
            handles.append(sfn.register_forward_hook(make_hook(key)))

        for proj, sfn in layer.self_attn.sparse_fns.items():
            key = f'L{layer_idx}_attn_{proj}'
            handles.append(sfn.register_forward_hook(make_hook(key)))

    return stats, handles


def run_generation(model, tokenizer, dataset, text_fields, num_prompts, num_tokens):
    model.eval()
    count = 0
    for sample in tqdm(dataset, total=num_prompts, desc="Generating"):
        if count >= num_prompts:
            break
        text = extract_text(sample, text_fields)
        if not text:
            continue
        input_ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).input_ids.to("cuda:0")
        with torch.no_grad():
            model.generate(
                input_ids,
                max_new_tokens=num_tokens,
                do_sample=False,
                use_cache=True,
            )
        count += 1


def build_results_df(calibration_sparsity, stats):
    rows = []
    for (layer_idx, proj_name), cal_sp in sorted(calibration_sparsity.items()):
        module, proj = proj_name.split('.')
        key = f'L{layer_idx}_{module}_{proj}'

        s      = stats.get(key, {'zeros': 0, 'total': 0})
        inf_sp = s['zeros'] / s['total'] if s['total'] > 0 else float('nan')

        rows.append({
            'layer':                layer_idx,
            'projection':           proj_name,
            'calibration_sparsity': round(cal_sp, 4),
            'inference_sparsity':   round(inf_sp, 4) if s['total'] > 0 else float('nan'),
            'difference':           round(inf_sp - cal_sp, 4) if s['total'] > 0 else float('nan'),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Threshold-Guided TopK experiment: calibration (histogram) vs "
            "decode-phase inference sparsity per layer."
        )
    )
    parser.add_argument('--model_name',   type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--teal_path',    type=str, required=True,
                        help="Directory with histograms/ and lookup/ subdirs")
    parser.add_argument('--sparsity',     type=float, default=0.5)
    parser.add_argument('--greedy_flag',  action='store_true',
                        help="Per-layer greedy sparsities from lookup/ use karo")
    parser.add_argument('--dataset',      type=str, default="alpaca",
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Evaluation dataset (inference pe chalega)")
    parser.add_argument('--cal_dataset',  type=str, required=True,
                        choices=["alpaca", "wikitext2", "c4"],
                        help="Calibration dataset (histograms kis data pe bane the)")
    parser.add_argument('--num_prompts',  type=int, default=50)
    parser.add_argument('--num_tokens',   type=int, default=200)
    parser.add_argument('--output_csv',   type=str, default=None,
                        help="Override default save path (optional)")
    parser.add_argument('--model_type',   type=str, default="Llama-2-7B",
                        choices=list(WEIGHT_DICT.keys()))
    args = parser.parse_args()

    # ── Output path: analysis/data/module_wise_sparsity/tg_topk/{cal_dataset}/ ──
    # Same structure as analyze_sparsity.py outputs under calibration/
    TEAL_ROOT   = os.path.abspath(os.path.join(current_dir, os.pardir))
    mode_tag    = "greedy" if args.greedy_flag else "uniform"
    sp_tag      = str(int(args.sparsity * 100))
    cal_tag     = args.cal_dataset                         # e.g. "alpaca"
    eval_tag    = args.dataset                             # e.g. "wikitext2"
    default_csv = os.path.join(
        TEAL_ROOT, "analysis", "data", "module_wise_sparsity",
        "tg_topk", cal_tag,
        f"sparsity_analysis_tg_topk_{cal_tag}cal_{eval_tag}_{mode_tag}.csv",
    )
    output_csv  = args.output_csv or default_csv
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # ── HuggingFace mein sparse model types register karo ──
    AutoConfig.register("llama_sparse",   LlamaSparseConfig)
    AutoConfig.register("mistral_sparse", MistralSparseConfig)
    AutoModelForCausalLM.register(LlamaSparseConfig,   LlamaSparseForCausalLM)
    AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)

    # ── Step 1: Model + tokenizer load ──
    print(f"Loading model: {args.model_name}")
    tokenizer = get_tokenizer(args.model_name)
    model = get_sparse_model(
        args.model_name,
        device="auto",
        histogram_path=os.path.join(args.teal_path, "histograms"),
    )

    # ── Sparsity set karo (SparsifyFn thresholds set ho jaate hain) ──
    if args.greedy_flag:
        greedy_path = os.path.join(args.teal_path, "lookup")
        print(f"Loading greedy per-layer sparsities from: {greedy_path}  (target={args.sparsity})")
        model.load_greedy_sparsities(greedy_path, args.sparsity)
    else:
        print(f"Setting uniform sparsity: {args.sparsity}")
        model.set_uniform_sparsity(args.sparsity)

    # ── Step 2: Calibration sparsity — PEHLE compute karo (SparsifyFn.distr chahiye) ──
    print("Computing calibration sparsity from histograms (before TopK replacement)...")
    calibration_sparsity = compute_calibration_sparsity(model)

    # ── Step 3: SparsifyFn → ThresholdGuidedTopKSparsifyFn replace karo ──
    lookup_path = os.path.join(args.teal_path, "lookup") if args.greedy_flag else None
    print("Replacing SparsifyFn with ThresholdGuidedTopKSparsifyFn for inference...")
    apply_threshold_guided_topk_to_sparse_model(
        model,
        sparsity=args.sparsity,
        lookup_path=lookup_path,
    )

    # ── Step 4: Decode hooks register karo (ab ThresholdGuidedTopK objects pe) ──
    print("Registering inference sparsity measurement hooks...")
    stats, handles = register_decode_hooks(model)

    # ── Step 5: Inference dataset load ──
    cfg = DATASET_CONFIGS[args.dataset]
    print(
        f"Loading dataset: {args.dataset}  "
        f"({cfg['hf_name']}, split={cfg['split']}, start={cfg['start']}, "
        f"size={args.num_prompts})"
    )
    dataset = get_dataset(
        cfg["hf_name"],
        subset=cfg["subset"],
        split=cfg["split"],
        size=args.num_prompts,
        start=cfg["start"],
    )

    # ── Step 6: Generation (real decode path trigger karo) ──
    print(
        f"Running generation: {args.num_prompts} prompts × {args.num_tokens} decode tokens "
        f"= {args.num_prompts * args.num_tokens:,} total decode steps per projection"
    )
    run_generation(model, tokenizer, dataset, cfg["text_fields"], args.num_prompts, args.num_tokens)

    for h in handles:
        h.remove()

    # ── Step 7: Results table ──
    df = build_results_df(calibration_sparsity, stats)

    print("\n" + "=" * 90)
    print("PER-LAYER SPARSITY  (Calibration histogram  vs  ThresholdGuidedTopK decode-phase)")
    print("=" * 90)
    print(df.to_string(index=False))

    # Block-level summary
    df['proj_type'] = df['projection'].str.split('.').str[-1]
    block_summary = (
        df.groupby('proj_type')[['calibration_sparsity', 'inference_sparsity', 'difference']]
        .agg(['mean', 'std'])
        .round(4)
    )
    block_summary.columns = [
        f"{col[0].replace('calibration_sparsity', 'cal').replace('inference_sparsity', 'inf').replace('difference', 'diff')}_{col[1]}"
        for col in block_summary.columns
    ]
    print("\n" + "=" * 80)
    print("BLOCK-LEVEL SUMMARY")
    print("=" * 80)
    print(block_summary.to_string())

    # Model-level weighted average
    weights      = WEIGHT_DICT[args.model_type]
    total_weight = sum(weights.values())
    df['weight'] = df['proj_type'].map(weights)

    per_layer_cal = df.groupby('layer').apply(
        lambda g: (g['calibration_sparsity'] * g['weight']).sum() / total_weight
    )
    per_layer_inf = df.groupby('layer').apply(
        lambda g: (g['inference_sparsity'] * g['weight']).sum() / total_weight
    )

    model_cal  = per_layer_cal.mean()
    model_inf  = per_layer_inf.mean()
    model_diff = model_inf - model_cal

    print("\n" + "=" * 60)
    print(f"MODEL-LEVEL WEIGHTED AVERAGE ({args.model_type} weights)")
    print("=" * 60)
    print(f"  Calibration sparsity (TEAL threshold) : {model_cal:.4f}")
    print(f"  Inference  sparsity  (TG-TopK)        : {model_inf:.4f}")
    print(f"  Difference (inf - cal)                : {model_diff:+.4f}")

    df.to_csv(output_csv, index=False)
    print(f"\nFull results saved to: {output_csv}")
