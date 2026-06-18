import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
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


# ── Same weight_dict as greedyopt.py: parameter-count-based weights per projection ──
WEIGHT_DICT = {
    "Llama-3-8B":  {'q': 1, 'k': 1/4, 'v': 1/4, 'o': 1, 'gate': 3.5,    'up': 3.5,    'down': 3.5},
    "Llama-3-70B": {'q': 1, 'k': 1/8, 'v': 1/8, 'o': 1, 'gate': 3.5,    'up': 3.5,    'down': 3.5},
    "Llama-2-7B":  {'q': 1, 'k': 1/8, 'v': 1/8, 'o': 1, 'gate': 2.6875, 'up': 2.6875, 'down': 2.6875},
    "Llama-2-13B": {'q': 1, 'k': 1/8, 'v': 1/8, 'o': 1, 'gate': 2.7,    'up': 2.7,    'down': 2.7},
    "Llama-2-70B": {'q': 1, 'k': 1/8, 'v': 1/8, 'o': 1, 'gate': 3.5,    'up': 3.5,    'down': 3.5},
    "Mistral-7B":  {'q': 1, 'k': 1/8, 'v': 1/8, 'o': 1, 'gate': 3.5,    'up': 3.5,    'down': 3.5},
}


def compute_calibration_sparsity(model):
    """
    Analytically compute the fraction of calibration activations that would be
    zeroed per layer per projection, using CDF(threshold) - CDF(-threshold)
    from each SparsifyFn's calibration histogram.

    Returns: dict mapping (layer_idx, "mlp.gate") → float
    """
    results = {}
    for layer_idx, layer in enumerate(model.model.layers):
        for proj, sfn in layer.mlp.sparse_fns.items():
            t = torch.tensor(sfn.threshold, dtype=torch.float32)
            cal_sp = (sfn.distr.cdf(t) - sfn.distr.cdf(-t)).item()
            results[(layer_idx, f'mlp.{proj}')] = cal_sp

        for proj, sfn in layer.self_attn.sparse_fns.items():
            t = torch.tensor(sfn.threshold, dtype=torch.float32)
            cal_sp = (sfn.distr.cdf(t) - sfn.distr.cdf(-t)).item()
            results[(layer_idx, f'attn.{proj}')] = cal_sp

    return results


def register_decode_hooks(model):
    """
    Attach forward hooks to every SparsifyFn. Each hook fires after
    SparsifyFn.forward() and, when seq_len == 1 (decode phase), counts zeros
    in the sparse output tensor.

    Returns:
        stats   – dict mapping key → {'zeros': int, 'total': int}
        handles – list of hook handles (call h.remove() when done)
    """
    stats = defaultdict(lambda: {'zeros': 0, 'total': 0})
    handles = []

    def make_hook(key):
        def hook(module, input, output):
            # input[0] is the tensor x passed to SparsifyFn.forward()
            if input[0].size(1) == 1:          # decode phase only
                zeros = (output == 0).sum().item()
                total = output.numel()
                stats[key]['zeros'] += zeros
                stats[key]['total'] += total
        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        for proj, sfn in layer.mlp.sparse_fns.items():
            key = f'L{layer_idx}_mlp_{proj}'
            h = sfn.register_forward_hook(make_hook(key))
            handles.append(h)

        for proj, sfn in layer.self_attn.sparse_fns.items():
            key = f'L{layer_idx}_attn_{proj}'
            h = sfn.register_forward_hook(make_hook(key))
            handles.append(h)

    return stats, handles


# ── Dataset configurations ──────────────────────────────────────────────────
# Each entry: (hf_name, subset, split, start, text_fields)
#   text_fields: list of keys to try in order; first non-empty one is used
DATASET_CONFIGS = {
    "alpaca": {
        "hf_name":     "tatsu-lab/alpaca",
        "subset":      None,
        "split":       "train",
        "start":       300,          # skip calibration samples 0-300
        "text_fields": ["text", "instruction"],
    },
    "wikitext2": {
        "hf_name":     "wikitext",
        "subset":      "wikitext-2-raw-v1",
        "split":       "validation",   # test split was used entirely for calibration
        "start":       0,
        "text_fields": ["text"],
    },
    "c4": {
        "hf_name":     "allenai/c4",
        "subset":      "en",
        "split":       "validation",
        "start":       500,            # skip calibration samples 0-499
        "text_fields": ["text"],
    },
}


def extract_text(sample, text_fields):
    """Return the first non-empty string found among text_fields."""
    for field in text_fields:
        val = sample.get(field, "")
        if val and val.strip():
            return val.strip()
    return ""


def run_generation(model, tokenizer, dataset, text_fields, num_prompts, num_tokens):
    """
    Run greedy autoregressive generation on dataset prompts.
    Each call produces:
      - 1 prefill forward pass  (seq_len = prompt_len) → hooks skip these
      - num_tokens decode passes (seq_len = 1)         → hooks count zeros here
    """
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
                do_sample=False,   # greedy decoding — deterministic
                use_cache=True,    # KV cache needed for real decode phase
            )

        count += 1


def build_results_df(calibration_sparsity, stats):
    """
    Merge calibration sparsity (analytical) with inference sparsity (measured)
    into a single DataFrame.
    """
    rows = []
    for (layer_idx, proj_name), cal_sp in sorted(calibration_sparsity.items()):
        module, proj = proj_name.split('.')          # e.g. "mlp", "gate"
        key = f'L{layer_idx}_{module}_{proj}'        # e.g. "L0_mlp_gate"

        s = stats.get(key, {'zeros': 0, 'total': 0})
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
        description="Compare calibration vs decode-phase inference sparsity per layer."
    )
    parser.add_argument(
        '--model_name', type=str, default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model name"
    )
    parser.add_argument(
        '--teal_path', type=str, required=True,
        help="Directory containing histograms/ subdirectory (output of grab_acts.py)"
    )
    parser.add_argument(
        '--sparsity', type=float, default=0.5,
        help="Sparsity level to apply (default: 0.5)"
    )
    parser.add_argument(
        '--greedy_flag', action='store_true',
        help="Use per-layer greedy sparsities from lookup/ instead of uniform"
    )
    parser.add_argument(
        '--dataset', type=str, default="alpaca",
        choices=list(DATASET_CONFIGS.keys()),
        help="Inference dataset: alpaca | wikitext2 | c4  (default: alpaca)"
    )
    parser.add_argument(
        '--num_prompts', type=int, default=50,
        help="Number of prompts to generate from (default: 50)"
    )
    parser.add_argument(
        '--num_tokens', type=int, default=200,
        help="Decode tokens to generate per prompt (default: 200)"
    )
    parser.add_argument(
        '--output_csv', type=str, default=None,
        help="Path for the output CSV (default: sparsity_analysis_{dataset}.csv)"
    )
    parser.add_argument(
        '--model_type', type=str, default="Llama-2-7B",
        choices=list(WEIGHT_DICT.keys()),
        help="Model type for weighted sparsity (must match WEIGHT_DICT key)"
    )
    args = parser.parse_args()

    # ── Register sparse model types with HuggingFace ──
    AutoConfig.register("llama_sparse", LlamaSparseConfig)
    AutoConfig.register("mistral_sparse", MistralSparseConfig)
    AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
    AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)

    # ── Load model + tokenizer ──
    print(f"Loading model: {args.model_name}")
    tokenizer = get_tokenizer(args.model_name)
    model = get_sparse_model(
        args.model_name,
        device="auto",
        histogram_path=os.path.join(args.teal_path, "histograms"),
    )

    # ── Set sparsity ──
    if args.greedy_flag:
        greedy_path = os.path.join(args.teal_path, "lookup")
        print(f"Loading greedy per-layer sparsities from: {greedy_path}  (target={args.sparsity})")
        model.load_greedy_sparsities(greedy_path, args.sparsity)
    else:
        print(f"Setting uniform sparsity: {args.sparsity}")
        model.set_uniform_sparsity(args.sparsity)

    # ── Step 1: Calibration sparsity (analytical, no forward pass needed) ──
    print("Computing calibration sparsity from histograms...")
    calibration_sparsity = compute_calibration_sparsity(model)

    # ── Step 2: Register decode-phase hooks ──
    print("Registering inference sparsity measurement hooks...")
    stats, handles = register_decode_hooks(model)

    # ── Step 3: Load inference dataset ──
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

    # ── Step 4: Autoregressive generation (triggers real decode path) ──
    print(
        f"Running generation: {args.num_prompts} prompts × {args.num_tokens} decode tokens "
        f"= {args.num_prompts * args.num_tokens:,} total decode steps per projection"
    )
    run_generation(model, tokenizer, dataset, cfg["text_fields"], args.num_prompts, args.num_tokens)

    # ── Remove hooks after generation ──
    for h in handles:
        h.remove()

    # ── Step 5: Build comparison table ──
    df = build_results_df(calibration_sparsity, stats)

    print("\n" + "=" * 85)
    print("PER-LAYER SPARSITY COMPARISON  (Calibration histogram vs Decode-phase inference)")
    print("=" * 85)
    print(df.to_string(index=False))

    # ── Summary grouped by projection type (block-level) ──
    df['proj_type'] = df['projection'].str.split('.').str[-1]
    block_summary = (
        df.groupby('proj_type')[['calibration_sparsity', 'inference_sparsity', 'difference']]
        .agg(['mean', 'std'])
        .round(4)
    )
    # Flatten multi-level columns: ('calibration_sparsity', 'mean') → 'cal_mean'
    block_summary.columns = [
        f"{col[0].replace('calibration_sparsity', 'cal').replace('inference_sparsity', 'inf').replace('difference', 'diff')}_{col[1]}"
        for col in block_summary.columns
    ]
    print("\n" + "=" * 80)
    print("BLOCK-LEVEL SUMMARY: avg sparsity across all 32 layers, per projection type")
    print("=" * 80)
    print(block_summary.to_string())

    # ── Model-level aggregate (greedyopt-style weighted average) ──
    # Same formula as greedyopt.py's f(): Σ(sparsity * weight) / Σ(weight), averaged across layers
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
    print(f"MODEL-LEVEL WEIGHTED AVERAGE ({args.model_type} weights, same as greedyopt)")
    print("=" * 60)
    print(f"  Calibration sparsity : {model_cal:.4f}")
    print(f"  Inference  sparsity  : {model_inf:.4f}")
    print(f"  Difference (inf-cal) : {model_diff:+.4f}")

    # ── Save ──
    mode_tag = "greedy" if args.greedy_flag else "uniform"
    output_csv = args.output_csv or f"sparsity_analysis_{args.dataset}_{mode_tag}.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nFull results saved to: {output_csv}")
