"""
Perplexity comparison: Dense vs TEAL vs TopK vs TG-TopK (multiple clip margins).

Usage:
  python eval_ppl_comparison.py --calibration_dataset alpaca --datasets c4,wikitext2
  python eval_ppl_comparison.py --calibration_dataset alpaca --datasets c4,wikitext2 \\
      --clip_margins 0.01,0.15,0.20 --skip_dense

Output (per calibration dataset):
  analysis/ppl_comparison/{cal_dataset}/ppl_comparison_{cal}_{sp_tag}.json
  analysis/ppl_comparison/{cal_dataset}/ppl_comparison_{cal}_{sp_tag}.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM
from datasets import load_dataset

# ── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
TEAL_ROOT  = SCRIPT_DIR.parent

sys.path.insert(0, str(TEAL_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from teal.model import LlamaSparseForCausalLM, LlamaSparseConfig
from teal.model import MistralSparseForCausalLM, MistralSparseConfig
from utils.utils import get_sparse_model, get_tokenizer
from utils.eval_ppl import eval_ppl
from topk_utils import apply_topk_to_sparse_model, remove_topk_from_sparse_model
from topk_threshold_guided_utils import (
    apply_threshold_guided_topk_to_sparse_model,
    remove_threshold_guided_topk_from_sparse_model,
)

AutoConfig.register("llama_sparse", LlamaSparseConfig)
AutoConfig.register("mistral_sparse", MistralSparseConfig)
AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)

# ── Calibration paths ────────────────────────────────────────────────────────
TEAL_PATHS = {
    "alpaca": {
        "histograms": TEAL_ROOT / "models" / "Llama-2-7B-alpaca"   / "histograms",
        "lookup":     TEAL_ROOT / "models" / "Llama-2-7B-alpaca"   / "lookup",
    },
    "wikitext2": {
        "histograms": TEAL_ROOT / "models" / "Llama-2-7B-wikitext2" / "histograms",
        "lookup":     TEAL_ROOT / "models" / "Llama-2-7B-wikitext2" / "lookup",
    },
    "c4": {
        "histograms": TEAL_ROOT / "models" / "Llama-2-7B-c4"       / "histograms",
        "lookup":     TEAL_ROOT / "models" / "Llama-2-7B-c4"       / "lookup",
    },
}


# ── Dataset loaders ──────────────────────────────────────────────────────────

def load_inference_dataset(name: str, num_samples: int = 500):
    if name == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True,
                          trust_remote_code=True)
        samples = [s for s in ds.take(num_samples) if s.get("text", "").strip()]
    elif name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",
                          trust_remote_code=True)
        samples = [{"text": s["text"]} for s in ds if s["text"].strip()]
    elif name == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True,
                          trust_remote_code=True)
        samples = [s for s in ds.take(num_samples) if s.get("text", "").strip()]
    else:
        raise ValueError(f"Unknown dataset: {name!r}. Choose from alpaca, wikitext2, c4.")
    print(f"  Loaded {len(samples)} samples for {name}")
    return samples


# ── Helpers ──────────────────────────────────────────────────────────────────

def _run_ppl(model, tokenizer, dataset):
    return eval_ppl(model, tokenizer, device="cuda", dataset=dataset)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON : {path}")


def save_csv(results: dict, methods: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset"] + methods)
        for ds_name, res in results.items():
            writer.writerow([ds_name] + [res.get(m, "") for m in methods])
    print(f"Saved CSV  : {path}")


def print_table(results: dict, methods: list, cal: str, sparsity: float):
    print(f"\n{'='*80}")
    print(f" PPL COMPARISON  |  calibration={cal}  |  sparsity={sparsity}")
    print(f" Lower PPL = better")
    print(f"{'='*80}")
    header = f"{'Dataset':<14}" + "".join(f"{m:>18}" for m in methods)
    print(header)
    print("─" * len(header))
    for ds_name, res in results.items():
        row = f"{ds_name:<14}" + "".join(
            f"{str(res.get(m, 'N/A')):>18}" for m in methods
        )
        print(row)
    if "dense" in methods:
        print("─" * len(header))
        print("  Δ vs dense (positive = worse):")
        for ds_name, res in results.items():
            if "dense" not in res:
                continue
            row = f"{ds_name:<14}"
            for m in methods:
                if m == "dense" or m not in res:
                    row += f"{'—':>18}"
                else:
                    delta = res[m] - res["dense"]
                    row += f"{delta:>+17.3f}"
            print(row)
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PPL comparison: Dense vs TEAL vs TopK vs TG-TopK"
    )
    parser.add_argument("--model",               type=str,   default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--sparsity",            type=float, default=0.5)
    parser.add_argument("--calibration_dataset", type=str,   default="alpaca",
                        choices=["alpaca", "wikitext2", "c4"])
    parser.add_argument("--datasets",            type=str,   default="c4,wikitext2",
                        help="Comma-separated eval datasets: alpaca, wikitext2, c4")
    parser.add_argument("--num_samples",         type=int,   default=500)
    parser.add_argument("--clip_margins",        type=str,   default="0.01,0.15,0.20",
                        help="Comma-separated clip margins for TG-TopK, e.g. '0.01,0.15,0.20'")
    parser.add_argument("--topk_mode",           type=str,   default="global",
                        choices=["global", "per_token"],
                        help="TopK granularity: 'global' (B*S*H) or 'per_token' (per hidden dim)")
    parser.add_argument("--output_dir",          type=str,   default=None,
                        help="Override default output dir")
    parser.add_argument("--skip_dense",          action="store_true")
    parser.add_argument("--skip_teal",           action="store_true")
    parser.add_argument("--skip_topk",           action="store_true")
    parser.add_argument("--skip_tg_topk",        action="store_true")
    args = parser.parse_args()

    cal       = args.calibration_dataset
    sp_tag    = f"sp{int(args.sparsity * 100)}"
    mode_tag  = args.topk_mode          # "global" or "per_token"
    paths     = TEAL_PATHS[cal]
    cms       = [float(c.strip()) for c in args.clip_margins.split(",")]

    output_dir = Path(args.output_dir) if args.output_dir else \
                 TEAL_ROOT / "analysis" / "ppl_comparison" / cal
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = [d.strip() for d in args.datasets.split(",")]

    print(f"\n{'='*70}")
    print(f" PPL Comparison")
    print(f" Calibration : {cal}")
    print(f" Sparsity    : {args.sparsity}")
    print(f" TopK mode   : {mode_tag}")
    print(f" Eval sets   : {dataset_names}")
    print(f" TG-TopK cms : {cms}")
    print(f" Output      : {output_dir}")
    print(f"{'='*70}\n")

    # Model ek baar load karo
    print("Loading sparse model...")
    tokenizer = get_tokenizer(args.model)
    model = get_sparse_model(
        args.model,
        device="auto",
        histogram_path=str(paths["histograms"]),
    )
    model.eval()
    print("Model loaded.\n")

    results  = {}   # {ds_name: {method: ppl}}
    methods  = []   # ordered list for CSV/print

    for ds_name in dataset_names:
        print(f"\n{'─'*60}")
        print(f"  Eval dataset: {ds_name.upper()}")
        print(f"{'─'*60}")
        dataset = load_inference_dataset(ds_name, args.num_samples)
        results[ds_name] = {}

        # 1. Dense
        if not args.skip_dense:
            model.set_uniform_sparsity(0.0)
            print("  [Dense] computing PPL...")
            ppl = _run_ppl(model, tokenizer, dataset)
            results[ds_name]["dense"] = round(ppl, 3)
            print(f"  [Dense] PPL = {ppl:.3f}")
            if "dense" not in methods:
                methods.append("dense")

        # 2. TEAL greedy
        if not args.skip_teal:
            model.load_greedy_sparsities(str(paths["lookup"]), args.sparsity)
            print(f"  [TEAL greedy] computing PPL...")
            ppl = _run_ppl(model, tokenizer, dataset)
            results[ds_name]["teal_greedy"] = round(ppl, 3)
            print(f"  [TEAL greedy] PPL = {ppl:.3f}")
            model.set_uniform_sparsity(0.0)
            if "teal_greedy" not in methods:
                methods.append("teal_greedy")

        # 3. TopK exact-sparsity
        if not args.skip_topk:
            topk_key = f"topk_{mode_tag}"
            model.load_greedy_sparsities(str(paths["lookup"]), args.sparsity)
            apply_topk_to_sparse_model(
                model, args.sparsity, lookup_path=str(paths["lookup"]),
                topk_mode=mode_tag,
            )
            print(f"  [TopK {mode_tag}] computing PPL...")
            ppl = _run_ppl(model, tokenizer, dataset)
            results[ds_name][topk_key] = round(ppl, 3)
            print(f"  [TopK {mode_tag}] PPL = {ppl:.3f}")
            remove_topk_from_sparse_model(model)
            model.set_uniform_sparsity(0.0)
            if topk_key not in methods:
                methods.append(topk_key)

        # 4. TG-TopK — har clip margin ke liye
        if not args.skip_tg_topk:
            for cm in cms:
                key = f"tg_topk_cm{cm}_{mode_tag}"
                model.load_greedy_sparsities(str(paths["lookup"]), args.sparsity)
                apply_threshold_guided_topk_to_sparse_model(
                    model, args.sparsity,
                    lookup_path=str(paths["lookup"]),
                    clip_margin=cm,
                    topk_mode=mode_tag,
                )
                print(f"  [TG-TopK cm={cm} {mode_tag}] computing PPL...")
                ppl = _run_ppl(model, tokenizer, dataset)
                results[ds_name][key] = round(ppl, 3)
                print(f"  [TG-TopK cm={cm} {mode_tag}] PPL = {ppl:.3f}")
                remove_threshold_guided_topk_from_sparse_model(model)
                model.set_uniform_sparsity(0.0)
                if key not in methods:
                    methods.append(key)

    # Save
    json_path = output_dir / f"ppl_comparison_{cal}_{sp_tag}_{mode_tag}.json"
    csv_path  = output_dir / f"ppl_comparison_{cal}_{sp_tag}_{mode_tag}.csv"
    save_json(results, json_path)
    save_csv(results, methods, csv_path)
    print_table(results, methods, cal, args.sparsity)


if __name__ == "__main__":
    main()
