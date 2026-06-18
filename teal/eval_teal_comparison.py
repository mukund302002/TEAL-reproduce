"""
Evaluate TEAL (greedy threshold-based) vs TopK (exact-sparsity) activation sparsity.

Both methods use LlamaSparseForCausalLM (TEAL's native sparse model class).

  TEAL  : greedy per-layer per-projection thresholds from lookup/ CSVs.
           Calibrated on a specific dataset (Alpaca or Wikitext-2).
           Actual inference sparsity depends on runtime activation distribution.

  TopK  : SparsifyFn objects replaced with TopKSparsifyFn at inference time.
           No calibration needed — exact sparsity guaranteed per call.

Usage:
  # Alpaca calibration (pre-computed histograms + lookup)
  python eval_teal_comparison.py --calibration_dataset alpaca \\
      --output_dir ../analysis/comparison/alpaca

  # Wikitext-2 calibration (run grab_acts_wikitext2.py + greedyopt.py first)
  python eval_teal_comparison.py --calibration_dataset wikitext2 \\
      --output_dir ../analysis/comparison/wikitext2 --skip_dense

Results saved under output_dir/:
  dense_results.json
  teal_{cal_dataset}_sp{XX}_results.json
  topk_sp{XX}_results.json
  comparison_summary_{cal_dataset}_sp{XX}.csv
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
TEAL_ROOT   = SCRIPT_DIR.parent                                         # TEAL/
TEAL_UTILS  = TEAL_ROOT / "utils"
LM_EVAL_DIR = TEAL_ROOT.parent / "LayerNorm-Scaling" / "lm-evaluation-harness"

sys.path.insert(0, str(SCRIPT_DIR))   # topk_utils.py
sys.path.insert(0, str(TEAL_ROOT))    # utils/utils.py
sys.path.insert(0, str(LM_EVAL_DIR))

from teal.model import LlamaSparseForCausalLM, LlamaSparseConfig          # noqa: E402
from teal.model import MistralSparseForCausalLM, MistralSparseConfig      # noqa: E402
from utils.utils import get_sparse_model, get_tokenizer                    # noqa: E402
from topk_utils import apply_topk_to_sparse_model, remove_topk_from_sparse_model  # noqa: E402

AutoConfig.register("llama_sparse", LlamaSparseConfig)
AutoConfig.register("mistral_sparse", MistralSparseConfig)
AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)

# Pre-computed paths (built by grab_acts.py / grab_acts_wikitext2.py + greedyopt.py)
TEAL_PATHS = {
    "alpaca": {
        "histograms": TEAL_ROOT / "models" / "Llama-2-7B-alpaca"           / "histograms",
        "lookup":     TEAL_ROOT / "models" / "Llama-2-7B-alpaca"           / "lookup",
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sparse_model(model_name: str, histogram_path: str):
    """Load LlamaSparseForCausalLM with distributions loaded from histograms."""
    print(f"Loading sparse model: {model_name}")
    print(f"  Histograms: {histogram_path}")
    tokenizer = get_tokenizer(model_name)
    model = get_sparse_model(model_name, device="auto", histogram_path=histogram_path)
    model.eval()
    return model, tokenizer


def evaluate_model(model, tokenizer, tasks: list, batch_size: int, num_fewshot: int = 0) -> dict:
    """Evaluate using lm_eval Python API."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        log_samples=False,
    )
    return results


def extract_summary(results: dict) -> dict:
    summary = {}
    for task, metrics in results["results"].items():
        for metric, value in metrics.items():
            if not metric.endswith("_stderr") and not metric.startswith("alias"):
                key = f"{task}/{metric}"
                summary[key] = round(float(value), 4) if isinstance(value, float) else value
    return summary


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved: {path}")


def write_summary_csv(all_summaries: dict, output_dir: Path, tag: str):
    csv_path = output_dir / f"comparison_summary_{tag}.csv"
    all_keys = sorted({k for s in all_summaries.values() for k in s.keys()})
    methods  = sorted(all_summaries.keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task/metric"] + methods)
        for key in all_keys:
            row = [key] + [all_summaries[m].get(key, "") for m in methods]
            writer.writerow(row)

    print(f"\nSummary CSV saved: {csv_path}")
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY  ({tag})")
    print(f"{'='*80}")
    header = f"{'Metric':<45}" + "".join(f"{m:>18}" for m in methods)
    print(header)
    print("-" * len(header))
    for key in all_keys:
        row_str = f"{key:<45}" + "".join(
            f"{str(all_summaries[m].get(key, ''))[:17]:>18}" for m in methods
        )
        print(row_str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare TEAL greedy vs TopK exact-sparsity"
    )
    parser.add_argument("--model",               type=str,   default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--sparsity",            type=float, default=0.5)
    parser.add_argument("--calibration_dataset", type=str,   default="alpaca",
                        choices=["alpaca", "wikitext2", "c4"])
    parser.add_argument("--tasks",               type=str,
                        default="arc_easy,arc_challenge,hellaswag,piqa,winogrande,sciq,openbookqa,boolq")
    parser.add_argument("--output_dir",          type=str,   default=str(TEAL_ROOT / "analysis" / "comparison"))
    parser.add_argument("--batch_size",          type=int,   default=4)
    parser.add_argument("--num_fewshot",         type=int,   default=0)
    parser.add_argument("--skip_dense",          action="store_true")
    parser.add_argument("--skip_teal",           action="store_true")
    parser.add_argument("--skip_topk",           action="store_true")
    args = parser.parse_args()

    tasks      = [t.strip() for t in args.tasks.split(",")]
    sp_tag     = f"sp{int(args.sparsity * 100)}"
    cal_tag    = args.calibration_dataset
    full_tag   = f"{cal_tag}_{sp_tag}"
    output_dir = Path(args.output_dir) / cal_tag   # e.g. analysis/comparison/alpaca/

    paths = TEAL_PATHS[cal_tag]

    print(f"\n{'='*65}")
    print(f" TEAL(greedy) vs TopK  |  sparsity={args.sparsity}  |  calibration={cal_tag}")
    print(f" Tasks  : {tasks}")
    print(f" Output : {output_dir}")
    print(f"{'='*65}\n")

    # Load sparse model once — reuse across all evaluations
    model, tokenizer = load_sparse_model(args.model, str(paths["histograms"]))
    all_summaries = {}

    # ------------------------------------------------------------------
    # 1. Dense baseline — threshold=0 (default state of sparse model)
    # ------------------------------------------------------------------
    dense_json = output_dir / "dense_results.json"
    if not args.skip_dense:
        print("--- Evaluating: Dense (threshold=0) ---")
        # Sparse model loads with threshold=0 by default → dense behavior
        results = evaluate_model(model, tokenizer, tasks, args.batch_size, args.num_fewshot)
        save_json(results, dense_json)
        all_summaries["dense"] = extract_summary(results)
        print("Dense done.\n")
    elif dense_json.exists():
        with open(dense_json) as f:
            all_summaries["dense"] = extract_summary(json.load(f))
        print("Dense: loaded from cache.\n")

    # ------------------------------------------------------------------
    # 2. TEAL greedy — load per-layer per-projection thresholds from lookup/
    # ------------------------------------------------------------------
    teal_json = output_dir / f"teal_{full_tag}_results.json"
    if not args.skip_teal:
        lookup_path = paths["lookup"]
        if not lookup_path.exists():
            raise FileNotFoundError(
                f"Lookup not found: {lookup_path}\n"
                f"Run greedyopt.py first:\n"
                f"  python greedyopt.py --model_name {args.model} "
                f"--model_type Llama-2-7B "
                f"--teal_path ../models/Llama-2-7B-wikitext2"
            )

        print(f"--- Evaluating: TEAL greedy (calibration={cal_tag}, sparsity={args.sparsity}) ---")
        print(f"  Lookup: {lookup_path}")
        model.load_greedy_sparsities(str(lookup_path), args.sparsity)
        results = evaluate_model(model, tokenizer, tasks, args.batch_size, args.num_fewshot)
        save_json(results, teal_json)
        all_summaries[f"teal_{cal_tag}"] = extract_summary(results)

        # Reset to dense (threshold=0) for next evaluation
        model.set_uniform_sparsity(0.0)
        print("TEAL done. Thresholds reset to 0.\n")

    elif teal_json.exists():
        with open(teal_json) as f:
            all_summaries[f"teal_{cal_tag}"] = extract_summary(json.load(f))
        print(f"TEAL ({cal_tag}): loaded from cache.\n")

    # ------------------------------------------------------------------
    # 3. TopK — replace SparsifyFn with TopKSparsifyFn
    # ------------------------------------------------------------------
    topk_json = output_dir / f"topk_{sp_tag}_results.json"
    if not args.skip_topk:
        print(f"--- Evaluating: TopK exact-sparsity (sparsity={args.sparsity}, greedy distribution) ---")
        apply_topk_to_sparse_model(model, args.sparsity, lookup_path=str(paths["lookup"]))
        results = evaluate_model(model, tokenizer, tasks, args.batch_size, args.num_fewshot)
        save_json(results, topk_json)
        all_summaries["topk"] = extract_summary(results)
        remove_topk_from_sparse_model(model)
        print("TopK done. Original SparsifyFn restored.\n")

    elif topk_json.exists():
        with open(topk_json) as f:
            all_summaries["topk"] = extract_summary(json.load(f))
        print("TopK: loaded from cache.\n")

    # ------------------------------------------------------------------
    # 4. Save combined summary CSV
    # ------------------------------------------------------------------
    if len(all_summaries) >= 2:
        write_summary_csv(all_summaries, output_dir, full_tag)


if __name__ == "__main__":
    main()
