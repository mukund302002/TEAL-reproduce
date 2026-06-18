"""
3-way comparison: Dense vs TEAL vs TopK vs TG-TopK.

eval_teal_comparison.py ka extended version — TG-TopK method add kiya.

Dense, TEAL, TopK ke existing JSONs cache se load hote hain (agar available hain).
Sirf TG-TopK fresh run hota hai (ya woh bhi cache se agar already run ho chuka ho).

Output structure (same as eval_teal_comparison.py):
  analysis/comparison/{sp_tag}/{cal_dataset}/
    dense_results.json                     ← cache se load hoga
    teal_{cal}_{sp_tag}_results.json       ← cache se load hoga
    topk_{sp_tag}_results.json             ← cache se load hoga
    tg_topk_{cal}_{sp_tag}_results.json    ← naya (fresh run ya cache)
    comparison_summary_3way_{cal}_{sp_tag}.csv  ← naya 4-column CSV

Usage:
  python teal/eval_three_way_comparison.py \\
      --calibration_dataset alpaca \\
      --output_dir analysis/comparison/50 \\
      --sparsity 0.5
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
TEAL_ROOT   = SCRIPT_DIR.parent
TEAL_UTILS  = TEAL_ROOT / "utils"
LM_EVAL_DIR = TEAL_ROOT.parent / "LayerNorm-Scaling" / "lm-evaluation-harness"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(TEAL_ROOT))
sys.path.insert(0, str(LM_EVAL_DIR))

from teal.model import LlamaSparseForCausalLM, LlamaSparseConfig
from teal.model import MistralSparseForCausalLM, MistralSparseConfig
from utils.utils import get_sparse_model, get_tokenizer
from topk_utils import apply_topk_to_sparse_model, remove_topk_from_sparse_model
from topk_threshold_guided_utils import (
    apply_threshold_guided_topk_to_sparse_model,
    remove_threshold_guided_topk_from_sparse_model,
)

AutoConfig.register("llama_sparse",   LlamaSparseConfig)
AutoConfig.register("mistral_sparse", MistralSparseConfig)
AutoModelForCausalLM.register(LlamaSparseConfig,   LlamaSparseForCausalLM)
AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)

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

# Default few-shot counts per task (standard lm-eval benchmarks).
# --task_shots CLI argument se override ho sakta hai.
DEFAULT_TASK_SHOTS = {
    "arc_easy":      25,
    "arc_challenge": 25,
    "hellaswag":     10,
    "piqa":           0,
    "winogrande":     5,
    "sciq":           0,
    "openbookqa":     0,
    "boolq":          0,
}


# ---------------------------------------------------------------------------
# Helpers (eval_teal_comparison.py se same)
# ---------------------------------------------------------------------------

def load_sparse_model(model_name: str, histogram_path: str):
    print(f"Loading sparse model: {model_name}")
    print(f"  Histograms: {histogram_path}")
    tokenizer = get_tokenizer(model_name)
    model = get_sparse_model(model_name, device="auto", histogram_path=str(histogram_path))
    model.eval()
    return model, tokenizer


def evaluate_model(model, tokenizer, tasks: list, batch_size: int,
                   task_shots: dict) -> dict:
    """
    task_shots : dict mapping task_name → num_fewshot
                 e.g. {"arc_easy": 25, "hellaswag": 10, "piqa": 0}
    Tasks with same shot count are batched into one simple_evaluate call.
    Results are merged into a single dict.
    """
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from collections import defaultdict

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    # Group tasks by shot count → minimize number of simple_evaluate calls
    shots_to_tasks = defaultdict(list)
    for t in tasks:
        shots_to_tasks[task_shots.get(t, 0)].append(t)

    combined = {"results": {}}
    for shots, task_group in shots_to_tasks.items():
        print(f"  [{shots}-shot] tasks: {task_group}")
        res = simple_evaluate(
            model=lm,
            tasks=task_group,
            num_fewshot=shots,
            batch_size=batch_size,
            log_samples=False,
        )
        combined["results"].update(res["results"])

    return combined


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


def load_json_summary(path: Path, label: str) -> dict | None:
    """JSON se summary load karo agar file exist karti hai."""
    if path.exists():
        with open(path) as f:
            print(f"{label}: loaded from cache ({path.name})")
            return extract_summary(json.load(f))
    return None


def write_summary_csv(all_summaries: dict, output_dir: Path, tag: str):
    csv_path = output_dir / f"comparison_summary_3way_{tag}.csv"
    all_keys = sorted({k for s in all_summaries.values() for k in s.keys()})
    methods  = sorted(all_summaries.keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task/metric"] + methods)
        for key in all_keys:
            row = [key] + [all_summaries[m].get(key, "") for m in methods]
            writer.writerow(row)

    print(f"\nSummary CSV saved: {csv_path}")
    print(f"\n{'='*85}")
    print(f"3-WAY COMPARISON  ({tag})")
    print(f"{'='*85}")
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
        description="3-way comparison: Dense vs TEAL vs TopK vs TG-TopK"
    )
    parser.add_argument("--model",               type=str,   default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--sparsity",            type=float, default=0.5)
    parser.add_argument("--calibration_dataset", type=str,   default="alpaca",
                        choices=["alpaca", "wikitext2", "c4"])
    parser.add_argument("--tasks",               type=str,
                        default="arc_easy,arc_challenge,hellaswag,piqa,winogrande,sciq,openbookqa,boolq")
    parser.add_argument("--output_dir",          type=str,
                        default=str(TEAL_ROOT / "analysis" / "comparison" / "50"),
                        help="Parent dir; cal_dataset subfolder auto-created")
    parser.add_argument("--batch_size",          type=int,   default=4)
    parser.add_argument("--num_fewshot",         type=int,   default=0,
                        help="Default shots for all tasks (overridden by --task_shots)")
    parser.add_argument("--task_shots",          type=str,   default=None,
                        help="Per-task shot override: 'arc_easy:25,arc_challenge:25,hellaswag:10'")
    parser.add_argument("--skip_dense",          action="store_true")
    parser.add_argument("--skip_teal",           action="store_true")
    parser.add_argument("--skip_topk",           action="store_true")
    parser.add_argument("--skip_tg_topk",        action="store_true")
    parser.add_argument("--clip_margin",         type=float, default=0.015,
                        help="TG-TopK adjustment window (default: 0.015)")
    args = parser.parse_args()

    tasks      = [t.strip() for t in args.tasks.split(",")]

    # task_shots dict build karo
    # Priority: --task_shots CLI > DEFAULT_TASK_SHOTS > --num_fewshot (fallback)
    task_shots = {t: DEFAULT_TASK_SHOTS.get(t, args.num_fewshot) for t in tasks}
    if args.task_shots:
        for entry in args.task_shots.split(","):
            task, shots = entry.strip().split(":")
            task_shots[task.strip()] = int(shots.strip())
    print(f"  Shot counts: { {t: s for t, s in task_shots.items() if s > 0} or 'all 0-shot' }")

    sp_tag     = f"sp{int(args.sparsity * 100)}"
    cal_tag    = args.calibration_dataset
    cm_tag     = f"cm{args.clip_margin}"
    full_tag   = f"{cal_tag}_{sp_tag}"
    output_dir = Path(args.output_dir) / cal_tag    # e.g. analysis/comparison/50/alpaca/
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = TEAL_PATHS[cal_tag]

    print(f"\n{'='*70}")
    print(f" Dense vs TEAL vs TopK vs TG-TopK")
    print(f" sparsity={args.sparsity}  |  calibration={cal_tag}  |  clip_margin={args.clip_margin}")
    print(f" Tasks  : {tasks}")
    print(f" Output : {output_dir}")
    print(f"{'='*70}\n")

    # Model ek baar load karo — sab evaluations mein reuse hoga
    model, tokenizer = load_sparse_model(args.model, str(paths["histograms"]))
    all_summaries = {}

    # ------------------------------------------------------------------
    # 1. Dense baseline
    # ------------------------------------------------------------------
    dense_json = output_dir / "dense_results.json"
    if not args.skip_dense:
        cached = load_json_summary(dense_json, "Dense")
        if cached:
            all_summaries["dense"] = cached
        else:
            print("--- Evaluating: Dense ---")
            results = evaluate_model(model, tokenizer, tasks, args.batch_size, task_shots)
            save_json(results, dense_json)
            all_summaries["dense"] = extract_summary(results)
            print("Dense done.\n")

    # ------------------------------------------------------------------
    # 2. TEAL greedy
    # ------------------------------------------------------------------
    teal_json = output_dir / f"teal_{full_tag}_results.json"
    if not args.skip_teal:
        cached = load_json_summary(teal_json, f"TEAL ({cal_tag})")
        if cached:
            all_summaries[f"teal_{cal_tag}"] = cached
        else:
            print(f"--- Evaluating: TEAL greedy (calibration={cal_tag}, sparsity={args.sparsity}) ---")
            model.load_greedy_sparsities(str(paths["lookup"]), args.sparsity)
            results = evaluate_model(model, tokenizer, tasks, args.batch_size, task_shots)
            save_json(results, teal_json)
            all_summaries[f"teal_{cal_tag}"] = extract_summary(results)
            model.set_uniform_sparsity(0.0)
            print("TEAL done. Reset to dense.\n")

    # ------------------------------------------------------------------
    # 3. TopK exact-sparsity
    # ------------------------------------------------------------------
    topk_json = output_dir / f"topk_{sp_tag}_results.json"
    if not args.skip_topk:
        cached = load_json_summary(topk_json, "TopK")
        if cached:
            all_summaries["topk"] = cached
        else:
            print(f"--- Evaluating: TopK (sparsity={args.sparsity}, greedy distribution) ---")
            model.load_greedy_sparsities(str(paths["lookup"]), args.sparsity)
            apply_topk_to_sparse_model(model, args.sparsity, lookup_path=str(paths["lookup"]))
            results = evaluate_model(model, tokenizer, tasks, args.batch_size, task_shots)
            save_json(results, topk_json)
            all_summaries["topk"] = extract_summary(results)
            remove_topk_from_sparse_model(model)
            model.set_uniform_sparsity(0.0)
            print("TopK done. Reset to dense.\n")

    # ------------------------------------------------------------------
    # 4. TG-TopK (Threshold-Guided TopK)
    # ------------------------------------------------------------------
    tg_topk_key  = f"tg_topk_{cal_tag}_{cm_tag}"
    tg_topk_json = output_dir / f"tg_topk_{full_tag}_{cm_tag}_results.json"
    if not args.skip_tg_topk:
        cached = load_json_summary(tg_topk_json, f"TG-TopK ({cal_tag}, {cm_tag})")
        if cached:
            all_summaries[tg_topk_key] = cached
        else:
            print(f"--- Evaluating: TG-TopK (calibration={cal_tag}, sparsity={args.sparsity}, clip_margin={args.clip_margin}) ---")
            # Step 1: thresholds set karo (sfn.threshold populate hoga)
            model.load_greedy_sparsities(str(paths["lookup"]), args.sparsity)
            # Step 2: SparsifyFn → ThresholdGuidedTopKSparsifyFn
            apply_threshold_guided_topk_to_sparse_model(
                model, args.sparsity, lookup_path=str(paths["lookup"]),
                clip_margin=args.clip_margin,
            )
            results = evaluate_model(model, tokenizer, tasks, args.batch_size, task_shots)
            save_json(results, tg_topk_json)
            all_summaries[tg_topk_key] = extract_summary(results)
            # Step 3: original SparsifyFn restore karo, phir dense pe reset
            remove_threshold_guided_topk_from_sparse_model(model)
            model.set_uniform_sparsity(0.0)
            print("TG-TopK done. Reset to dense.\n")

    # ------------------------------------------------------------------
    # 5. Combined summary CSV
    # ------------------------------------------------------------------
    if len(all_summaries) >= 2:
        write_summary_csv(all_summaries, output_dir, f"{full_tag}_{cm_tag}")


if __name__ == "__main__":
    main()
