"""
PPL sweep: calibrated model × inference datasets (greedy TEAL only).

Ek calibration model lo, saare (ya kuch) inference datasets pe PPL compute karo.
3×3 matrix ke liye teen baar chalao (alpaca / c4 / wikitext2 calibration).

Usage:
  python eval_ppl_cal_sweep.py --cal_dataset alpaca
  python eval_ppl_cal_sweep.py --cal_dataset c4
  python eval_ppl_cal_sweep.py --cal_dataset wikitext2
  python eval_ppl_cal_sweep.py --cal_dataset alpaca --inf_datasets alpaca,c4
  python eval_ppl_cal_sweep.py --cal_dataset alpaca --num_samples 200

Output (example for c4 calibration, sp50):
  analysis/data/calibration/c4/ppl_c4cal_sp50_greedy.json
  analysis/data/calibration/c4/ppl_c4cal_sp50_greedy.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from datasets import load_dataset

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
TEAL_ROOT  = SCRIPT_DIR.parent

sys.path.insert(0, str(TEAL_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from teal.model import LlamaSparseForCausalLM, LlamaSparseConfig          # noqa: E402
from teal.model import MistralSparseForCausalLM, MistralSparseConfig      # noqa: E402
from utils.utils import get_sparse_model, get_tokenizer                    # noqa: E402
from utils.eval_ppl import eval_ppl                                        # noqa: E402

AutoConfig.register("llama_sparse", LlamaSparseConfig)
AutoConfig.register("mistral_sparse", MistralSparseConfig)
AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)

# ── Constants ─────────────────────────────────────────────────────────────────

# Calibration dataset → model folder name (under TEAL_ROOT/models/)
CAL_MODEL_DIRS = {
    "alpaca":    "Llama-2-7B-alpaca",
    "c4":        "Llama-2-7B-c4",
    "wikitext2": "Llama-2-7B-wikitext2",
}

# Calibration dataset → tag used in output filenames
CAL_TAGS = {
    "alpaca":    "alphacal",
    "c4":        "c4cal",
    "wikitext2": "wikitext2cal",
}


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_inference_dataset(name: str, num_samples: int = 50):
    """
    Inference dataset load karo as list of {"text": ...} dicts.
    Calibration mein use hue samples automatically skip hote hain:
      alpaca    : train[300:]      (grab_acts used 0-299)
      c4        : validation[500:] (grab_acts_c4 used 0-499)
      wikitext2 : validation split (grab_acts_wikitext2 used entire test split)
    """
    if name == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True,
                          trust_remote_code=True)
        samples = [s for s in ds.skip(300).take(num_samples) if s.get("text", "").strip()]

    elif name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation",
                          trust_remote_code=True)
        samples = [{"text": s["text"]} for s in ds if s["text"].strip()]

    elif name == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True,
                          trust_remote_code=True)
        samples = [s for s in ds.skip(500).take(num_samples) if s.get("text", "").strip()]

    else:
        raise ValueError(f"Unknown dataset: {name!r}. Choose from: alpaca, wikitext2, c4")

    print(f"  Loaded {len(samples)} samples for '{name}'")
    return samples


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    datasets_list = list(CAL_MODEL_DIRS.keys())

    parser = argparse.ArgumentParser(
        description="PPL sweep: greedy TEAL on one calibration model × multiple inference datasets"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model name (weights source)"
    )
    parser.add_argument(
        "--cal_dataset", type=str, required=True, choices=datasets_list,
        help="Calibration model to use: alpaca | c4 | wikitext2"
    )
    parser.add_argument(
        "--inf_datasets", type=str, default="alpaca,c4,wikitext2",
        help="Comma-separated inference datasets (default: all three)"
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.5,
        help="Target sparsity for greedy lookup (default: 0.5)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=500,
        help="Samples from streaming datasets — alpaca and c4 (default: 500). "
             "wikitext2 uses full test set regardless."
    )
    args = parser.parse_args()

    inf_datasets = [d.strip() for d in args.inf_datasets.split(",")]
    sp_tag       = f"sp{int(args.sparsity * 100)}"
    cal_tag      = CAL_TAGS[args.cal_dataset]

    # Paths
    model_dir      = TEAL_ROOT / "models" / CAL_MODEL_DIRS[args.cal_dataset]
    histogram_path = model_dir / "histograms"
    lookup_path    = model_dir / "lookup"
    out_dir        = TEAL_ROOT / "analysis" / "data" / "PPL" / args.cal_dataset

    # Validate paths
    if not histogram_path.exists():
        raise FileNotFoundError(f"Histograms not found: {histogram_path}")
    if not lookup_path.exists():
        raise FileNotFoundError(
            f"Lookup not found: {lookup_path}\n"
            f"Run greedyopt.py for the {args.cal_dataset}-calibrated model first."
        )

    print(f"\n{'='*65}")
    print(f" PPL Sweep — Greedy TEAL")
    print(f" Calibration model  : {args.cal_dataset}  ({model_dir.name})")
    print(f" Inference datasets : {inf_datasets}")
    print(f" Sparsity           : {args.sparsity}")
    print(f" Num samples        : {args.num_samples}  (streaming datasets only)")
    print(f" Output dir         : {out_dir}")
    print(f"{'='*65}\n")

    # ── Load model once, reuse for all inference datasets ──
    print(f"Loading {args.cal_dataset}-calibrated sparse model...")
    tokenizer = get_tokenizer(args.model)
    model = get_sparse_model(
        args.model,
        device="auto",
        histogram_path=str(histogram_path),
    )
    model.load_greedy_sparsities(str(lookup_path), args.sparsity)
    model.eval()
    print("Model loaded + greedy sparsities set.\n")

    results = {}  # {inf_dataset: ppl}

    for inf_ds in inf_datasets:
        print(f"\n{'─'*55}")
        print(f"  Inference dataset: {inf_ds.upper()}")
        print(f"{'─'*55}")
        dataset = load_inference_dataset(inf_ds, args.num_samples)
        ppl = eval_ppl(model, tokenizer, device="cuda", dataset=dataset)
        results[inf_ds] = round(ppl, 4)
        print(f"  PPL ({inf_ds}) = {ppl:.4f}")

    # ── Print summary ──
    print(f"\n{'='*50}")
    print(f" RESULTS  (cal={args.cal_dataset}, {sp_tag}, greedy)")
    print(f"{'='*50}")
    print(f"  {'Inference dataset':<18}  PPL")
    print(f"  {'─'*30}")
    for inf_ds, ppl in results.items():
        print(f"  {inf_ds:<18}  {ppl:.4f}")

    # ── Save ──
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"ppl_{cal_tag}_{sp_tag}_greedy.json"
    csv_path  = out_dir / f"ppl_{cal_tag}_{sp_tag}_greedy.csv"

    with open(json_path, "w") as f:
        json.dump({
            "cal_dataset": args.cal_dataset,
            "sparsity":    args.sparsity,
            "num_samples": args.num_samples,
            "results":     results,
        }, f, indent=2)
    print(f"\nSaved JSON : {json_path}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["inf_dataset", "ppl"])
        for inf_ds, ppl in results.items():
            writer.writerow([inf_ds, ppl])
    print(f"Saved CSV  : {csv_path}")


if __name__ == "__main__":
    main()
