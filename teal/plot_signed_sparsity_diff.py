"""
Plot signed sparsity difference (inference - calibration) per block.

Positive  → inference is MORE sparse than calibration (over-sparsifying)
Negative  → inference is LESS sparse than calibration (under-sparsifying / conservative)

Usage:
  python plot_signed_sparsity_diff.py --dataset alpaca
  python plot_signed_sparsity_diff.py --dataset wikitext2
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
ANALYSIS_DIR = SCRIPT_DIR.parent / "analysis"
CAL_DATA_DIR = ANALYSIS_DIR / "data" / "module_wise_sparsity" / "calibration"

# {cal_dataset: {inf_dataset: csv_path}}
DATASET_FILES = {
    "alpaca": {
        "alpaca":    CAL_DATA_DIR / "alpaca"    / "sparsity_analysis_alphacal_alpaca_greedy.csv",
        "c4":        CAL_DATA_DIR / "alpaca"    / "sparsity_analysis_alphacal_c4_greedy.csv",
        "wikitext2": CAL_DATA_DIR / "alpaca"    / "sparsity_analysis_alphacal_wikitext2_greedy.csv",
    },
    "c4": {
        "alpaca":    CAL_DATA_DIR / "c4"        / "sparsity_analysis_c4cal_alpaca_greedy.csv",
        "c4":        CAL_DATA_DIR / "c4"        / "sparsity_analysis_c4cal_c4_greedy.csv",
        "wikitext2": CAL_DATA_DIR / "c4"        / "sparsity_analysis_c4cal_wikitext2_greedy.csv",
    },
    "wikitext2": {
        "alpaca":    CAL_DATA_DIR / "wikitext2" / "sparsity_analysis_wikitext2cal_alpaca_greedy.csv",
        "c4":        CAL_DATA_DIR / "wikitext2" / "sparsity_analysis_wikitext2cal_c4_greedy.csv",
        "wikitext2": CAL_DATA_DIR / "wikitext2" / "sparsity_analysis_wikitext2cal_wikitext2_greedy.csv",
    },
}

PROJ_GROUPS = {
    "MLP":      ["mlp.gate", "mlp.up", "mlp.down"],
    "Attention": ["attn.q",  "attn.k", "attn.v", "attn.o"],
}

# Same as greedyopt.py — parameter-count-based weights per projection
WEIGHT_DICT = {
    "Llama-3-8B":  {'q': 1, 'k': 1/4,   'v': 1/4,   'o': 1, 'gate': 3.5,    'up': 3.5,    'down': 3.5},
    "Llama-3-70B": {'q': 1, 'k': 1/8,   'v': 1/8,   'o': 1, 'gate': 3.5,    'up': 3.5,    'down': 3.5},
    "Llama-2-7B":  {'q': 1, 'k': 1/8,   'v': 1/8,   'o': 1, 'gate': 2.6875, 'up': 2.6875, 'down': 2.6875},
    "Llama-2-13B": {'q': 1, 'k': 1/8,   'v': 1/8,   'o': 1, 'gate': 2.7,    'up': 2.7,    'down': 2.7},
    "Llama-2-70B": {'q': 1, 'k': 1/8,   'v': 1/8,   'o': 1, 'gate': 3.5,    'up': 3.5,    'down': 3.5},
    "Mistral-7B":  {'q': 1, 'k': 1/8,   'v': 1/8,   'o': 1, 'gate': 3.5,    'up': 3.5,    'down': 3.5},
}


def weighted_mean(g, weights):
    """Weighted mean of g['difference'] using greedyopt weights."""
    total_w = g['weight'].sum()
    return (g['difference'] * g['weight']).sum() / total_w


def plot_signed_diff(df, cal_dataset, inf_dataset, out_dir, weights):
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = sorted(df["layer"].unique())

    # ── attach per-projection weights (same as greedyopt) ──────────────
    df = df.copy()
    df['proj_type'] = df['projection'].str.split('.').str[-1]
    df['weight']    = df['proj_type'].map(weights)

    # ── per-block weighted averages ─────────────────────────────────────
    block_avg   = df.groupby("layer").apply(weighted_mean, weights=weights)

    mlp_projs   = PROJ_GROUPS["MLP"]
    attn_projs  = PROJ_GROUPS["Attention"]
    block_mlp   = df[df["projection"].isin(mlp_projs)].groupby("layer").apply(weighted_mean, weights=weights)
    block_attn  = df[df["projection"].isin(attn_projs)].groupby("layer").apply(weighted_mean, weights=weights)

    # ── Plot 1: Overall signed difference per block ─────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in block_avg]
    ax.bar(layers, block_avg.values, color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(15.5, color="gray", linewidth=1.2, linestyle="--", label="Block 16 boundary")

    # shade regions
    ax.axvspan(-0.5, 15.5, alpha=0.04, color="red",  label="Early blocks (0-15)")
    ax.axvspan(15.5, 31.5, alpha=0.04, color="blue", label="Late blocks (16-31)")

    over  = mpatches.Patch(color="#d62728", label="Over-sparsifying (inference > calibration)")
    under = mpatches.Patch(color="#1f77b4", label="Under-sparsifying (inference < calibration)")
    ax.legend(handles=[over, under], fontsize=8, loc="upper right")

    ax.set_xlabel("Block (Layer) Index", fontsize=11)
    ax.set_ylabel("Avg signed difference\n(inference − calibration)", fontsize=11)
    ax.set_title(f"Signed Sparsity Difference per Block — cal: {cal_dataset}  inf: {inf_dataset}", fontsize=12)
    ax.set_xticks(layers)
    ax.set_xticklabels(layers, fontsize=7)
    plt.tight_layout()
    out = out_dir / f"signed_diff_overall_{inf_dataset}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Plot 2: MLP vs Attention split ──────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for ax, data, title, cpos, cneg in zip(
        axes,
        [block_mlp, block_attn],
        ["MLP projections (gate / up / down)", "Attention projections (q / k / v / o)"],
        ["#e6550d", "#31a354"],
        ["#6baed6", "#9ecae1"],
    ):
        cols = [cpos if v > 0 else cneg for v in data.values]
        ax.bar(layers, data.values, color=cols, edgecolor="white", linewidth=0.4)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(15.5, color="gray", linewidth=1.2, linestyle="--")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Signed diff", fontsize=9)
        ax.set_xticks(layers)
        ax.set_xticklabels(layers, fontsize=7)

    axes[-1].set_xlabel("Block (Layer) Index", fontsize=11)
    fig.suptitle(f"Signed Sparsity Difference: MLP vs Attention — cal: {cal_dataset}  inf: {inf_dataset}", fontsize=12)
    plt.tight_layout()
    out = out_dir / f"signed_diff_mlp_attn_{inf_dataset}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    
    # ── Print summary statistics (weighted) ─────────────────────────────
    early_layers = block_avg[block_avg.index <= 15]
    late_layers  = block_avg[block_avg.index >  15]

    early_raw = df[df["layer"] <= 15]
    late_raw  = df[df["layer"] >  15]

    print(f"\n── cal: {cal_dataset}  inf: {inf_dataset}  Summary (greedyopt-weighted) ──")
    print(f"Early blocks (0-15)  | mean={early_layers.mean():+.4f}  std={early_layers.std():.4f}  "
          f"over={( early_raw['difference']>0).sum()}/{len(early_raw)}  "
          f"under={(early_raw['difference']<0).sum()}/{len(early_raw)}")
    print(f"Late  blocks (16-31) | mean={late_layers.mean():+.4f}  std={late_layers.std():.4f}  "
          f"over={(late_raw['difference']>0).sum()}/{len(late_raw)}  "
          f"under={(late_raw['difference']<0).sum()}/{len(late_raw)}")


def main():
    datasets = list(DATASET_FILES.keys())
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal_dataset", type=str, default="alpaca", choices=datasets,
                        help="Calibration dataset (which model's histograms were used)")
    parser.add_argument("--inf_dataset", type=str, default="alpaca", choices=datasets,
                        help="Inference dataset (which dataset was used during generation)")
    parser.add_argument("--model_type", type=str, default="Llama-2-7B",
                        choices=list(WEIGHT_DICT.keys()),
                        help="Model type for weighted sparsity (must match WEIGHT_DICT key)")
    args = parser.parse_args()

    csv_path = DATASET_FILES[args.cal_dataset][args.inf_dataset]
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = ANALYSIS_DIR / "plots" / "new_signed_sparsity_diff" / "calibration" / args.cal_dataset
    df = pd.read_csv(csv_path)
    plot_signed_diff(df, args.cal_dataset, args.inf_dataset, out_dir, weights=WEIGHT_DICT[args.model_type])


if __name__ == "__main__":
    main()
