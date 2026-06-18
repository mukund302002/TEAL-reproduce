"""
Grouped bar chart of per-layer average absolute sparsity difference (%)
across multiple inference datasets.

For each dataset CSV:
  1. Compute |difference| * 100 for every (layer, projection) row
  2. For each of the 32 layers: average across all 7 projections (q,k,v,o,gate,up,down)
  3. Plot as grouped bars — one group per layer, one bar per dataset
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


DATASET_LABELS = {
    "alpaca":    "Alpaca",
    "wikitext2": "WikiText-2",
    "c4":        "C4",
}

COLORS = {
    "alpaca":    "#2196F3",   # blue
    "wikitext2": "#FF9800",   # orange
    "c4":        "#4CAF50",   # green
}


def per_layer_avg_abs_diff(csv_path: str) -> pd.Series:
    """
    Returns a Series of length 32 (index = layer 0..31).
    Each value = mean( |difference| * 100 ) across all 7 projections of that layer.
    """
    df = pd.read_csv(csv_path)
    df["abs_diff_pct"] = df["difference"].abs() * 100
    return df.groupby("layer")["abs_diff_pct"].mean()


def main():
    parser = argparse.ArgumentParser(
        description="Grouped bar chart of per-layer avg |sparsity difference| (%) across datasets."
    )
    parser.add_argument("--alpaca_csv",    type=str, required=True)
    parser.add_argument("--wikitext2_csv", type=str, required=True)
    parser.add_argument("--c4_csv",        type=str, required=True)
    parser.add_argument(
        "--output", type=str, default="sparsity_diff_barchart.png",
        help="Output image path (default: sparsity_diff_barchart.png)"
    )
    args = parser.parse_args()

    # ── Load per-layer values ──
    datasets = {
        "alpaca":    args.alpaca_csv,
        "wikitext2": args.wikitext2_csv,
        "c4":        args.c4_csv,
    }

    series = {}
    for key, path in datasets.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        series[key] = per_layer_avg_abs_diff(path)
        print(f"{DATASET_LABELS[key]:12s}  "
              f"mean={series[key].mean():.2f}%  "
              f"min={series[key].min():.2f}%  "
              f"max={series[key].max():.2f}%")

    # ── Grouped bar chart ──
    n_layers  = 32
    n_datasets = len(datasets)
    layers    = np.arange(n_layers)

    bar_width = 0.25
    offsets   = np.array([-1, 0, 1]) * bar_width   # centre each group on its layer tick

    fig, ax = plt.subplots(figsize=(18, 5))

    for i, (key, vals) in enumerate(series.items()):
        ax.bar(
            layers + offsets[i],
            vals.values,
            width=bar_width,
            color=COLORS[key],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
            label=f"{DATASET_LABELS[key]}",
        )

    ax.set_xlabel("Block index", fontsize=12)
    ax.set_ylabel("Avg |calibration − inference| sparsity  (%)", fontsize=12)
    # ax.set_title(
    #     "Per-layer sparsity gap: calibration vs decode-phase inference\n"
    #     "(each bar = mean across all 7 projections of that layer)",
    #     fontsize=13,
    # )
    ax.set_xticks(layers)
    ax.set_xticklabels(layers, fontsize=7)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    plt.savefig(args.output, dpi=150)
    print(f"\nPlot saved to: {args.output}")


if __name__ == "__main__":
    main()
