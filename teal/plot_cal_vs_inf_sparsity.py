"""
Plot calibration sparsity vs inference sparsity for TEAL and TopK methods.

TEAL (threshold-based):
  - Calibration sparsity: fraction zeroed on C4/Alpaca calibration data
  - Inference sparsity  : fraction actually zeroed at runtime (measured via hooks)
  - These may DIFFER because the fixed threshold responds to a new data distribution

TopK (exact-sparsity):
  - Calibration sparsity = Inference sparsity = target (always, by design)
  - Shown as a flat reference line

Input : analysis/data/sparsity_analysis_{dataset}_greedy.csv
         (columns: layer, projection, calibration_sparsity, inference_sparsity, difference, proj_type)

Usage:
  python plot_cal_vs_inf_sparsity.py --dataset alpaca
  python plot_cal_vs_inf_sparsity.py --dataset c4
  python plot_cal_vs_inf_sparsity.py --dataset wikitext2
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

SCRIPT_DIR   = Path(__file__).parent
ANALYSIS_DIR = SCRIPT_DIR.parent / "analysis"
DATA_DIR     = ANALYSIS_DIR / "data"
PLOT_DIR     = ANALYSIS_DIR / "plots"

PROJ_DISPLAY = {
    "k": "Attn-K", "o": "Attn-O", "q": "Attn-Q", "v": "Attn-V",
    "down": "MLP-Down", "gate": "MLP-Gate", "up": "MLP-Up",
}

BLOCK_COLORS = {
    "k": "#90CAF9", "q": "#1565C0", "v": "#42A5F5", "o": "#0D47A1",
    "gate": "#EF9A9A", "up": "#C62828", "down": "#E53935",
}


def load_csv(dataset: str) -> pd.DataFrame:
    path = DATA_DIR / f"sparsity_analysis_{dataset}_greedy.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def plot_per_layer_per_proj(df: pd.DataFrame, target_sparsity: float,
                             dataset: str, output_dir: Path):
    """
    One subplot per projection type (7 total).
    X-axis: layer (0-31), Y-axis: sparsity value.
    Lines: TEAL calibration, TEAL inference, TopK (flat at target).
    """
    proj_types = sorted(df["proj_type"].unique(), key=lambda x: PROJ_DISPLAY.get(x, x))
    n = len(proj_types)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5),
                             sharex=True)
    axes = axes.flatten()

    for ax_idx, proj in enumerate(proj_types):
        ax = axes[ax_idx]
        sub = df[df["proj_type"] == proj].sort_values("layer")

        layers   = sub["layer"].values
        cal_sp   = sub["calibration_sparsity"].values
        inf_sp   = sub["inference_sparsity"].values
        color    = BLOCK_COLORS.get(proj, "#999")

        ax.plot(layers, cal_sp, color=color,  linestyle="--",
                linewidth=1.5, label="TEAL calibration", alpha=0.9)
        ax.plot(layers, inf_sp, color=color,  linestyle="-",
                linewidth=1.5, label="TEAL inference",   alpha=0.9)
        ax.axhline(target_sparsity, color="green", linestyle=":",
                   linewidth=1.2, label=f"TopK (exact={target_sparsity})")

        ax.fill_between(layers, cal_sp, inf_sp,
                        alpha=0.12, color=color, label="gap")

        ax.set_title(PROJ_DISPLAY.get(proj, proj), fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Sparsity", fontsize=9)
        ax.set_xlabel("Layer", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

        if ax_idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    # Hide unused subplots
    for ax_idx in range(len(proj_types), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(
        f"Calibration vs Inference Sparsity per Projection — {dataset.upper()}\n"
        f"(TEAL threshold-based  vs  TopK exact-sparsity)",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    out = output_dir / f"cal_vs_inf_per_proj_{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_difference_heatmap(df: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Heatmap: rows = projection types, cols = layers.
    Cell value = inference_sparsity - calibration_sparsity.
    Red = inference > calibration (more sparse than expected)
    Blue = inference < calibration (less sparse than expected)
    """
    proj_types = sorted(df["proj_type"].unique(), key=lambda x: PROJ_DISPLAY.get(x, x))
    num_layers = df["layer"].max() + 1

    mat = np.full((len(proj_types), num_layers), np.nan)
    for r, proj in enumerate(proj_types):
        sub = df[df["proj_type"] == proj].sort_values("layer")
        for _, row in sub.iterrows():
            mat[r, int(row["layer"])] = row["difference"]

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-0.15, vmax=0.15)

    ax.set_yticks(range(len(proj_types)))
    ax.set_yticklabels([PROJ_DISPLAY.get(p, p) for p in proj_types], fontsize=9)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_xticks(range(0, num_layers, 4))
    ax.set_xticklabels(range(0, num_layers, 4), fontsize=8)

    plt.colorbar(im, ax=ax, label="Inference − Calibration sparsity",
                 fraction=0.03, pad=0.02)

    ax.set_title(
        f"TEAL: Sparsity Gap (Inference − Calibration) — {dataset.upper()}\n"
        f"TopK gap = 0.0 by design (not shown)",
        fontsize=12
    )
    plt.tight_layout()

    out = output_dir / f"sparsity_gap_heatmap_{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def print_summary(df: pd.DataFrame, target_sparsity: float):
    print("\n" + "=" * 70)
    print("SPARSITY SUMMARY (TEAL greedy)")
    print("=" * 70)
    print(f"{'Projection':<12} {'Cal mean':>10} {'Inf mean':>10} {'Diff mean':>12} {'Diff std':>10}")
    print("-" * 70)
    for proj in sorted(df["proj_type"].unique()):
        sub = df[df["proj_type"] == proj]
        cal_m  = sub["calibration_sparsity"].mean()
        inf_m  = sub["inference_sparsity"].mean()
        diff_m = sub["difference"].mean()
        diff_s = sub["difference"].std()
        print(f"{PROJ_DISPLAY.get(proj, proj):<12} {cal_m:>10.4f} {inf_m:>10.4f} {diff_m:>+12.4f} {diff_s:>10.4f}")

    print("-" * 70)
    print(f"{'TopK (all projs)':<12} {target_sparsity:>10.4f} {target_sparsity:>10.4f} {'0.0000':>12} {'0.0000':>10}")
    print("=" * 70)
    print(f"\nNote: TopK gap is always 0 by design (dynamic quantile per call).")


def main():
    parser = argparse.ArgumentParser(
        description="Plot calibration vs inference sparsity for TEAL and TopK"
    )
    parser.add_argument("--dataset", type=str, default="alpaca",
                        choices=["alpaca", "c4", "wikitext2"],
                        help="Which inference dataset CSV to load")
    parser.add_argument("--target_sparsity", type=float, default=0.5,
                        help="TopK target sparsity (flat reference line)")
    args = parser.parse_args()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_csv(args.dataset)
    print(f"Loaded {len(df)} rows from {args.dataset} CSV.")

    plot_per_layer_per_proj(df, args.target_sparsity, args.dataset, PLOT_DIR)
    plot_difference_heatmap(df, args.dataset, PLOT_DIR)
    print_summary(df, args.target_sparsity)

    print(f"\nAll plots saved to: {PLOT_DIR}/")


if __name__ == "__main__":
    main()
