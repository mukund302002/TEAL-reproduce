"""
Module-wise calibration vs TG-TopK inference sparsity comparison.

plot_module_sparsity_diff.py ka TG-TopK experiment variant.

Data source : analysis/data/module_wise_sparsity/tg_topk/{cal_dataset}/
Output      : analysis/plots/module_wise_sparsity_diff/tg_topk/{cal_dataset}/

4 plots (same as original):
  1. 7-panel line plot  — cal (black) vs tg-topk inference (coloured) per projection
  2. Heatmaps           — cal + 3 inf datasets, same 0–1 color scale
  3. Diff barchart      — (inf − cal) per layer per projection
  4. Cal sparsity bar   — mean calibration sparsity per module

Usage:
  python teal/plot_module_sparsity_diff_tg_topk.py --cal_dataset alpaca
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

SCRIPT_DIR    = Path(__file__).parent
ANALYSIS_DIR  = SCRIPT_DIR.parent / "analysis"
TG_DATA_DIR   = ANALYSIS_DIR / "data" / "module_wise_sparsity" / "tg_topk"

INF_DATASETS = ["alpaca", "c4", "wikitext2"]

PROJ_ORDER = ["attn.q", "attn.k", "attn.v", "attn.o", "mlp.gate", "mlp.up", "mlp.down"]

INF_COLORS = {
    "alpaca":    "#1f77b4",
    "c4":        "#d62728",
    "wikitext2": "#2ca02c",
}
INF_STYLES = {
    "alpaca":    "-",
    "c4":        "--",
    "wikitext2": ":",
}


# ── Data loader ───────────────────────────────────────────────────────────────

def load_data(cal_dataset: str) -> dict[str, pd.DataFrame]:
    data = {}
    for inf in INF_DATASETS:
        path = TG_DATA_DIR / cal_dataset / f"sparsity_analysis_tg_topk_{cal_dataset}cal_{inf}_greedy.csv"
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        df["layer"] = df["layer"].astype(int)
        data[inf] = df.sort_values(["layer", "projection"]).reset_index(drop=True)
    return data


# ── Figure 1: 7-panel line plot ───────────────────────────────────────────────

def plot_cal_vs_inf(data: dict[str, pd.DataFrame], cal_dataset: str, out_dir: Path):
    """
    7 subplots — one per projection.
    Black thick line  = calibration sparsity (from TEAL histogram threshold).
    Coloured lines    = TG-TopK inference sparsity per eval dataset.
    """
    ref_df = data[INF_DATASETS[0]]
    layers = sorted(ref_df["layer"].unique())

    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    axes_flat = axes.flatten()

    for ax_idx, proj in enumerate(PROJ_ORDER):
        ax = axes_flat[ax_idx]

        # ── calibration sparsity (thick black) ──
        cal_rows = ref_df[ref_df["projection"] == proj].sort_values("layer")
        ax.plot(
            cal_rows["layer"], cal_rows["calibration_sparsity"],
            color="black", linewidth=2.2, linestyle="-",
            marker="s", markersize=2.8, label="calibration", zorder=5
        )

        # ── TG-TopK inference sparsity per dataset ──
        for inf in INF_DATASETS:
            df   = data[inf]
            rows = df[df["projection"] == proj].sort_values("layer")
            ax.plot(
                rows["layer"], rows["inference_sparsity"],
                color=INF_COLORS[inf],
                linestyle=INF_STYLES[inf],
                linewidth=1.3,
                marker="o", markersize=2.2,
                label=f"tg-topk inf={inf}", alpha=0.9
            )

        ax.axvline(15.5, color="gray", linewidth=1.0, linestyle="--", alpha=0.5)
        ax.set_title(proj, fontsize=11, fontweight="bold")
        ax.set_ylabel("Sparsity", fontsize=8)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(layers[::4])
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.25)

    axes_flat[-1].set_visible(False)

    handles = [
        mlines.Line2D([], [], color="black",                 linewidth=2.2, linestyle="-",  marker="s", markersize=4, label="calibration (TEAL threshold)"),
        mlines.Line2D([], [], color=INF_COLORS["alpaca"],    linewidth=1.3, linestyle="-",  marker="o", markersize=3, label="tg-topk inf=alpaca"),
        mlines.Line2D([], [], color=INF_COLORS["c4"],        linewidth=1.3, linestyle="--", marker="o", markersize=3, label="tg-topk inf=c4"),
        mlines.Line2D([], [], color=INF_COLORS["wikitext2"], linewidth=1.3, linestyle=":",  marker="o", markersize=3, label="tg-topk inf=wikitext2"),
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=10,
               title="Sparsity source", title_fontsize=10,
               bbox_to_anchor=(0.98, 0.03))

    fig.suptitle(
        f"Module-wise Calibration vs TG-TopK Inference Sparsity  (cal={cal_dataset})\n"
        "Black = calibration threshold sparsity  |  Coloured = TG-TopK inference sparsity",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0.0, 1, 0.95])

    out = out_dir / f"module_cal_vs_inf_tg_topk_{cal_dataset}cal.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 2: Heatmaps ────────────────────────────────────────────────────────

def plot_heatmaps(data: dict[str, pd.DataFrame], cal_dataset: str, out_dir: Path):
    """
    4 panels: calibration | tg-topk inf=alpaca | tg-topk inf=c4 | tg-topk inf=wikitext2
    Same 0–1 color scale across all panels.
    """
    ref_df = data[INF_DATASETS[0]]
    layers = sorted(ref_df["layer"].unique())

    fig, axes = plt.subplots(1, 4, figsize=(24, 4))

    panel_specs = [("calibration (TEAL thr)", None)] + [
        (f"tg-topk inf={inf}", inf) for inf in INF_DATASETS
    ]

    for ax, (title, inf_key) in zip(axes, panel_specs):
        mat = np.full((len(PROJ_ORDER), len(layers)), np.nan)

        df       = ref_df if inf_key is None else data[inf_key]
        col_name = "calibration_sparsity" if inf_key is None else "inference_sparsity"

        for r, proj in enumerate(PROJ_ORDER):
            rows = df[df["projection"] == proj].sort_values("layer")
            for c, layer in enumerate(layers):
                val = rows[rows["layer"] == layer][col_name].values
                if len(val):
                    mat[r, c] = val[0]

        im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Sparsity", shrink=0.85)

        ax.set_yticks(range(len(PROJ_ORDER)))
        ax.set_yticklabels(PROJ_ORDER, fontsize=9)
        ax.set_xticks(range(0, len(layers), 4))
        ax.set_xticklabels(layers[::4], fontsize=8)
        ax.set_xlabel("Layer index", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axvline(15.5, color="white", linewidth=1.5, linestyle="--", alpha=0.7)

    fig.suptitle(
        f"Module-wise Sparsity Heatmap  (cal={cal_dataset})  —  TG-TopK experiment\n"
        "Same color scale (0–1) across all panels",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()

    out = out_dir / f"module_sparsity_heatmap_tg_topk_{cal_dataset}cal.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 3: Difference barchart ────────────────────────────────────────────

def plot_diff_barchart(data: dict[str, pd.DataFrame], cal_dataset: str, out_dir: Path,
                       inf_dataset: str = "alpaca"):
    """
    7 subplots — one per projection.
    Bar: X = layer index, Y = tg-topk inference − calibration sparsity.
    Green = inference more sparse, Red = inference less sparse.
    """
    df     = data[inf_dataset]
    layers = sorted(df["layer"].unique())

    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    axes_flat = axes.flatten()

    for ax_idx, proj in enumerate(PROJ_ORDER):
        ax = axes_flat[ax_idx]

        rows  = df[df["projection"] == proj].sort_values("layer")
        diffs = rows["difference"].values

        colors = ["#2ca02c" if d >= 0 else "#d62728" for d in diffs]
        ax.bar(layers, diffs, color=colors, edgecolor="white", linewidth=0.4, width=0.75)
        ax.axhline(0, color="black", linewidth=1.0)

        mean_val = diffs.mean()
        ax.axhline(mean_val, color="navy", linewidth=1.3, linestyle="--",
                   label=f"mean={mean_val:+.4f}")

        ax.set_title(proj, fontsize=11, fontweight="bold")
        ax.set_ylabel("tg-topk inf − cal", fontsize=8)
        ax.set_xticks(layers[::4])
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.25)

    axes_flat[-1].set_visible(False)
    axes_flat[-2].set_xlabel("Layer index", fontsize=9)

    fig.suptitle(
        f"Per-layer (TG-TopK inference − calibration) Sparsity  "
        f"[cal={cal_dataset}, inf={inf_dataset}]\n"
        "Green = inference more sparse  |  Red = inference less sparse  |  "
        "Blue dashed = mean",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = out_dir / f"module_diff_barchart_tg_topk_{cal_dataset}cal_{inf_dataset}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 4: Average calibration sparsity per module ────────────────────────

def plot_cal_sparsity_per_module(data: dict[str, pd.DataFrame], cal_dataset: str, out_dir: Path):
    """
    Bar chart: X = module, Y = mean calibration sparsity across 32 layers.
    Error bars = std. (Calibration values same as original TEAL — yahan sirf reference ke liye.)
    """
    ref_df = data[INF_DATASETS[0]]

    means, stds = [], []
    for proj in PROJ_ORDER:
        vals = ref_df[ref_df["projection"] == proj]["calibration_sparsity"]
        means.append(vals.mean())
        stds.append(vals.std())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(PROJ_ORDER, means, yerr=stds, capsize=5,
                  color="#4c72b0", edgecolor="white", linewidth=0.6,
                  error_kw={"elinewidth": 1.4, "ecolor": "gray"})

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0.5, color="red", linewidth=1.2, linestyle="--", label="target sparsity = 0.5")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean calibration sparsity (across 32 layers)", fontsize=10)
    ax.set_xlabel("Module", fontsize=10)
    ax.set_title(
        f"Average Calibration Sparsity per Module  (cal={cal_dataset})\n"
        "Error bars = std across layers  |  same as TEAL threshold values",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = out_dir / f"cal_sparsity_per_module_tg_topk_{cal_dataset}cal.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    print(f"\n{'Module':<14}  {'Mean':>8}  {'Std':>8}")
    print("─" * 34)
    for proj, m, s in zip(PROJ_ORDER, means, stds):
        print(f"{proj:<14}  {m:>8.4f}  {s:>8.4f}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(data: dict[str, pd.DataFrame], cal_dataset: str):
    ref_df = data[INF_DATASETS[0]]
    print(f"\n{'='*90}")
    print(f" Module-wise Sparsity Summary — TG-TopK  (cal={cal_dataset})")
    print(f"{'='*90}")
    print(f"{'Projection':<14}  {'Cal (mean)':>12}  " +
          "  ".join(f"TG-TopK inf={inf} (mean)" for inf in INF_DATASETS))
    print("─" * 90)

    for proj in PROJ_ORDER:
        cal_vals = ref_df[ref_df["projection"] == proj]["calibration_sparsity"]
        row = f"{proj:<14}  {cal_vals.mean():>12.4f}"
        for inf in INF_DATASETS:
            inf_vals = data[inf][data[inf]["projection"] == proj]["inference_sparsity"]
            row += f"  {inf_vals.mean():>22.4f}"
        print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Module-wise sparsity plots for TG-TopK experiment"
    )
    parser.add_argument(
        "--cal_dataset", type=str, default="alpaca",
        choices=["alpaca", "c4", "wikitext2"],
        help="Calibration dataset (default: alpaca)"
    )
    args = parser.parse_args()

    out_dir = ANALYSIS_DIR / "plots" / "module_wise_sparsity_diff" / "tg_topk" / args.cal_dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TG-TopK CSVs for cal={args.cal_dataset}...")
    data = load_data(args.cal_dataset)
    print(f"  Loaded {len(INF_DATASETS)} inference datasets.\n")

    plot_cal_vs_inf(data, args.cal_dataset, out_dir)
    plot_heatmaps(data, args.cal_dataset, out_dir)
    for inf in INF_DATASETS:
        plot_diff_barchart(data, args.cal_dataset, out_dir, inf_dataset=inf)
    plot_cal_sparsity_per_module(data, args.cal_dataset, out_dir)
    print_summary(data, args.cal_dataset)


if __name__ == "__main__":
    main()
