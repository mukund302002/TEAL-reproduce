"""
Plot TEAL (threshold-based) vs TopK (exact-sparsity) comparison results.

Reads comparison_summary_sp{XX}.csv files from output_dir and produces:
  1. Bar chart: per-task accuracy — Dense vs TEAL vs TopK
  2. Bar chart: accuracy delta vs Dense — TEAL vs TopK (positive = better than dense)

Usage:
  python plot_teal_comparison.py \\
      --results_dir ./results/teal_comparison \\
      --sparsity 0.5 \\
      --output_dir ./plots/teal_comparison
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


# Task display names (shorter for axis labels)
TASK_LABELS = {
    "arc_easy/acc_norm,none":      "ARC-Easy",
    "arc_challenge/acc_norm,none": "ARC-Challenge",
    "hellaswag/acc_norm,none":     "HellaSwag",
    "piqa/acc,none":               "PIQA",
    "winogrande/acc,none":         "WinoGrande",
    "mmlu/acc,none":               "MMLU",
    "gsm8k/exact_match,strict-match": "GSM8K",
    # fallbacks (some lm_eval versions use different keys)
    "arc_easy/acc,none":           "ARC-Easy",
    "arc_challenge/acc,none":      "ARC-Challenge",
}

METHOD_COLORS = {
    "dense":   "#4CAF50",   # green
    "teal":    "#2196F3",   # blue
    "topk":    "#FF9800",   # orange
}

METHOD_LABELS = {
    "dense": "Dense",
    "teal":  "TEAL (threshold)",
    "topk":  "TopK (exact)",
}


def load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    return df


def get_method_key(col: str, sp_tag: str) -> str:
    """Map CSV column name to short method key (dense / teal / topk)."""
    if col == "dense":
        return "dense"
    if col.startswith(f"teal_{sp_tag}"):
        return "teal"
    if col.startswith(f"topk_{sp_tag}"):
        return "topk"
    return col


def extract_accuracy_rows(df: pd.DataFrame, sp_tag: str):
    """
    Filter rows that look like accuracy/acc_norm metrics and
    return a tidy dict: { task_label: { method: value } }
    """
    task_data = {}
    for metric_key, row in df.iterrows():
        # Only keep accuracy-type metrics; skip perplexity / word_perplexity
        if "acc" not in str(metric_key) and "exact_match" not in str(metric_key):
            continue
        label = TASK_LABELS.get(metric_key)
        if label is None:
            # Try partial match
            for k, v in TASK_LABELS.items():
                if k.split("/")[0] in metric_key:
                    label = v
                    break
        if label is None:
            label = metric_key.split("/")[0]  # fallback: use raw task name

        method_vals = {}
        for col in df.columns:
            method = get_method_key(col, sp_tag)
            try:
                method_vals[method] = float(row[col])
            except (ValueError, TypeError):
                pass
        if method_vals:
            task_data[label] = method_vals

    return task_data


def plot_accuracy_bars(task_data: dict, sparsity: float, output_path: Path):
    """Grouped bar chart: absolute accuracy per task."""
    tasks   = sorted(task_data.keys())
    methods = ["dense", "teal", "topk"]
    methods = [m for m in methods if any(m in td for td in task_data.values())]

    n_tasks   = len(tasks)
    n_methods = len(methods)
    x = np.arange(n_tasks)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, n_tasks * 1.5), 5))

    for i, method in enumerate(methods):
        vals = [task_data[t].get(method, np.nan) for t in tasks]
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width,
                      label=METHOD_LABELS.get(method, method),
                      color=METHOD_COLORS.get(method, "#999"),
                      alpha=0.85, edgecolor="white")

        # Annotate bar tops
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_title(
        f"TEAL vs TopK — Accuracy per Task\n"
        f"(LLaMA-2-7B, {int(sparsity*100)}% activation sparsity)",
        fontsize=12
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_delta_bars(task_data: dict, sparsity: float, output_path: Path):
    """
    Delta bar chart: accuracy change vs Dense baseline.
    Positive = better than dense (unusual for sparsity), negative = degradation.
    """
    tasks   = sorted(task_data.keys())
    methods = ["teal", "topk"]
    methods = [m for m in methods if any(m in td for td in task_data.values())]

    n_tasks   = len(tasks)
    n_methods = len(methods)
    x = np.arange(n_tasks)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, n_tasks * 1.5), 5))

    for i, method in enumerate(methods):
        deltas = []
        for t in tasks:
            dense_v = task_data[t].get("dense", np.nan)
            m_v     = task_data[t].get(method, np.nan)
            deltas.append(m_v - dense_v if not np.isnan(dense_v) and not np.isnan(m_v) else np.nan)

        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, deltas, width,
                      label=METHOD_LABELS.get(method, method),
                      color=METHOD_COLORS.get(method, "#999"),
                      alpha=0.85, edgecolor="white")

        for bar, v in zip(bars, deltas):
            if not np.isnan(v):
                va = "bottom" if v >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.002 if v >= 0 else -0.002),
                        f"{v:+.3f}", ha="center", va=va, fontsize=7, rotation=45)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Accuracy delta vs Dense", fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax.set_title(
        f"TEAL vs TopK — Accuracy Change vs Dense Baseline\n"
        f"(LLaMA-2-7B, {int(sparsity*100)}% activation sparsity)",
        fontsize=12
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot TEAL vs TopK comparison"
    )
    parser.add_argument("--results_dir", type=str,
                        default="./results/teal_comparison")
    parser.add_argument("--sparsity",    type=float, default=0.5)
    parser.add_argument("--output_dir",  type=str,
                        default="./plots/teal_comparison")
    args = parser.parse_args()

    sp_tag      = f"sp{int(args.sparsity * 100)}"
    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)

    csv_path = results_dir / f"comparison_summary_{sp_tag}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Summary CSV not found: {csv_path}\n"
            "Run eval_teal_comparison.py first."
        )

    print(f"Loading: {csv_path}")
    df = load_summary(csv_path)
    print(df.to_string())

    task_data = extract_accuracy_rows(df, sp_tag)
    if not task_data:
        print("No accuracy metrics found in CSV. Check metric key names.")
        return

    plot_accuracy_bars(
        task_data, args.sparsity,
        output_dir / f"accuracy_bars_{sp_tag}.png"
    )
    plot_delta_bars(
        task_data, args.sparsity,
        output_dir / f"delta_bars_{sp_tag}.png"
    )

    print(f"\nAll plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
