"""
Phase 3: Transfer Gain Analysis.

Compares aligned vs unaligned encoder transfer performance.
Produces: gain table, per-dataset bar chart, summary statistics.
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from feature2.evaluation.transfer_comparison import (
    compute_transfer_gain_table, print_gain_table
)
from feature2.evaluation.low_data_curves import (
    aggregate_low_data_results,
    plot_low_data_curves,
    print_degradation_table,
    compute_data_efficiency_score,
)


def plot_gain_bars(gain_table: dict, save_path: str):
    """Bar chart of transfer gain per alignment strategy."""
    alignments = list(gain_table.keys())
    datasets = list(next(iter(gain_table.values())).keys())
    strategies = list(
        next(iter(next(iter(gain_table.values())).values())).keys()
    )

    # Use first strategy for simplicity
    strategy = strategies[0]

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    colors = {"contrastive": "#3498DB", "domain": "#E74C3C", "prototype": "#2ECC71"}

    for ax, dataset in zip(axes, datasets):
        gains = []
        labels = []
        bar_colors = []

        for alignment in alignments:
            g = gain_table.get(alignment, {}).get(dataset, {}).get(strategy, {})
            if g:
                gains.append(g.get("absolute_gain", 0))
                labels.append(alignment)
                bar_colors.append(colors.get(alignment, "gray"))

        x = np.arange(len(labels))
        bars = ax.bar(x, gains, color=bar_colors, alpha=0.8, edgecolor="black")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(f"{dataset.upper()} [{strategy}]", fontsize=12)
        ax.set_ylabel("Δ ROC-AUC vs Unaligned", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

        for bar, gain in zip(bars, gains):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{gain:+.4f}",
                ha="center", va="bottom", fontsize=9,
            )

    plt.suptitle("Transfer Gain: Aligned vs Unaligned Encoder", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved gain bar chart: {save_path}")


def main():
    result_dir = "results/feature2/transfer_comparison"
    low_data_dir = "results/feature2/low_data"
    figures_dir = "results/feature2/figures"
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 70)
    print("Phase 3: Transfer Gain Analysis")
    print("=" * 70)

    # ── 1. Transfer comparison gain ───────────────────
    comparison_path = os.path.join(result_dir, "full_comparison.json")
    if os.path.exists(comparison_path):
        with open(comparison_path) as f:
            comparison = json.load(f)

        gain_table = compute_transfer_gain_table(comparison)
        print_gain_table(gain_table)

        bar_path = os.path.join(figures_dir, "transfer_gain_bars.png")
        try:
            plot_gain_bars(gain_table, bar_path)
        except Exception as e:
            print(f"  Bar chart failed: {e}")
    else:
        print(f"No comparison results found at {comparison_path}")
        print("Run: python feature2/scripts/run_transfer_pipeline.py first")

    # ── 2. Low-data learning curves ───────────────────
    for dataset_name in ["sider", "muv"]:
        print(f"\n--- Low-Data Curves: {dataset_name.upper()} ---")
        try:
            curve_data = aggregate_low_data_results(low_data_dir, dataset_name)
            if curve_data:
                print_degradation_table(curve_data, dataset_name)

                # Efficiency scores
                eff = compute_data_efficiency_score(curve_data)
                print(f"\nData Efficiency Scores ({dataset_name}):")
                for strat, score in sorted(eff.items(), key=lambda x: -x[1]):
                    print(f"  {strat:<30}: {score:.4f}")

                curve_path = os.path.join(
                    figures_dir, f"low_data_curves_{dataset_name}.png"
                )
                plot_low_data_curves(
                    curve_data, dataset_name, curve_path
                )
            else:
                print(f"  No low-data results for {dataset_name}")
                print("  Run: python feature2/scripts/run_low_data_experiment.py")
        except Exception as e:
            print(f"  Curves failed: {e}")

    print("\n" + "=" * 70)
    print("Phase 3 Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()