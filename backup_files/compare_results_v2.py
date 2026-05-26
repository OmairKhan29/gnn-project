"""
scripts/compare_results_v2.py
Comprehensive results comparison using actual ablation JSON data.
Handles the actual result format from run_ablations.py output.

Usage:
    python scripts/compare_results_v2.py
"""
import json
import os
import sys
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

sys.path.insert(0, ".")

matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_strategy_results(
    path: str = "results/ablations/ablation_strategy.json",
) -> Dict:
    """Load and parse strategy ablation results."""
    with open(path) as f:
        return json.load(f)


def compute_per_task_stats(
    strategy_data: Dict,
) -> Dict[str, Dict[str, Dict]]:
    """
    Compute mean ± std per task across seeds for each strategy.

    Returns
    -------
    dict[strategy -> dict[task_name -> {mean, std, values}]]
    """
    stats = {}

    for strategy, result in strategy_data["results"].items():
        stats[strategy] = {}
        per_seed = result["per_seed"]

        # Collect all task names
        all_tasks = set()
        for seed_r in per_seed:
            all_tasks.update(seed_r["test_auc_per_task"].keys())

        for task in sorted(all_tasks):
            values = [
                s["test_auc_per_task"][task]
                for s in per_seed
                if task in s["test_auc_per_task"]
            ]
            stats[strategy][task] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }

    return stats


def print_full_results_table(
    strategy_results: Dict,
    per_task_stats: Dict,
):
    """Print comprehensive results table."""
    strategies = list(strategy_results["results"].keys())
    tasks = sorted(next(iter(per_task_stats.values())).keys())

    header = f"{'Task':<30}"
    for s in strategies:
        short_s = s.replace("task_conditioned", "TC").replace("_pcgrad", "+PCG").replace("hard_sharing", "HS")
        header += f"  {short_s:>22}"
    print("\n" + "=" * 100)
    print("PER-TASK ROC-AUC RESULTS (mean ± std across 3 seeds)")
    print("=" * 100)
    print(header)
    print("-" * 100)

    for task in tasks:
        short_task = "_".join(task.split("_")[1:])
        row = f"{short_task:<30}"

        # Find best strategy for this task
        task_means = {
            s: per_task_stats[s][task]["mean"]
            for s in strategies
            if task in per_task_stats[s]
        }
        best_strategy = max(task_means, key=task_means.get)

        for strategy in strategies:
            if task not in per_task_stats[strategy]:
                row += f"  {'N/A':>22}"
                continue

            m = per_task_stats[strategy][task]["mean"]
            s = per_task_stats[strategy][task]["std"]

            cell = f"{m:.4f} ± {s:.4f}"
            if strategy == best_strategy:
                cell = f"*{cell}*"  # Mark best

            row += f"  {cell:>22}"

        print(row)

    # Summary row
    print("-" * 100)
    summary = f"{'AVERAGE':<30}"
    for strategy in strategies:
        means = [per_task_stats[strategy][t]["mean"] for t in tasks
                 if t in per_task_stats[strategy]]
        avg = np.mean(means)
        std = np.std(means)
        cell = f"{avg:.4f} ± {std:.4f}"
        summary += f"  {cell:>22}"
    print(summary)
    print("=" * 100)

    # Count wins per strategy
    print("\nSTRATEGY WIN COUNT (best AUC per task):")
    wins = {s: 0 for s in strategies}

    for task in tasks:
        task_means = {
            s: per_task_stats[s][task]["mean"]
            for s in strategies
            if task in per_task_stats[s]
        }
        if task_means:
            winner = max(task_means, key=task_means.get)
            wins[winner] += 1

    for strategy, count in wins.items():
        short = strategy.replace("task_conditioned", "TC").replace("_pcgrad", "+PCG")
        pct = 100 * count / len(tasks)
        print(f"  {short}: {count}/{len(tasks)} tasks ({pct:.1f}%)")


def generate_paper_figure(
    strategy_results: Dict,
    per_task_stats: Dict,
    output_dir: str = "results/figures",
):
    """Generate publication-ready strategy comparison figure."""
    os.makedirs(output_dir, exist_ok=True)

    strategies = list(strategy_results["results"].keys())
    tasks = sorted(next(iter(per_task_stats.values())).keys())
    short_tasks = ["_".join(t.split("_")[1:]) for t in tasks]

    # ── Figure 1: Overall performance comparison ──────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    strategy_labels = ["Hard Sharing", "Task-Conditioned", "TC + PCGrad"]

    # (a) Overall AUC bar chart
    ax = axes[0]
    strategy_data = strategy_results["results"]

    for i, (strategy, label, color) in enumerate(zip(strategies, strategy_labels, colors)):
        m = strategy_data[strategy]["mean_auc"]
        s = strategy_data[strategy]["std_auc"]
        bar = ax.bar(i, m, color=color, edgecolor="black", linewidth=0.5,
                     alpha=0.85, yerr=s, capsize=5)
        ax.text(i, m + s + 0.002, f"{m:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategy_labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("(a) Average Performance")
    ax.grid(axis="y", alpha=0.3)
    y_min = min(strategy_data[s]["mean_auc"] for s in strategies) - 0.01
    y_max = max(strategy_data[s]["mean_auc"] for s in strategies) + 0.015
    ax.set_ylim(y_min, y_max)

    # (b) Per-task heatmap
    ax = axes[1]
    matrix = np.array([
        [per_task_stats[s][t]["mean"] for t in tasks]
        for s in strategies
    ])

    sns.heatmap(
        matrix,
        xticklabels=[t.replace("tox21_", "").replace("clintox_", "ct_")
                     for t in short_tasks],
        yticklabels=strategy_labels,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        annot_kws={"fontsize": 6},
        linewidths=0.3,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("(b) Per-Task ROC-AUC")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=8)

    # (c) Delta over hard sharing
    ax = axes[2]
    baseline_means = {
        t: per_task_stats["hard_sharing"][t]["mean"]
        for t in tasks
        if t in per_task_stats["hard_sharing"]
    }

    for strategy, label, color in zip(
        strategies[1:], strategy_labels[1:], colors[1:]
    ):
        deltas = [
            per_task_stats[strategy][t]["mean"] - baseline_means[t]
            for t in tasks
            if t in per_task_stats[strategy] and t in baseline_means
        ]
        x = range(len(deltas))
        ax.plot(x, deltas, "o-", color=color, label=label,
                linewidth=1.5, markersize=5, alpha=0.85)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(
        [t.replace("tox21_", "").replace("clintox_", "")
         for t in short_tasks],
        rotation=45, ha="right", fontsize=7
    )
    ax.set_ylabel("Δ ROC-AUC vs Hard Sharing")
    ax.set_title("(c) Per-Task Improvement")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Multi-Task Strategy Comparison on MoleculeNet Benchmarks",
        fontsize=13, y=1.02
    )
    plt.tight_layout()

    save_path = os.path.join(output_dir, "strategy_comparison_full.png")
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("=" * 80)
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print("=" * 80)

    ablation_path = "results/ablations/ablation_strategy.json"

    if not os.path.exists(ablation_path):
        print(f"Results file not found: {ablation_path}")
        return

    # Load results
    strategy_results = load_strategy_results(ablation_path)
    per_task_stats = compute_per_task_stats(strategy_results)

    # Print table
    print_full_results_table(strategy_results, per_task_stats)

    # Generate figure
    generate_paper_figure(strategy_results, per_task_stats)

    # Save processed stats
    output = {
        strategy: {
            "overall": {
                "mean": strategy_results["results"][strategy]["mean_auc"],
                "std": strategy_results["results"][strategy]["std_auc"],
            },
            "per_task": {
                task: {
                    "mean": per_task_stats[strategy][task]["mean"],
                    "std": per_task_stats[strategy][task]["std"],
                }
                for task in per_task_stats[strategy]
            },
        }
        for strategy in strategy_results["results"]
    }

    with open("results/strategy_comparison_processed.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nProcessed results saved: results/strategy_comparison_processed.json")


if __name__ == "__main__":
    main()
