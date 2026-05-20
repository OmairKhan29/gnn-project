"""
Low-Data Learning Curve Generation.

Plots ROC-AUC vs data fraction for:
    - scratch
    - linear_probe (unaligned)
    - linear_probe (aligned: best alignment from Phase 2)
    - finetune (aligned)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# Data Aggregator
# ─────────────────────────────────────────────

def aggregate_low_data_results(
    results_dir: str,
    dataset_name: str,
) -> Dict[str, Dict[float, Dict]]:
    """
    Load and aggregate low-data results from JSON files.

    Returns:
        {strategy: {fraction: {"mean": float, "std": float}}}
    """
    import glob

    aggregated = {}
    pattern = os.path.join(results_dir, f"*{dataset_name}*.json")

    for fpath in glob.glob(pattern):
        try:
            with open(fpath) as f:
                r = json.load(f)
        except Exception:
            continue

        strategy = r.get("strategy", r.get("transfer_strategy", "unknown"))
        fraction = r.get("fraction", 1.0)
        test_auc = r.get("test_auc", 0.5)

        if strategy not in aggregated:
            aggregated[strategy] = {}
        if fraction not in aggregated[strategy]:
            aggregated[strategy][fraction] = []
        aggregated[strategy][fraction].append(test_auc)

    # Compute mean ± std per fraction
    result = {}
    for strategy, frac_data in aggregated.items():
        result[strategy] = {}
        for frac, aucs in frac_data.items():
            result[strategy][frac] = {
                "mean": float(np.mean(aucs)),
                "std": float(np.std(aucs)),
                "n": len(aucs),
                "values": aucs,
            }

    return result


# ─────────────────────────────────────────────
# Learning Curve Plotter
# ─────────────────────────────────────────────

def plot_low_data_curves(
    curve_data: Dict[str, Dict[float, Dict]],
    dataset_name: str,
    save_path: str,
    title: Optional[str] = None,
):
    """
    Plot learning curves: AUC vs data fraction.

    Args:
        curve_data: {strategy: {fraction: {mean, std}}}
        dataset_name: Name of transfer dataset
        save_path: Output PNG path
        title: Plot title (auto-generated if None)
    """
    STRATEGY_STYLE = {
        "scratch": {
            "color": "#E74C3C", "marker": "o",
            "linestyle": "--", "label": "Scratch (no pretrain)",
        },
        "linear_probe": {
            "color": "#3498DB", "marker": "s",
            "linestyle": "-", "label": "Linear Probe (unaligned)",
        },
        "linear_probe_contrastive": {
            "color": "#2ECC71", "marker": "^",
            "linestyle": "-", "label": "Linear Probe (contrastive)",
        },
        "linear_probe_prototype": {
            "color": "#9B59B6", "marker": "D",
            "linestyle": "-", "label": "Linear Probe (prototype)",
        },
        "linear_probe_domain": {
            "color": "#F39C12", "marker": "v",
            "linestyle": "-", "label": "Linear Probe (domain)",
        },
        "top_layers": {
            "color": "#1ABC9C", "marker": "P",
            "linestyle": "-.", "label": "Fine-tune (top layers)",
        },
        "full": {
            "color": "#E67E22", "marker": "*",
            "linestyle": "-.", "label": "Fine-tune (full)",
        },
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for strategy, frac_results in curve_data.items():
        if not frac_results:
            continue

        fractions = sorted(frac_results.keys())
        means = [frac_results[f]["mean"] for f in fractions]
        stds = [frac_results[f]["std"] for f in fractions]

        style = STRATEGY_STYLE.get(
            strategy,
            {
                "color": "gray", "marker": "x",
                "linestyle": ":", "label": strategy,
            },
        )

        pct_labels = [int(f * 100) for f in fractions]

        ax.plot(
            pct_labels, means,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=style["label"],
            linewidth=2,
            markersize=8,
        )
        ax.fill_between(
            pct_labels,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15,
            color=style["color"],
        )

    ax.set_xlabel("Training Data (%)", fontsize=12)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title(title or f"Low-Data Performance: {dataset_name.upper()}", fontsize=13)
    ax.set_xticks([10, 25, 50, 100])
    ax.set_xticklabels(["10%", "25%", "50%", "100%"])
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved learning curve: {save_path}")


# ─────────────────────────────────────────────
# Degradation Table
# ─────────────────────────────────────────────

def print_degradation_table(
    curve_data: Dict[str, Dict[float, Dict]],
    dataset_name: str,
):
    """
    Print table: how much AUC drops as data is reduced.
    """
    print(f"\nDegradation Table: {dataset_name.upper()}")
    print("=" * 65)
    print(f"{'Strategy':<30} {'100%':>8} {'50%':>8} {'25%':>8} {'10%':>8}")
    print("-" * 65)

    for strategy, frac_data in sorted(curve_data.items()):
        row = f"{strategy:<30}"
        for frac in [1.0, 0.5, 0.25, 0.10]:
            if frac in frac_data:
                row += f" {frac_data[frac]['mean']:>8.4f}"
            else:
                row += f" {'N/A':>8}"
        print(row)


# ─────────────────────────────────────────────
# Efficiency Score
# ─────────────────────────────────────────────

def compute_data_efficiency_score(
    curve_data: Dict[str, Dict[float, Dict]],
    target_auc: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute data efficiency score for each strategy.

    Score = Area under the learning curve (normalized).
    Higher = better performance across all data regimes.

    Optionally: minimum fraction to reach target_auc.
    """
    scores = {}
    fractions_standard = [0.10, 0.25, 0.50, 1.00]

    for strategy, frac_data in curve_data.items():
        aucs = []
        for frac in fractions_standard:
            if frac in frac_data:
                aucs.append(frac_data[frac]["mean"])
            else:
                aucs.append(None)

        # AUC under the learning curve (trapezoidal)
        valid_pairs = [
            (fractions_standard[i], aucs[i])
            for i in range(len(aucs))
            if aucs[i] is not None
        ]
        if len(valid_pairs) > 1:
            xs = [p[0] for p in valid_pairs]
            ys = [p[1] for p in valid_pairs]
            area = float(np.trapz(ys, xs))
        else:
            area = 0.0

        scores[strategy] = area

    return scores