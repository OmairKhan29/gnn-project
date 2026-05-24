"""
Transfer Learning Metrics for Feature 2.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score
import json
import os


# ─────────────────────────────────────────────
# Core Transfer Metrics
# ─────────────────────────────────────────────

def compute_transfer_gain(
    transfer_auc: float,
    scratch_auc: float,
) -> Dict:
    """
    Compute transfer learning gain over scratch baseline.

    Returns:
        absolute_gain: AUC difference
        relative_gain: Percentage improvement
        is_positive: Whether transfer helped
    """
    absolute_gain = transfer_auc - scratch_auc
    relative_gain = (absolute_gain / max(scratch_auc, 1e-8)) * 100.0

    return {
        "transfer_auc": transfer_auc,
        "scratch_auc": scratch_auc,
        "absolute_gain": absolute_gain,
        "relative_gain_pct": relative_gain,
        "is_positive": absolute_gain > 0,
    }


def aggregate_results_across_seeds(
    results_list: List[Dict],
    metric_key: str = "test_auc",
) -> Dict:
    """
    Aggregate results across multiple seeds.

    Returns:
        mean, std, min, max, n_seeds
    """
    values = [r[metric_key] for r in results_list if metric_key in r]

    if not values:
        return {"mean": 0.0, "std": 0.0, "n_seeds": 0}

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n_seeds": len(values),
        "values": values,
    }


def compute_low_data_degradation(
    results_by_fraction: Dict[float, float],
) -> Dict:
    """
    Compute performance degradation as data fraction decreases.

    Args:
        results_by_fraction: {fraction: mean_auc}

    Returns:
        degradation metrics
    """
    fractions = sorted(results_by_fraction.keys(), reverse=True)
    full_auc = results_by_fraction.get(1.0, results_by_fraction[fractions[0]])

    degradation = {}
    for frac in fractions:
        auc = results_by_fraction[frac]
        degradation[frac] = {
            "auc": auc,
            "absolute_drop": full_auc - auc,
            "relative_drop_pct": ((full_auc - auc) / max(full_auc, 1e-8)) * 100,
        }

    return degradation


# ─────────────────────────────────────────────
# Results Loader
# ─────────────────────────────────────────────

def load_all_phase1_results(result_dir: str) -> Dict:
    """
    Load all Phase 1 result JSON files from result_dir.

    Returns:
        Dict mapping experiment_name → results dict
    """
    all_results = {}

    for fname in os.listdir(result_dir):
        if fname.endswith("_results.json"):
            fpath = os.path.join(result_dir, fname)
            with open(fpath) as f:
                results = json.load(f)
            exp_name = results.get("experiment_name", fname.replace("_results.json", ""))
            all_results[exp_name] = results

    return all_results


def build_comparison_table(results: Dict) -> str:
    """
    Build a formatted comparison table from results.

    Args:
        results: {experiment_name: {test_auc: float, val_auc: float, ...}}

    Returns:
        Formatted string table
    """
    header = f"{'Experiment':<40} {'Val AUC':>10} {'Test AUC':>10}"
    sep = "-" * 65
    lines = [header, sep]

    for name, r in sorted(results.items()):
        val_auc = r.get("val_auc", 0.0)
        test_auc = r.get("test_auc", 0.0)
        lines.append(f"{name:<40} {val_auc:>10.4f} {test_auc:>10.4f}")

    return "\n".join(lines)