"""
Statistical Significance Testing for Feature 2.

Computes:
    - Paired t-tests (aligned vs unaligned)
    - One-sample t-tests (vs random baseline 0.5)
    - 95% Confidence Intervals (bootstrap)
    - Effect sizes (Cohen's d)
    - Multiple testing correction (Bonferroni)
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json
import os


# ─────────────────────────────────────────────
# Paired T-Test
# ─────────────────────────────────────────────

def paired_t_test(
    group_a: List[float],
    group_b: List[float],
    alternative: str = "two-sided",  # "two-sided", "greater", "less"
    alpha: float = 0.05,
) -> Dict:
    """
    Paired t-test comparing two groups.

    Args:
        group_a: Values for condition A (list of scores)
        group_b: Values for condition B (same length, same seeds)
        alternative: Alternative hypothesis
        alpha: Significance level

    Returns:
        Dictionary with t-statistic, p-value, conclusion, confidence interval
    """
    if len(group_a) != len(group_b):
        raise ValueError("Groups must have equal length")

    if len(group_a) < 2:
        return {
            "can_compute": False,
            "reason": "Need at least 2 samples",
            "t_stat": None,
            "p_value": None,
            "significant": False,
        }

    stat, p_val = stats.ttest_rel(group_a, group_b)
    
    # Determine significance based on alternative
    if alternative == "two-sided":
        significant = p_val < alpha
        ci_low, ci_high = stats.t.interval(
            1 - alpha,
            len(group_a) - 1,
            loc=np.mean(np.array(group_a) - np.array(group_b)),
            scale=stats.sem(np.array(group_a) - np.array(group_b)),
        )
    elif alternative == "greater":
        p_one_tail = p_val / 2
        significant = p_one_tail < alpha
        # One-sided upper CI
        ci_low, _ = stats.t.interval(
            1 - alpha,
            len(group_a) - 1,
            loc=np.mean(np.array(group_a) - np.array(group_b)),
            scale=stats.sem(np.array(group_a) - np.array(group_b)),
        )
        ci_high = float('inf')
    else:  # less
        p_one_tail = p_val / 2
        significant = p_one_tail < alpha
        _, ci_high = stats.t.interval(
            1 - alpha,
            len(group_a) - 1,
            loc=np.mean(np.array(group_a) - np.array(group_b)),
            scale=stats.sem(np.array(group_a) - np.array(group_b)),
        )
        ci_low = float('-inf')

    mean_diff = np.mean(np.array(group_a) - np.array(group_b))
    std_diff = np.std(np.array(group_a) - np.array(group_b))

    return {
        "can_compute": True,
        "alternative": alternative,
        "alpha": alpha,
        "n_samples": len(group_a),
        "mean_a": float(np.mean(group_a)),
        "mean_b": float(np.mean(group_b)),
        "mean_difference": float(mean_diff),
        "std_difference": float(std_diff),
        "t_statistic": float(stat),
        "p_value": float(p_val),
        "ci_lower": float(ci_low),
        "ci_upper": float(ci_high),
        "significant": bool(significant),
    }


# ─────────────────────────────────────────────
# Effect Size (Cohen's d)
# ─────────────────────────────────────────────

def cohen_d(
    group_a: List[float],
    group_b: List[float],
) -> Dict:
    """
    Compute Cohen's d effect size for paired samples.

    Interpretation:
        0.2 = small effect
        0.5 = medium effect
        0.8 = large effect

    Returns:
        Dictionary with effect size and interpretation
    """
    if len(group_a) < 2 or len(group_b) < 2:
        return {"d": None, "interpretation": "insufficient samples"}

    mean_diff = np.mean(np.array(group_a) - np.array(group_b))
    pooled_std = np.sqrt((np.var(group_a) + np.var(group_b)) / 2)

    if pooled_std == 0:
        d = float('inf') if mean_diff > 0 else float('-inf')
    else:
        d = mean_diff / pooled_std

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {
        "cohen_d": float(d),
        "abs_cohen_d": float(abs_d),
        "interpretation": interpretation,
        "mean_difference": float(mean_diff),
        "pooled_std": float(pooled_std),
    }


# ─────────────────────────────────────────────
# Bootstrap Confidence Interval
# ─────────────────────────────────────────────

def bootstrap_confidence_interval(
    values: List[float],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    statistic: callable = np.mean,
) -> Dict:
    """
    Compute bootstrap confidence interval for any statistic.

    Args:
        values: Original sample values
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Desired CI level (0.95 = 95%)
        statistic: Statistic to compute (mean, median, etc.)

    Returns:
        Dictionary with estimate, CI bounds, std_error
    """
    rng = np.random.RandomState(42)
    original_stat = statistic(np.array(values))

    bootstrapped_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrapped_stats.append(statistic(sample))

    bootstrapped_stats = np.array(bootstrapped_stats)
    alpha = 1 - confidence_level
    lower = float(np.percentile(bootstrapped_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrapped_stats, 100 * (1 - alpha / 2)))

    return {
        "estimate": float(original_stat),
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
        "std_error": float(np.std(bootstrapped_stats)),
        "ci_width": upper - lower,
    }


# ─────────────────────────────────────────────
# Bonferroni Correction
# ─────────────────────────────────────────────

def bonferroni_correction(
    p_values: List[float],
    n_tests: Optional[int] = None,
) -> Dict:
    """
    Apply Bonferroni correction for multiple hypothesis testing.

    Args:
        p_values: List of raw p-values
        n_tests: Number of tests (defaults to len(p_values))

    Returns:
        Dictionary with adjusted p-values and significance flags
    """
    if n_tests is None:
        n_tests = len(p_values)

    corrected_p = [min(p * n_tests, 1.0) for p in p_values]
    significant_raw = [p < 0.05 for p in p_values]
    significant_corrected = [p < 0.05 for p in corrected_p]

    return {
        "raw_p_values": p_values,
        "corrected_p_values": corrected_p,
        "significance_raw": significant_raw,
        "significance_corrected": significant_corrected,
        "n_tests": n_tests,
        "bonferroni_threshold": 0.05 / n_tests,
    }


# ─────────────────────────────────────────────
# Summary Statistics
# ─────────────────────────────────────────────

def compute_summary_statistics(
    values: List[float],
) -> Dict:
    """
    Compute basic descriptive statistics.
    """
    arr = np.array(values)
    
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "sem": float(stats.sem(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
    }


# ─────────────────────────────────────────────
# Load and Analyze Results
# ─────────────────────────────────────────────

def load_ablation_results(result_path: str) -> Dict:
    """Load ablation results from JSON file."""
    with open(result_path) as f:
        return json.load(f)


def analyze_ablations(
    results: Dict,
    baseline_key: str = "no_alignment",
    target_metric: str = "test_auc",
) -> Dict:
    """
    Perform statistical analysis on ablation results.

    Compares each ablation against baseline using paired t-tests.
    """
    analysis = {}
    
    for ablation_name, seed_results in results.items():
        if ablation_name == baseline_key:
            continue
            
        # Extract test_aucs for valid seeds
        ablation_aucs = []
        baseline_aucs = []
        
        seed_keys = sorted([k for k in seed_results.keys() if k.isdigit()])
        
        for seed_str in seed_keys:
            ablation_r = seed_results.get(seed_str, {})
            baseline_r = results.get(baseline_key, {}).get(seed_str, {})
            
            if target_metric in ablation_r and target_metric in baseline_r:
                ablation_aucs.append(ablation_r[target_metric])
                baseline_aucs.append(baseline_r[target_metric])
        
        if len(ablation_aucs) >= 2:
            # Paired t-test
            t_result = paired_t_test(ablation_aucs, baseline_aucs)
            
            # Effect size
            effect = cohen_d(ablation_aucs, baseline_aucs)
            
            # Bootstrap CI
            ci = bootstrap_confidence_interval(ablation_aucs)
            
            analysis[ablation_name] = {
                "comparison_vs_baseline": t_result,
                "effect_size": effect,
                "bootstrap_ci": ci,
                "summary": compute_summary_statistics(ablation_aucs),
            }
    
    return analysis


def format_significance_stars(p_value: float) -> str:
    """Convert p-value to significance stars notation."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    elif p_value < 0.1:
        return "."
    else:
        return ""