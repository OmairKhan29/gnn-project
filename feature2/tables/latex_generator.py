"""
LaTeX Table Generator for Feature 2.

Generates publication-quality tables with:
    - Proper column formatting
    - Star notation for significance
    - Mean ± Std formatting
    - Clean spacing
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional


# ─────────────────────────────────────────────
# Main Results Table
# ─────────────────────────────────────────────

def generate_main_results_table(
    comparison_results: Dict,
    save_path: str = "results/feature2/tables/main_results.tex",
) -> str:
    """
    Generate LaTeX table for transfer comparison results.

    Columns: Strategy | SIDER | MUV | Avg Improvement
    Rows: No Alignment, Contrastive, Domain, Prototype
    """
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{Transfer Performance Comparison Across Alignment Strategies}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Strategy & SIDER (ROC-AUC) & MUV (ROC-AUC) & Avg Δ (pct) \\\\\\\\")
    lines.append("\\midrule")

    strategies = ["unaligned", "contrastive", "domain", "prototype"]
    datasets = ["sider", "muv"]
    all_deltas = []

    for strat in strategies:
        row = []
        row.append(strat.replace("_", "\\textunderscore").capitalize())
        
        for ds in datasets:
            if strat in comparison_results and ds in comparison_results[strat]:
                linear_probe = comparison_results[strat][ds].get("linear_probe", {})
                auc = linear_probe.get("test_auc_mean", 0.5)
                std = linear_probe.get("test_auc_std", 0.0)
                row.append(f"{auc:.4f} \\pm {std:.4f}")
            else:
                row.append("-")
        
        # Compute avg delta vs unaligned
        if strat != "unaligned":
            deltas = []
            for ds in datasets:
                baseline = comparison_results.get("unaligned", {}).get(ds, {}).get("linear_probe", {}).get("test_auc_mean", 0.5)
                aligned = comparison_results.get(strat, {}).get(ds, {}).get("linear_probe", {}).get("test_auc_mean", baseline)
                delta = (aligned - baseline) / max(baseline, 1e-8) * 100
                deltas.append(delta)
            if deltas:
                avg_delta = np.mean(deltas)
                all_deltas.append(avg_delta)
                row.append(f"{avg_delta:+.2f}%")
            else:
                row.append("-")
        else:
            row.append("-")
        
        lines.append(" & ".join(row) + " \\\\\\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex_content = "\n".join(lines)
    with open(save_path, "w") as f:
        f.write(tex_content)

    print(f"Generated main results table: {save_path}")
    return tex_content


# ─────────────────────────────────────────────
# Ablation Study Table
# ─────────────────────────────────────────────

def generate_ablation_table(
    ablation_analysis: Dict,
    save_path: str = "results/feature2/tables/ablation_study.tex",
) -> str:
    """
    Generate LaTeX table for ablation study.

    Columns: Ablation | Test AUC | vs Baseline Δ | p-value | Cohen's d
    """
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation Study Results on Multi-Task Classification}")
    lines.append("\\label{tab:ablation_study}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Ablation & Test AUC & Δ vs Baseline & p-value & Cohen's d \\\\\\\\")
    lines.append("\\midrule")

    # First row: full model (baseline)
    lines.append("Full Model (Prototype) & \\multicolumn{4}{c}{Baseline} \\\\\\\\")

    for abl_name, analysis in ablation_analysis.items():
        abl_name_display = abl_name.replace("_", "-")
        
        summary = analysis.get("summary", {})
        mean_auc = summary.get("mean", 0.5)
        std_auc = summary.get("std", 0.0)
        
        cmp = analysis.get("comparison_vs_baseline", {})
        delta = cmp.get("mean_difference", 0.0)
        p_val = cmp.get("p_value", 1.0)
        stars = format_significance_stars(p_val)
        
        effect = analysis.get("effect_size", {})
        cohens_d = effect.get("cohen_d", 0.0)
        
        lines.append(f"{abl_name_display:<15} "
                     f"& {mean_auc:.4f} \\pm {std_auc:.4f} "
                     f"& {delta:+.4f} "
                     f"& {p_val:.3f}{stars} "
                     f"& {cohens_d:+.2f} \\\\\\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex_content = "\n".join(lines)
    with open(save_path, "w") as f:
        f.write(tex_content)

    print(f"Generated ablation table: {save_path}")
    return tex_content


# ─────────────────────────────────────────────
# Low-Data Performance Table
# ─────────────────────────────────────────────

def generate_low_data_table(
    low_data_results: List[Dict],
    dataset: str = "sider",
    save_path: str = "results/feature2/tables/low_data_performance.tex",
) -> str:
    """
    Generate LaTeX table for low-data learning results.

    Columns: Fraction (%) | Scratch | Linear Probe | Fine-tune (Aligned)
    """
    # Filter by dataset
    filtered = [r for r in low_data_results if r.get("dataset") == dataset]

    # Aggregate by fraction × strategy
    agg = {}
    for r in filtered:
        frac = int(r.get("fraction", 1.0) * 100)
        strategy = r.get("strategy", "unknown")
        if frac not in agg:
            agg[frac] = {}
        if strategy not in agg[frac]:
            agg[frac][strategy] = []
        if "error" not in r:
            agg[frac][strategy].append(r.get("test_auc", 0.5))

    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Low-Data Performance on {dataset.upper()} Dataset}}")
    lines.append("\\label{tab:low_data_" + dataset.lower() + "}")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Training Data & Scratch & Linear Probe & Fine-tune (Aligned) \\\\\\\\")
    lines.append("\\midrule")

    for frac in sorted(agg.keys(), reverse=True):
        row = []
        row.append(f"{frac}%")

        # Scratch
        scratch_vals = agg[frac].get("scratch", [])
        if scratch_vals:
            mean = np.mean(scratch_vals)
            std = np.std(scratch_vals)
            row.append(f"{mean:.4f} \\pm {std:.4f}")
        else:
            row.append("-")

        # Linear probe
        lp_vals = agg[frac].get("linear_probe", [])
        if lp_vals:
            mean = np.mean(lp_vals)
            std = np.std(lp_vals)
            row.append(f"{mean:.4f} \\pm {std:.4f}")
        else:
            row.append("-")

        # Fine-tune
        ft_vals = agg[frac].get("top_layers", [])
        if ft_vals:
            mean = np.mean(ft_vals)
            std = np.std(ft_vals)
            row.append(f"{mean:.4f} \\pm {std:.4f}")
        else:
            row.append("-")

        lines.append(" & ".join(row) + " \\\\\\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex_content = "\n".join(lines)
    with open(save_path, "w") as f:
        f.write(tex_content)

    print(f"Generated low-data table for {dataset}: {save_path}")
    return tex_content


# ─────────────────────────────────────────────
# Embedding Quality Metrics Table
# ─────────────────────────────────────────────

def generate_embedding_metrics_table(
    metrics_files: List[str],
    save_path: str = "results/feature2/tables/embedding_quality.tex",
) -> str:
    """
    Generate LaTeX table for embedding quality metrics.

    Columns: Strategy | Silhouette | Davies-Bouldin | Inter-Dataset Similarity
    """
    metrics_by_strategy = {}

    for fpath in metrics_files:
        if os.path.exists(fpath):
            with open(fpath) as f:
                m = json.load(f)
            name = os.path.basename(fpath).replace(".json", "")
            metrics_by_strategy[name] = m

    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{Embedding Quality Metrics Across Alignment Strategies}")
    lines.append("\\label{tab:embedding_quality}")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Strategy & Silhouette Score & Davies-Bouldin & Avg Similarity \\\\\\\\")
    lines.append("\\midrule")

    for strategy, m in metrics_by_strategy.items():
        cluster = m.get("cluster_scores", {})
        sil = cluster.get("silhouette", 0.0)
        db = cluster.get("davies_bouldin", 0.0)

        sim_matrix = m.get("similarity_matrix", [])
        if sim_matrix:
            off_diag_sims = []
            n = len(sim_matrix)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        off_diag_sims.append(sim_matrix[i][j])
            avg_sim = np.mean(off_diag_sims)
        else:
            avg_sim = 0.0

        lines.append(f"{strategy:<20} "
                     f"& {sil:.4f} "
                     f"& {db:.4f} "
                     f"& {avg_sim:.4f} \\\\\\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex_content = "\n".join(lines)
    with open(save_path, "w") as f:
        f.write(tex_content)

    print(f"Generated embedding metrics table: {save_path}")
    return tex_content