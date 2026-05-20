"""
Master Figure Builder for Feature 2 Phase 4.

Generates all 9 publication-ready figures:
    F1: Transfer Gain Bar Chart (aligned vs unaligned)
    F2: Low-Data Learning Curves (SIDER)
    F3: Low-Data Learning Curves (MUV)
    F4: t-SNE Embeddings (per-alignment comparison)
    F5: Per-Task ROC-AUC Comparison Bars
    F6: Ablation Study Line Plots
    F7: Gradient Conflict Heatmap
    F8: Prototype Similarity Heatmap
    F9: Cross-Dataset Cosine Similarity Matrix
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from feature2.evaluation.low_data_curves import plot_low_data_curves, aggregate_low_data_results


# ─────────────────────────────────────────────
# Figure 1: Transfer Gain Bars
# ─────────────────────────────────────────────

def fig_transfer_gain_bars(
    comparison_results: Dict,
    save_path: str,
):
    """Generate Figure 1: Transfer gain bar chart."""
    alignments = ["contrastive", "domain", "prototype"]
    datasets = ["sider", "muv"]
    colors = {
        "contrastive": "#3498DB",
        "domain": "#E74C3C",
        "prototype": "#2ECC71",
    }

    fig, axes = plt.subplots(1, len(datasets), figsize=(12, 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        gains = []
        labels = []
        bar_colors = []

        for alignment in alignments:
            g = comparison_results.get(alignment, {}).get(dataset, {}).get("linear_probe", {})
            baseline = comparison_results.get("unaligned", {}).get(dataset, {}).get("linear_probe", {}).get("test_auc_mean", 0.5)
            aligned = g.get("test_auc_mean", baseline)
            delta = aligned - baseline
            if delta != 0:
                gains.append(delta)
                labels.append(alignment.capitalize())
                bar_colors.append(colors.get(alignment, "gray"))

        x = np.arange(len(labels))
        bars = ax.bar(x, gains, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_title(f"{dataset.upper()} Transfer Gain", fontsize=13, fontweight="bold")
        ax.set_ylabel("Δ ROC-AUC vs Unaligned", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3, linestyle="-")
        ax.set_ylim(bottom=min(-0.02, min(gains) - 0.01), top=max(0.02, max(gains) + 0.01))

        # Add value labels
        for bar, gain in zip(bars, gains):
            height = bar.get_height()
            sign = "+" if gain >= 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    height + 0.002,
                    f"{sign}{gain:.4f}",
                    ha="center", va="bottom", fontsize=10)

    plt.suptitle("Figure 1: Transfer Performance Improvement via Representation Alignment",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Generated: Figure 1 - Transfer Gain Bars")


# ─────────────────────────────────────────────
# Figure 2 & 3: Low-Data Learning Curves
# ─────────────────────────────────────────────

def fig_learning_curves(
    low_data_dir: str,
    dataset: str,
    save_path: str,
):
    """Generate learning curves for a single dataset."""
    curve_data = aggregate_low_data_results(low_data_dir, dataset)
    
    STRATEGY_STYLE = {
        "scratch": {
            "color": "#E74C3C", "marker": "o",
            "linestyle": "--", "label": "Scratch",
        },
        "linear_probe": {
            "color": "#3498DB", "marker": "s",
            "linestyle": "-", "label": "Linear Probe (Unaligned)",
        },
        "linear_probe_contrastive": {
            "color": "#9B59B6", "marker": "^",
            "linestyle": "-", "label": "Linear Probe (Contrastive)",
        },
        "top_layers": {
            "color": "#2ECC71", "marker": "D",
            "linestyle": "-.", "label": "Fine-tune (Aligned)",
        },
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy, frac_results in curve_data.items():
        fractions = sorted(frac_results.keys())
        means = [frac_results[f]["mean"] for f in fractions]
        stds = [frac_results[f]["std"] for f in fractions]

        style = STRATEGY_STYLE.get(strategy, {
            "color": "gray", "marker": "x", "linestyle": ":", "label": strategy
        })

        pct_labels = [int(f * 100) for f in fractions]

        ax.plot(pct_labels, means,
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], label=style["label"],
                linewidth=2.5, markersize=10)
        ax.fill_between(pct_labels,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=style["color"])

    ax.set_xlabel("Training Data Fraction (%)", fontsize=13)
    ax.set_ylabel("ROC-AUC", fontsize=13)
    ax.set_title(f"Figure {'2' if dataset == 'sider' else '3'}: Low-Data Learning Curves ({dataset.upper()})",
                 fontsize=14, fontweight="bold")
    ax.set_xticks([10, 25, 50, 100])
    ax.set_xticklabels(["10%", "25%", "50%", "100%"], fontsize=12)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(0.45, 0.80)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Generated: Figure {'2' if dataset == 'sider' else '3'} - Learning Curves ({dataset})")


# ─────────────────────────────────────────────
# Figure 4: t-SNE Embeddings
# ─────────────────────────────────────────────

def fig_tsne_embeddings(
    embeddings_file: str,
    save_path: str,
):
    """Generate t-SNE visualization of molecular embeddings."""
    from sklearn.manifold import TSNE
    
    with open(embeddings_file) as f:
        data = json.load(f)
    
    embeddings = data["embeddings"]
    labels = data["dataset_labels"]
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="random", n_iter=1000)
    proj = tsne.fit_transform(embeddings)
    
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, lbl in enumerate(unique_labels):
        idx = [j for j, l in enumerate(labels) if l == lbl]
        ax.scatter(proj[idx, 0], proj[idx, 1],
                   c=[colors[i]], label=lbl, s=30, alpha=0.7, edgecolor="white", linewidth=0.5)
    
    ax.set_title("Figure 4: Molecular Embedding Space (t-SNE)", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.legend(fontsize=11, loc="best")
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Generated: Figure 4 - t-SNE Embeddings")


# ─────────────────────────────────────────────
# Figure 5: Per-Task ROC-AUC Comparison
# ─────────────────────────────────────────────

def fig_per_task_comparison(
    comparison_results: Dict,
    dataset: str,
    save_path: str,
):
    """Generate grouped bar chart for per-task ROC-AUC."""
    strategies = ["unaligned", "contrastive", "prototype"]
    colors = {
        "unaligned": "#BDC3C7",
        "contrastive": "#3498DB",
        "prototype": "#2ECC71",
    }

    # Get task names from first available result
    for strat in strategies:
        if strat in comparison_results and dataset in comparison_results[strat]:
            per_task = comparison_results[strat][dataset].get("linear_probe", {}).get("test_per_task_auc", {})
            break
    
    if not per_task:
        print(f"  No per-task data for {dataset}, skipping")
        return
    
    task_names = list(per_task.keys())
    n_tasks = len(task_names)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(n_tasks)
    width = 0.25

    for i, strat in enumerate(strategies):
        if strat in comparison_results and dataset in comparison_results[strat]:
            pt = comparison_results[strat][dataset].get("linear_probe", {}).get("test_per_task_auc", {})
            vals = [pt.get(name, 0.5) for name in task_names]
        else:
            vals = [0.5] * n_tasks
        
        bars = ax.bar(x + i * width - width * (len(strategies)-1) / 2,
                      vals, width, label=strat.capitalize(), color=colors.get(strat, "gray"))
    
    ax.set_xlabel("Tasks", fontsize=12)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title(f"Figure 5: Per-Task ROC-AUC on {dataset.upper()}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([name.split("_")[0][:10] for name in task_names], rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Generated: Figure 5 - Per-Task Comparison ({dataset})")


# ─────────────────────────────────────────────
# Figure 6: Ablation Study
# ─────────────────────────────────────────────

def fig_ablation_plot(
    ablation_analysis: Dict,
    save_path: str,
):
    """Generate line/point plot for ablation results."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ablation_names = list(ablation_analysis.keys())
    basenames = ["No Condition", "Lambda 0.0", "Lambda 0.3", "Lambda 0.5", "Projection 64", "Temp 0.1"]
    
    test_aucs = []
    errors = []
    
    for name in ablation_names:
        summary = ablation_analysis[name].get("summary", {})
        mean_auc = summary.get("mean", 0.5)
        std_auc = summary.get("std", 0.0)
        test_aucs.append(mean_auc)
        errors.append(std_auc)
    
    x = np.arange(len(ablation_names))
    ax.errorbar(x, test_aucs, yerr=errors, fmt='o', capsize=5,
                color="#3498DB", markersize=10, linewidth=2)
    
    ax.set_xlabel("Ablation Configuration", fontsize=12)
    ax.set_ylabel("Test ROC-AUC", fontsize=12)
    ax.set_title("Figure 6: Ablation Study Results", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("_", " ").title()[:15] for name in ablation_names], rotation=45, ha="right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Generated: Figure 6 - Ablation Study")


# ─────────────────────────────────────────────
# Figure 7: Gradient Conflict Heatmap
# ─────────────────────────────────────────────

def fig_gradient_heatmap(
    gradient_file: str,
    save_path: str,
):
    """Generate heatmap of gradient conflicts between tasks."""
    if not os.path.exists(gradient_file):
        print(f"  Gradient file not found: {gradient_file}")
        return
    
    with open(gradient_file) as f:
        data = json.load(f)
    
    sim_matrix = np.array(data.get("similarity_matrix", []))
    task_names = data.get("task_names", [f"T{i}" for i in range(len(sim_matrix))])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    
    ax.set_xticks(np.arange(len(task_names)))
    ax.set_yticks(np.arange(len(task_names)))
    ax.set_xticklabels([n[:12] for n in task_names], fontsize=10, rotation=45, ha="right")
    ax.set_yticklabels([n[:12] for n in task_names], fontsize=10)
    
    ax.set_title("Figure 7: Gradient Conflict Heatmap (Cosine Similarity)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Target Tasks", fontsize=12)
    ax.set_ylabel("Source Tasks", fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine Similarity", fontsize=11)
    
    # Add values
    thresh = sim_matrix.max() / 2.
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            text = ax.text(j, i, f"{sim_matrix[i, j]:.2f}",
                          ha="center", va="center",
                          color="white" if abs(sim_matrix[i, j]) > thresh else "black",
                          fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Generated: Figure 7 - Gradient Heatmap")


# ─────────────────────────────────────────────
# Figure 8: Prototype Similarity Heatmap
# ─────────────────────────────────────────────

def fig_prototype_heatmap(
    prototype_similarity: np.ndarray,
    save_path: str,
):
    """Generate heatmap of prototype similarities."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(prototype_similarity, cmap="YlOrRd", vmin=0, vmax=1)
    n = len(prototype_similarity)
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([f"P{i+1}" for i in range(n)], fontsize=10)
    ax.set_yticklabels([f"P{i+1}" for i in range(n)], fontsize=10)
    
    ax.set_title("Figure 8: Prototype Embedding Similarity Matrix",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Prototypes", fontsize=12)
    ax.set_ylabel("Prototypes", fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Similarity", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Generated: Figure 8 - Prototype Similarity")


# ─────────────────────────────────────────────
# Figure 9: Cross-Dataset Similarity
# ─────────────────────────────────────────────

def fig_cross_dataset_similarity(
    sim_matrix: np.ndarray,
    dataset_names: List[str],
    save_path: str,
):
    """Generate cross-dataset embedding similarity heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(sim_matrix, cmap="viridis", vmin=0, vmax=1)
    n = len(dataset_names)
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([d.upper() for d in dataset_names], fontsize=11)
    ax.set_yticklabels([d.upper() for d in dataset_names], fontsize=11)
    
    ax.set_title("Figure 9: Cross-Dataset Embedding Similarity",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Dataset", fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean Cosine Similarity", fontsize=11)
    
    # Add values
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if sim_matrix[i, j] > 0.5 else "black",
                    fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Generated: Figure 9 - Cross-Dataset Similarity")


# ─────────────────────────────────────────────
# Master Generator
# ─────────────────────────────────────────────

def generate_all_figures(
    base_result_dir: str = "results/feature2",
    figures_dir: str = "results/feature2/figures",
):
    """Generate all 9 publication figures."""
    os.makedirs(figures_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("FEATURE 2 — PHASE 4: Generating All Publication Figures")
    print("=" * 70)

    # Load required data
    comparison_path = os.path.join(base_result_dir, "transfer_comparison", "full_comparison.json")
    ablation_path = os.path.join(base_result_dir, "ablations", "all_ablation_results.json")
    
    if os.path.exists(comparison_path):
        with open(comparison_path) as f:
            comparison_results = json.load(f)
    else:
        print("Warning: No comparison results found")
        comparison_results = {}

    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            ablation_results = json.load(f)
        from feature2.statistical.significance import analyze_ablations
        ablation_analysis = analyze_ablations(ablation_results)
    else:
        print("Warning: No ablation results found")
        ablation_analysis = {}

    # Figure 1: Transfer Gain Bars
    fig1_path = os.path.join(figures_dir, "fig1_transfer_gain.png")
    fig_transfer_gain_bars(comparison_results, fig1_path)

    # Figures 2 & 3: Learning Curves
    for i, dataset in enumerate(["sider", "muv"]):
        fig_path = os.path.join(figures_dir, f"fig{2+i}_learning_curves_{dataset}.png")
        fig_learning_curves(base_result_dir + "/low_data", dataset, fig_path)

    # Figure 4: t-SNE
    tsne_path = os.path.join(figures_dir, "embeddings.json")
    if os.path.exists(tsne_path):
        fig4_path = os.path.join(figures_dir, "fig4_tsne.png")
        fig_tsne_embeddings(tsne_path, fig4_path)

    # Figures 5a & 5b: Per-Task
    for dataset in ["sider", "muv"]:
        fig5_path = os.path.join(figures_dir, f"fig5_per_task_{dataset}.png")
        fig_per_task_comparison(comparison_results, dataset, fig5_path)

    # Figure 6: Ablation
    fig6_path = os.path.join(figures_dir, "fig6_ablation.png")
    fig_ablation_plot(ablation_analysis, fig6_path)

    # Figure 7: Gradient Heatmap
    grad_path = os.path.join(base_result_dir, "gradient_analysis", "gradient_sim.json")
    fig7_path = os.path.join(figures_dir, "fig7_gradient_heatmap.png")
    fig_gradient_heatmap(grad_path, fig7_path)

    # Figures 8 & 9: Similarity matrices (placeholder - would need actual proto/sim data)
    print("\nNote: Figures 8-9 require prototype/cross-dataset embedding data.")
    print("Use feature2/scripts/evaluate_embedding_quality.py first.")

    print("\n" + "=" * 70)
    print(f"All figures saved to: {figures_dir}/")
    print("=" * 70)