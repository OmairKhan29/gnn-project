"""
evaluation/pcgrad_analysis.py
Deep analysis of PCGrad behavior during training.
Produces paper-ready conflict statistics and visualizations.

Answers:
  1. How much does PCGrad reduce gradient conflict?
  2. Which task pairs conflict most?
  3. Does conflict ratio change over training?
  4. Is there correlation between conflict reduction and AUC improvement?
"""
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

matplotlib.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ─────────────────────────────────────────────────────────────────────────────
# Per-Batch Gradient Collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_task_gradients_per_batch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 20,
) -> Dict[int, List[torch.Tensor]]:
    """
    Collect per-task gradient vectors over multiple batches.

    Parameters
    ----------
    model       : multi-task model with compute_per_task_losses()
    loader      : DataLoader returning multi-task batches
    device      : torch.device
    num_batches : int — how many batches to sample

    Returns
    -------
    dict mapping task_id -> list of flat gradient tensors
    """
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]

    task_grads: Dict[int, List[torch.Tensor]] = {}
    batch_count = 0

    for batch in loader:
        if batch_count >= num_batches:
            break

        batch = batch.to(device)
        task_losses = model.compute_per_task_losses(batch)

        for task_id, loss in task_losses.items():
            # Retain graph for all tasks
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )

            flat = torch.cat([
                g.detach().view(-1) if g is not None
                else torch.zeros(p.numel(), device=device, dtype=p.dtype)
                for g, p in zip(grads, params)
            ])

            if task_id not in task_grads:
                task_grads[task_id] = []
            task_grads[task_id].append(flat.cpu())

        # Free graph after processing all tasks
        model.zero_grad()
        batch_count += 1

    return task_grads


def compute_mean_task_gradients(
    task_grads: Dict[int, List[torch.Tensor]],
) -> Dict[int, torch.Tensor]:
    """Average gradient vectors across batches for each task."""
    return {
        task_id: torch.stack(grads).mean(dim=0)
        for task_id, grads in task_grads.items()
        if len(grads) > 0
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise Conflict Matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_pairwise_conflict_matrix(
    mean_grads: Dict[int, torch.Tensor],
    task_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute pairwise cosine similarity matrix and conflict matrix.

    Parameters
    ----------
    mean_grads : dict[task_id -> Tensor[D]]
    task_names : list of str (indexed by task_id)

    Returns
    -------
    cosine_matrix   : np.ndarray [T, T] — values in [-1, 1]
    conflict_matrix : np.ndarray [T, T] — binary, 1 if conflicting
    labels          : list of str
    """
    task_ids = sorted(mean_grads.keys())
    n = len(task_ids)

    cosine_matrix = np.zeros((n, n))
    conflict_matrix = np.zeros((n, n))

    for i, ti in enumerate(task_ids):
        for j, tj in enumerate(task_ids):
            if i == j:
                cosine_matrix[i, j] = 1.0
                continue

            gi = mean_grads[ti]
            gj = mean_grads[tj]

            norm_i = torch.norm(gi)
            norm_j = torch.norm(gj)

            if norm_i < 1e-12 or norm_j < 1e-12:
                cosine_matrix[i, j] = 0.0
                continue

            cos = torch.dot(gi, gj) / (norm_i * norm_j)
            cosine_matrix[i, j] = cos.item()

            if cos.item() < 0:
                conflict_matrix[i, j] = 1.0

    labels = [
        "_".join(task_names[tid].split("_")[1:])
        for tid in task_ids
    ]

    return cosine_matrix, conflict_matrix, labels


# ─────────────────────────────────────────────────────────────────────────────
# Training-Time Conflict Tracking
# ─────────────────────────────────────────────────────────────────────────────

def plot_conflict_ratio_over_training(
    pcgrad_stats_history: List[Dict],
    save_path: Optional[str] = None,
    figsize: Tuple = (10, 5),
) -> plt.Figure:
    """
    Plot gradient conflict ratio over training epochs.

    Shows how PCGrad progressively reduces task conflicts.

    Parameters
    ----------
    pcgrad_stats_history : list of per-epoch stats dicts
        Each dict must have 'conflict_ratio_before' and 'conflict_reduction'
    """
    epochs = list(range(1, len(pcgrad_stats_history) + 1))
    conflict_before = [s.get("conflict_ratio_before", 0) for s in pcgrad_stats_history]
    conflict_after = [
        max(0, s.get("conflict_ratio_before", 0) - s.get("conflict_reduction", 0))
        for s in pcgrad_stats_history
    ]
    avg_cos_before = [s.get("avg_cosine_before", 0) for s in pcgrad_stats_history]
    avg_cos_after = [s.get("avg_cosine_after", 0) for s in pcgrad_stats_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Conflict ratio
    ax1.plot(epochs, conflict_before, color="#FF6B6B", linewidth=1.5,
             label="Before PCGrad", alpha=0.8)
    ax1.plot(epochs, conflict_after, color="#4ECDC4", linewidth=1.5,
             label="After PCGrad", alpha=0.8)
    ax1.fill_between(epochs, conflict_after, conflict_before,
                     alpha=0.15, color="#45B7D1", label="Reduction")
    ax1.set_xlabel("Training Epoch")
    ax1.set_ylabel("Gradient Conflict Ratio")
    ax1.set_title("Gradient Conflict Ratio During Training")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Cosine similarity
    ax2.plot(epochs, avg_cos_before, color="#FF6B6B", linewidth=1.5,
             label="Before PCGrad", alpha=0.8)
    ax2.plot(epochs, avg_cos_after, color="#4ECDC4", linewidth=1.5,
             label="After PCGrad", alpha=0.8)
    ax2.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Training Epoch")
    ax2.set_ylabel("Mean Cosine Similarity")
    ax2.set_title("Mean Gradient Cosine Similarity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_pairwise_conflict_heatmap(
    cosine_matrix: np.ndarray,
    labels: List[str],
    title: str = "Gradient Cosine Similarity",
    save_path: Optional[str] = None,
    figsize: Tuple = (12, 10),
) -> plt.Figure:
    """
    Heatmap of pairwise gradient cosine similarities.
    Red = conflicting, Green = aligned.
    """
    fig, ax = plt.subplots(figsize=figsize)

    mask = np.eye(len(labels), dtype=bool)

    sns.heatmap(
        cosine_matrix,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        mask=mask,
        annot_kws={"fontsize": 7},
        linewidths=0.5,
    )

    # Diagonal
    for i in range(len(labels)):
        ax.add_patch(plt.Rectangle(
            (i, i), 1, 1,
            fill=True, color="#DDDDDD",
            transform=ax.transData
        ))

    ax.set_title(title, fontsize=14, pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_conflict_before_after(
    cosine_before: np.ndarray,
    cosine_after: np.ndarray,
    labels: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple = (14, 6),
) -> plt.Figure:
    """
    Side-by-side comparison: gradient cosine similarity before vs after PCGrad.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    common_kwargs = dict(
        annot=True, fmt=".2f",
        cmap="RdYlGn", center=0,
        vmin=-1, vmax=1,
        annot_kws={"fontsize": 7},
        linewidths=0.3,
    )

    sns.heatmap(cosine_before, xticklabels=labels, yticklabels=labels,
                ax=ax1, **common_kwargs)
    ax1.set_title("Before PCGrad", fontsize=13)
    ax1.tick_params(axis="x", rotation=45, labelsize=8)
    ax1.tick_params(axis="y", rotation=0, labelsize=8)

    sns.heatmap(cosine_after, xticklabels=labels, yticklabels=labels,
                ax=ax2, **common_kwargs)
    ax2.set_title("After PCGrad Projection", fontsize=13)
    ax2.tick_params(axis="x", rotation=45, labelsize=8)
    ax2.tick_params(axis="y", rotation=0, labelsize=8)

    plt.suptitle("Task Gradient Cosine Similarity: PCGrad Effect", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Conflict vs AUC Correlation
# ─────────────────────────────────────────────────────────────────────────────

def plot_conflict_vs_auc_improvement(
    task_conflict_scores: Dict[str, float],
    task_auc_deltas: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: Tuple = (8, 6),
) -> plt.Figure:
    """
    Scatter plot: per-task conflict score vs AUC improvement.

    Hypothesis: tasks with more gradient conflict benefit more from PCGrad.

    Parameters
    ----------
    task_conflict_scores : dict[task_name -> avg conflict score]
        Higher = more conflict (e.g. fraction of negative cosine similarities)
    task_auc_deltas : dict[task_name -> AUC(PCGrad) - AUC(baseline)]
    """
    common_tasks = set(task_conflict_scores.keys()) & set(task_auc_deltas.keys())

    if not common_tasks:
        print("No common tasks between conflict scores and AUC deltas.")
        return None

    x = [task_conflict_scores[t] for t in common_tasks]
    y = [task_auc_deltas[t] for t in common_tasks]
    names = list(common_tasks)

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#4ECDC4" if delta >= 0 else "#FF6B6B" for delta in y]
    ax.scatter(x, y, c=colors, s=100, edgecolors="black", linewidth=0.5, zorder=3)

    # Labels
    for xi, yi, name in zip(x, y, names):
        short = "_".join(name.split("_")[1:])
        ax.annotate(short, (xi, yi), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)

    # Trend line
    if len(x) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x), max(x), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.4, linewidth=1.5, label="Trend")

    # Pearson correlation
    from scipy import stats as scipy_stats
    r, pval = scipy_stats.pearsonr(x, y)
    ax.text(0.05, 0.95,
            f"r = {r:.3f}, p = {pval:.3f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Task Gradient Conflict Score")
    ax.set_ylabel("Δ ROC-AUC (PCGrad vs Hard Sharing)")
    ax.set_title("Gradient Conflict vs AUC Improvement")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Full Analysis Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pcgrad_analysis(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_names: List[str],
    output_dir: str = "results/pcgrad_analysis",
    num_batches: int = 20,
):
    """
    Run complete PCGrad gradient analysis and save all outputs.

    Generates:
      - pairwise_cosine_before.png
      - conflict_stats.json
      - conflict_matrix.pkl
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Collecting task gradients...")
    task_grads = collect_task_gradients_per_batch(
        model, loader, device, num_batches
    )

    print(f"  Tasks with gradients: {list(task_grads.keys())}")

    # Mean gradients
    mean_grads = compute_mean_task_gradients(task_grads)

    # Pairwise matrices
    cosine_matrix, conflict_matrix, labels = compute_pairwise_conflict_matrix(
        mean_grads, task_names
    )

    # After-PCGrad simulation
    from training.pcgrad import project_conflicting_gradients
    task_ids = sorted(mean_grads.keys())
    grad_list = [mean_grads[tid] for tid in task_ids]
    projected = []

    for i in range(len(grad_list)):
        g = grad_list[i].clone()
        for j in range(len(grad_list)):
            if i == j:
                continue
            dot = torch.dot(g, grad_list[j])
            if dot < 0:
                norm_sq = torch.dot(grad_list[j], grad_list[j])
                if norm_sq > 1e-12:
                    g = g - (dot / norm_sq) * grad_list[j]
        projected.append(g)

    mean_grads_after = {tid: projected[i] for i, tid in enumerate(task_ids)}
    cosine_after, _, _ = compute_pairwise_conflict_matrix(mean_grads_after, task_names)

    # Conflict statistics
    n = len(task_ids)
    upper_tri = [(i, j) for i in range(n) for j in range(i + 1, n)]

    n_conflicts_before = sum(1 for i, j in upper_tri if cosine_matrix[i, j] < 0)
    n_conflicts_after = sum(1 for i, j in upper_tri if cosine_after[i, j] < 0)
    n_pairs = len(upper_tri)

    cosines_before_vals = [cosine_matrix[i, j] for i, j in upper_tri]
    cosines_after_vals = [cosine_after[i, j] for i, j in upper_tri]

    stats = {
        "num_tasks": n,
        "num_pairs": n_pairs,
        "num_conflicts_before": n_conflicts_before,
        "num_conflicts_after": n_conflicts_after,
        "conflict_ratio_before": n_conflicts_before / n_pairs,
        "conflict_ratio_after": n_conflicts_after / n_pairs,
        "conflict_reduction_pct": 100 * (n_conflicts_before - n_conflicts_after) / max(n_conflicts_before, 1),
        "avg_cosine_before": float(np.mean(cosines_before_vals)),
        "avg_cosine_after": float(np.mean(cosines_after_vals)),
        "std_cosine_before": float(np.std(cosines_before_vals)),
        "std_cosine_after": float(np.std(cosines_after_vals)),
    }

    print("\nGradient Conflict Statistics:")
    print(f"  Conflict pairs before PCGrad: {n_conflicts_before}/{n_pairs} "
          f"({stats['conflict_ratio_before']:.1%})")
    print(f"  Conflict pairs after PCGrad:  {n_conflicts_after}/{n_pairs} "
          f"({stats['conflict_ratio_after']:.1%})")
    print(f"  Conflict reduction:           {stats['conflict_reduction_pct']:.1f}%")
    print(f"  Mean cosine before:           {stats['avg_cosine_before']:.4f}")
    print(f"  Mean cosine after:            {stats['avg_cosine_after']:.4f}")

    # Save outputs
    with open(os.path.join(output_dir, "conflict_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(output_dir, "conflict_matrix.pkl"), "wb") as f:
        pickle.dump({
            "cosine_before": cosine_matrix,
            "cosine_after": cosine_after,
            "conflict_matrix": conflict_matrix,
            "labels": labels,
            "task_ids": task_ids,
        }, f)

    # Plots
    plot_pairwise_conflict_heatmap(
        cosine_matrix, labels,
        title="Task Gradient Cosine Similarity (Before PCGrad)",
        save_path=os.path.join(output_dir, "pairwise_cosine_before.png"),
    )

    plot_conflict_before_after(
        cosine_matrix, cosine_after, labels,
        save_path=os.path.join(output_dir, "pcgrad_before_after.png"),
    )

    print(f"\nAll outputs saved to: {output_dir}/")
    return stats, cosine_matrix, cosine_after, labels