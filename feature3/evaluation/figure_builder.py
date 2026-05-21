"""
Publication-quality figure generation for Feature 3.

9 figures covering:
1. Atom importance grid (Tox21 tasks)
2. Bond importance grid (ClinTox tasks)
3. Substructure importance bar chart (dataset-level)
4. Fidelity+ / Fidelity- comparison
5. Explanation stability heatmap
6. Task-to-task substructure overlap
7. Feature importance (node features) radar chart
8. Per-task explanation quality metrics
9. Correct vs incorrect prediction explanation comparison
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from typing import Dict, List, Optional
import json


FIGURE_DIR = 'results/feature3/figures/'


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ── Figure 1: Atom importance grid ─────────────────────────────────────

def plot_atom_importance_grid(
    explanations: List[Dict],
    smiles_list: List[str],
    visualizer,
    task_name: str,
    save: bool = True,
) -> plt.Figure:
    """
    3×3 grid of molecules with atom importance heatmaps.
    """
    n = min(9, len(explanations))
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    fig.suptitle(
        f'Atom Importance Heatmaps — {task_name}',
        fontsize=16, fontweight='bold'
    )

    axes_flat = axes.flatten()
    for i in range(n):
        ax = axes_flat[i]
        try:
            img = visualizer.draw_atom_importance(
                smiles_list[i],
                explanations[i]['node_importance'],
            )
            ax.imshow(img)
            pred = explanations[i]['prediction'].item()
            ax.set_title(f'Pred: {pred:.3f}', fontsize=10)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{e}', ha='center', va='center',
                    fontsize=8)
        ax.axis('off')

    for ax in axes_flat[n:]:
        ax.axis('off')

    plt.tight_layout()
    if save:
        ensure_dir(FIGURE_DIR)
        path = os.path.join(FIGURE_DIR, f'fig1_atom_importance_{task_name}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig


# ── Figure 2: Bond importance grid ─────────────────────────────────────

def plot_bond_importance_grid(
    explanations: List[Dict],
    smiles_list: List[str],
    data_list: List,
    visualizer,
    task_name: str,
    save: bool = True,
) -> plt.Figure:
    """
    3×3 grid of molecules with bond importance heatmaps.
    """
    n = min(9, len(explanations))
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    fig.suptitle(
        f'Bond Importance Heatmaps — {task_name}',
        fontsize=16, fontweight='bold'
    )

    axes_flat = axes.flatten()
    for i in range(n):
        ax = axes_flat[i]
        try:
            img = visualizer.draw_bond_importance(
                smiles_list[i],
                data_list[i].edge_index,
                explanations[i]['edge_mask'],
            )
            ax.imshow(img)
            pred = explanations[i]['prediction'].item()
            ax.set_title(f'Pred: {pred:.3f}', fontsize=10)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{e}', ha='center', va='center',
                    fontsize=8)
        ax.axis('off')

    for ax in axes_flat[n:]:
        ax.axis('off')

    plt.tight_layout()
    if save:
        ensure_dir(FIGURE_DIR)
        path = os.path.join(FIGURE_DIR, f'fig2_bond_importance_{task_name}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig


# ── Figure 3: Dataset-level substructure importance ─────────────────────

def plot_substructure_importance(
    group_summary: Dict[str, Dict],
    task_name: str,
    top_k: int = 15,
    save: bool = True,
) -> plt.Figure:
    """
    Horizontal bar chart: mean importance per functional group
    across all molecules in dataset.
    """
    # Filter to present groups and sort by mean importance
    present = {
        name: info for name, info in group_summary.items()
        if info['num_molecules'] > 0
    }
    sorted_items = sorted(
        present.items(),
        key=lambda x: x[1]['mean_importance'],
        reverse=True
    )[:top_k]

    names = [item[0].replace('_', ' ').title() for item in sorted_items]
    means = [item[1]['mean_importance'] for item in sorted_items]
    stds = [item[1]['std_importance'] for item in sorted_items]
    freqs = [item[1]['frequency'] for item in sorted_items]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        f'Substructure Importance — {task_name}',
        fontsize=14, fontweight='bold'
    )

    # Left: importance bars
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))[::-1]
    bars = ax1.barh(names, means, xerr=stds, color=colors,
                    capsize=3, height=0.7, error_kw={'linewidth': 1.5})
    ax1.set_xlabel('Mean Importance Score', fontsize=12)
    ax1.set_title('Average Atom Importance', fontsize=12)
    ax1.set_xlim(0, 1.15)
    ax1.axvline(x=0.5, color='grey', linestyle='--', alpha=0.6,
                label='Threshold=0.5')
    ax1.invert_yaxis()
    ax1.legend(fontsize=9)

    # Right: frequency bars
    ax2.barh(names, freqs, color='steelblue', height=0.7, alpha=0.75)
    ax2.set_xlabel('Molecule Frequency', fontsize=12)
    ax2.set_title('Occurrence in Dataset', fontsize=12)
    ax2.set_xlim(0, 1.05)
    ax2.axvline(x=0.5, color='grey', linestyle='--', alpha=0.6)
    ax2.invert_yaxis()

    plt.tight_layout()
    if save:
        ensure_dir(FIGURE_DIR)
        path = os.path.join(FIGURE_DIR, f'fig3_substructure_importance.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig


# ── Figure 4: Fidelity+ / Fidelity- comparison ─────────────────────────

def plot_fidelity_comparison(
    fidelity_results: Dict[str, Dict],
    save: bool = True,
) -> plt.Figure:
    """
    Grouped bar chart comparing Fidelity+ and Fidelity- across tasks.
    """
    tasks = list(fidelity_results.keys())
    fid_plus = [fidelity_results[t]['fidelity_plus_mean'] for t in tasks]
    fid_minus = [fidelity_results[t]['fidelity_minus_mean'] for t in tasks]
    fid_plus_std = [fidelity_results[t]['fidelity_plus_std'] for t in tasks]
    fid_minus_std = [fidelity_results[t]['fidelity_minus_std'] for t in tasks]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(tasks) * 0.8), 6))

    bars1 = ax.bar(x - width / 2, fid_plus, width, yerr=fid_plus_std,
                   label='Fidelity+', color='#2ecc71', capsize=4,
                   error_kw={'linewidth': 1.5})
    bars2 = ax.bar(x + width / 2, fid_minus, width, yerr=fid_minus_std,
                   label='Fidelity−', color='#e74c3c', capsize=4,
                   error_kw={'linewidth': 1.5})

    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Fidelity Score', fontsize=12)
    ax.set_title(
        'Explanation Fidelity by Task\n'
        'Fidelity+: important subgraph captures prediction\n'
        'Fidelity−: removing important subgraph changes prediction',
        fontsize=12
    )
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save:
        ensure_dir(FIGURE_DIR)
        path = os.path.join(FIGURE_DIR, 'fig4_fidelity_comparison.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig


# ── Figure 5: Stability heatmap ─────────────────────────────────────────

def plot_stability_heatmap(
    stability_results: Dict[str, Dict],
    save: bool = True,
) -> plt.Figure:
    """
    Heatmap of explanation stability (pairwise correlation)
    across tasks and molecule subsets.
    """
    tasks = list(stability_results.keys())
    corrs = [
        stability_results[t].get('mean_pairwise_correlation', 0)
        for t in tasks
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(tasks))
    colors = plt.cm.RdYlGn(np.array(corrs))
    bars = ax.bar(x, corrs, color=colors, width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean Pairwise Correlation', fontsize=12)
    ax.set_title(
        'Explanation Stability by Task\n'
        '(Higher = more consistent across random seeds)',
        fontsize=12
    )
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.7, color='grey', linestyle='--', alpha=0.6,
               label='Stability threshold (0.7)')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Value labels
    for bar, val in zip(bars, corrs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{val:.2f}',
            ha='center', va='bottom', fontsize=8
        )

    plt.tight_layout()
    if save:
        ensure_dir(FIGURE_DIR)
        path = os.path.join(FIGURE_DIR, 'fig5_stability_heatmap.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig


# ── Figure 6: Cross-task substructure overlap ───────────────────────────

def plot_cross_task_overlap(
    task_group_scores: Dict[str, Dict[str, float]],
    top_groups: int = 10,
    save: bool = True,
) -> plt.Figure:
    """
    Heatmap: rows = functional groups, cols = tasks
    Cell value = mean importance of group for that task.

    Reveals which substructures are universally vs task-specifically important.
    """
    tasks = list(task_group_scores.keys())

    # Get union of top groups across all tasks
    all_groups = set()
    for task_scores in task_group_scores.values():
        sorted_g = sorted(task_scores.items(), key=lambda x: x[1],
                          reverse=True)[:top_groups]
        all_groups.update([g[0] for g in sorted_g])

    groups = sorted(all_groups)

    # Build matrix
    matrix = np.zeros((len(groups), len(tasks)))
    for j, task in enumerate(tasks):
        for i, group in enumerate(groups):
            matrix[i, j] = task_group_scores[task].get(group, 0.0)

    fig, ax = plt.subplots(figsize=(max(10, len(tasks)), max(8, len(groups) * 0.5)))

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels(tasks, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(
        [g.replace('_', ' ').title() for g in groups], fontsize=9
    )
    ax.set_title(
        'Functional Group Importance Across Tasks',
        fontsize=13, fontweight='bold'
    )

    plt.colorbar(im, ax=ax, label='Mean Importance Score', fraction=0.02)

    # Annotate cells
    for i in range(len(groups)):
        for j in range(len(tasks)):
            val = matrix[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color='black' if val < 0.7 else 'white')

    plt.tight_layout()
    if save:
        ensure_dir(FIGURE_DIR)
        path = os.path.join(FIGURE_DIR, 'fig6_cross_task_overlap.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig


# ── Figure 7: Node feature importance radar ─────────────────────────────

def plot_feature_importance_radar(
    feature_names: List[str],
    feature_importance: np.ndarray,
    task_names: List[str],
    save: bool = True,
) -> plt.Figure:
    """
    Radar chart comparing node feature importance across tasks.

    feature_importance: [T, F] mean importance per task and feature.
    Shows which atom features (charge, hybridization, etc.) drive predictions.
    """
    # Select top features for readability
    top_k = min(12, len(feature_names))
    mean_imp = feature_importance.mean(axis=0)
    top_idx = np.argsort(mean_imp)[-top_k:][::-1]

    selected_names = [feature_names[i] for i in top_idx]
    selected_imp = feature_importance[:, top_idx]

    n_feat = len(selected_names)
    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]  # Close polygon

    fig, ax = plt.subplots(
        figsize=(8, 8), subplot_kw=dict(polar=True)
    )
    colors = plt.cm.tab10(np.linspace(0, 1, len(task_names)))

    for task_idx, (task, color) in enumerate(zip(task_names, colors)):
        values = selected_imp[task_idx].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2,
                color=color, label=task, alpha=0.7)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(selected_names, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title('Node Feature Importance by Task',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    if save:
        ensure_dir(FIGURE_DIR)
        path = os.path.join(FIGURE_DIR, 'fig7_feature_importance_radar.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig


# ── Figure 8: Per-task explanation metrics summary ──────────────────────

def plot_per_task_metrics(
    metrics_by_task: Dict[str, Dict],
    save: bool = True,
) -> plt.Figure:
    """
    Multi-panel summary of explanation quality per task.
    Panels: Fidelity+, Fidelity-, Sparsity, Stability
    """
    tasks = list(metrics_by_task.keys())
    metric_keys = ['fidelity_plus_mean', 'fidelity_minus_mean',
                   'sparsity_mean', 'stability_mean']
    metric_labels = ['Fidelity+', 'Fidelity−', 'Sparsity', 'Stability']
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Explanation Quality Metrics by Task',
                 fontsize=14, fontweight='bold')

    x = np.arange(len(tasks))
    for ax, key, label, color in zip(
        axes.flatten(), metric_keys, metric_labels, colors
    ):
        values = [metrics_by_task[t].get(key, 0) for t in tasks]
        std_key = key.replace('_mean', '_std')
        stds = [metrics_by_task[t].get(std_key, 0) for t in tasks]

        bars = ax.bar(x, values, yerr=stds, color=color, capsize=3,
                      width=0.6, error_kw={'linewidth': 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha='right', fontsize=8)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7
            )

    plt.tight_layout()
    if save:
        ensure_dir(FIGURE_DIR)
        path = os.path.join(FIGURE_DIR, 'fig8_per_task_metrics.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig


# ── Figure 9: Correct vs incorrect prediction comparison ────────────────

def plot_correct_vs_incorrect(
    correct_explanations: List[Dict],
    incorrect_explanations: List[Dict],
    correct_smiles: List[str],
    incorrect_smiles: List[str],
    visualizer,
    task_name: str,
    save: bool = True,
) -> plt.Figure:
    """
    Side-by-side comparison: correct vs incorrect prediction explanations.

    Shows whether explanations differ qualitatively for wrong predictions.
    """
    n_show = min(3, len(correct_explanations), len(incorrect_explanations))

    fig, axes = plt.subplots(2, n_show, figsize=(5 * n_show, 10))
    fig.suptitle(
        f'Correct vs Incorrect Predictions — {task_name}',
        fontsize=14, fontweight='bold'
    )

    for i in range(n_show):
        # Row 0: correct predictions
        ax = axes[0, i] if n_show > 1 else axes[0]
        try:
            img = visualizer.draw_atom_importance(
                correct_smiles[i],
                correct_explanations[i]['node_importance'],
            )
            ax.imshow(img)
            pred = correct_explanations[i]['prediction'].item()
            ax.set_title(f'✓ Correct\nPred: {pred:.3f}', fontsize=10,
                         color='green')
        except Exception:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center')
        ax.axis('off')

        # Row 1: incorrect predictions
        ax = axes[1, i] if n_show > 1 else axes[1]
        try:
            img = visualizer.draw_atom_importance(
                incorrect_smiles[i],
                incorrect_explanations[i]['node_importance'],
            )
            ax.imshow(img)
            pred = incorrect_explanations[i]['prediction'].item()
            ax.set_title(f'✗ Incorrect\nPred: {pred:.3f}', fontsize=10,
                         color='red')
        except Exception:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center')
        ax.axis('off')

    plt.tight_layout()
    if save:
        ensure_dir(FIGURE_DIR)
        path = os.path.join(FIGURE_DIR, 'fig9_correct_vs_incorrect.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig