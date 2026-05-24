"""
FigureBuilder
=============
Generates all 9 publication-quality figures for Feature 3.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from PIL import Image
import torch

from feature3.visualization.mol_visualizer import MoleculeVisualizer


class FigureBuilder:
    """
    Builds all Feature 3 publication figures.

    Args:
        output_dir: Directory to save figures
        dpi: Figure DPI (300 for publication)
        colormap: Colormap for heatmaps
    """

    def __init__(
        self,
        output_dir: str = 'results/feature3/figures',
        dpi: int = 300,
        colormap: str = 'RdYlGn',
    ):
        self.output_dir = output_dir
        self.dpi = dpi
        self.colormap = colormap
        self.visualizer = MoleculeVisualizer(colormap=colormap)
        os.makedirs(output_dir, exist_ok=True)

    def _save(self, fig: plt.Figure, filename: str) -> str:
        """Save figure and return path."""
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  [Fig] Saved: {path}")
        return path

    # ── Figure 1: Atom Importance Grid ──────────────────────────────────

    def fig1_atom_importance_grid(
        self,
        smiles_list: List[str],
        explanations: List[Dict],
        task_name: str,
        n_cols: int = 3,
    ) -> str:
        """
        Grid of molecule drawings with atom importance heatmaps.
        """
        n = min(9, len(smiles_list))
        n_rows = (n + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 5 * n_rows)
        )
        fig.suptitle(
            f'Atom Importance Heatmaps — {task_name}',
            fontsize=16, fontweight='bold'
        )

        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        for i in range(n):
            ax = axes_flat[i]
            try:
                img = self.visualizer.draw_atom_importance(
                    smiles_list[i],
                    explanations[i]['node_importance'],
                )
                if img:
                    ax.imshow(img)
                pred = explanations[i].get('prediction', 0.5)
                tgt = int(explanations[i].get('target', 0))
                color = 'green' if round(pred) == tgt else 'red'
                ax.set_title(
                    f'Pred: {pred:.2f}  |  True: {tgt}',
                    fontsize=9, color=color,
                )
            except Exception as e:
                ax.text(0.5, 0.5, f'Error:\n{e}',
                        ha='center', va='center', fontsize=7)
            ax.axis('off')

        for ax in axes_flat[n:]:
            ax.axis('off')

        # Colorbar
        sm = plt.cm.ScalarMappable(
            cmap=self.colormap,
            norm=plt.Normalize(0, 1)
        )
        sm.set_array([])
        fig.colorbar(
            sm, ax=axes_flat[:n],
            orientation='vertical',
            fraction=0.015, pad=0.02,
            label='Atom Importance',
        )

        plt.tight_layout()
        return self._save(fig, 'fig1_atom_importance_grid.png')

    # ── Figure 2: Bond Importance Grid ──────────────────────────────────

    def fig2_bond_importance_grid(
        self,
        smiles_list: List[str],
        explanations: List[Dict],
        data_list: List,
        task_name: str,
    ) -> str:
        """Grid of bond importance heatmaps."""
        n = min(9, len(smiles_list))
        n_rows = (n + 2) // 3

        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
        fig.suptitle(
            f'Bond Importance Heatmaps — {task_name}',
            fontsize=16, fontweight='bold'
        )

        axes_flat = np.array(axes).flatten()

        for i in range(n):
            ax = axes_flat[i]
            try:
                img = self.visualizer.draw_edge_importance(
                    smiles_list[i],
                    data_list[i].edge_index,
                    explanations[i]['edge_mask'],
                )
                if img:
                    ax.imshow(img)
                pred = explanations[i].get('prediction', 0.5)
                ax.set_title(f'Pred: {pred:.2f}', fontsize=9)
            except Exception as e:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
            ax.axis('off')

        for ax in axes_flat[n:]:
            ax.axis('off')

        plt.tight_layout()
        return self._save(fig, 'fig2_bond_importance_grid.png')

    # ── Figure 3: Substructure Importance Bar Chart ──────────────────────

    def fig3_substructure_importance(
        self,
        group_summary: Dict[str, Dict],
        task_name: str,
        top_k: int = 15,
    ) -> str:
        """
        Horizontal bar chart of functional group importance.
        Left: mean importance with std error bars
        Right: frequency in dataset
        """
        # Filter and sort
        present = {
            k: v for k, v in group_summary.items()
            if v.get('count', 0) > 0
        }
        sorted_items = sorted(
            present.items(),
            key=lambda x: x[1]['mean'],
            reverse=True
        )[:top_k]

        if not sorted_items:
            print("  [Fig3] No substructure data available")
            return ''

        names = [
            it[0].replace('_', ' ').title()
            for it in sorted_items
        ]
        means = [it[1]['mean'] for it in sorted_items]
        stds = [it[1]['std'] for it in sorted_items]
        freqs = [it[1]['frequency'] for it in sorted_items]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(names) * 0.45)))
        fig.suptitle(
            f'Functional Group Importance — {task_name}',
            fontsize=14, fontweight='bold'
        )

        colors = cm.get_cmap(self.colormap)(
            np.linspace(0.9, 0.3, len(names))
        )

        # Left: importance
        ax1.barh(
            names, means,
            xerr=stds,
            color=colors,
            capsize=3,
            error_kw={'linewidth': 1.5},
            height=0.65,
        )
        ax1.set_xlabel('Mean Importance Score', fontsize=12)
        ax1.set_title('Atom Importance per Group', fontsize=12)
        ax1.set_xlim(0, 1.15)
        ax1.axvline(0.5, color='grey', linestyle='--', alpha=0.5)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)

        for j, (m, s) in enumerate(zip(means, stds)):
            ax1.text(min(m + s + 0.02, 1.1), j, f'{m:.3f}',
                     va='center', fontsize=8)

        # Right: frequency
        ax2.barh(names, freqs, color='steelblue', height=0.65, alpha=0.8)
        ax2.set_xlabel('Molecule Frequency', fontsize=12)
        ax2.set_title('Occurrence in Explained Set', fontsize=12)
        ax2.set_xlim(0, 1.1)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        return self._save(fig, 'fig3_substructure_importance.png')

    # ── Figure 4: Fidelity Comparison ──────────────────────────────────

    def fig4_fidelity_comparison(
        self,
        metrics_by_task: Dict[str, Dict],
    ) -> str:
        """
        Grouped bar chart: Fidelity+ and Fidelity- per task.
        """
        tasks = list(metrics_by_task.keys())
        n = len(tasks)
        x = np.arange(n)
        width = 0.35

        fp = [metrics_by_task[t].get('fidelity_plus_mean', 0) for t in tasks]
        fm = [metrics_by_task[t].get('fidelity_minus_mean', 0) for t in tasks]
        fp_s = [metrics_by_task[t].get('fidelity_plus_std', 0) for t in tasks]
        fm_s = [metrics_by_task[t].get('fidelity_minus_std', 0) for t in tasks]

        fig, ax = plt.subplots(figsize=(max(10, n), 6))

        ax.bar(x - width / 2, fp, width, yerr=fp_s,
               label='Fidelity+ (important subgraph)',
               color='#2ecc71', capsize=4, error_kw={'lw': 1.5})
        ax.bar(x + width / 2, fm, width, yerr=fm_s,
               label='Fidelity− (removed subgraph)',
               color='#e74c3c', capsize=4, error_kw={'lw': 1.5})

        ax.set_xlabel('Task', fontsize=12)
        ax.set_ylabel('Fidelity Score', fontsize=12)
        ax.set_title(
            'Explanation Fidelity by Task\n'
            'Higher = explanation faithfully reflects prediction',
            fontsize=13, fontweight='bold'
        )
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.axhline(0.5, color='grey', linestyle='--', alpha=0.4)
        ax.grid(axis='y', alpha=0.3)

        # Value labels
        for xi, (f1, f2) in enumerate(zip(fp, fm)):
            ax.text(xi - width/2, f1 + fm_s[xi] + 0.02,
                    f'{f1:.2f}', ha='center', fontsize=7)
            ax.text(xi + width/2, f2 + fp_s[xi] + 0.02,
                    f'{f2:.2f}', ha='center', fontsize=7)

        plt.tight_layout()
        return self._save(fig, 'fig4_fidelity_comparison.png')

    # ── Figure 5: Explanation Stability ─────────────────────────────────

    def fig5_stability(
        self,
        stability_by_task: Dict[str, float],
    ) -> str:
        """
        Bar chart of explanation stability (pairwise correlation) per task.
        """
        tasks = list(stability_by_task.keys())
        corrs = [stability_by_task[t] for t in tasks]
        colors = cm.get_cmap(self.colormap)(np.array(corrs))

        fig, ax = plt.subplots(figsize=(max(8, len(tasks)), 5))

        bars = ax.bar(tasks, corrs, color=colors, width=0.6)
        ax.set_ylabel('Mean Pairwise Correlation', fontsize=12)
        ax.set_title(
            'Explanation Stability by Task\n'
            'Higher = explanations are consistent across random seeds',
            fontsize=13, fontweight='bold'
        )
        ax.set_ylim(0, 1.1)
        ax.axhline(0.7, color='grey', linestyle='--', alpha=0.6,
                   label='Stability threshold (0.7)')
        ax.legend(fontsize=10)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, corrs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', fontsize=8,
            )

        plt.tight_layout()
        return self._save(fig, 'fig5_stability.png')

    # ── Figure 6: Cross-Task Substructure Heatmap ────────────────────────

    def fig6_cross_task_heatmap(
        self,
        task_group_scores: Dict[str, Dict[str, float]],
        top_k_groups: int = 12,
    ) -> str:
        """
        Heatmap: rows = functional groups, cols = tasks.
        Reveals task-specific vs universal chemical motifs.
        """
        tasks = list(task_group_scores.keys())

        # Find top groups across all tasks
        all_scores: Dict[str, List[float]] = {}
        for task_scores in task_group_scores.values():
            for group, score in task_scores.items():
                if group not in all_scores:
                    all_scores[group] = []
                all_scores[group].append(score)

        # Rank by mean importance
        group_means = {
            g: np.mean(s) for g, s in all_scores.items() if s
        }
        top_groups = sorted(
            group_means.keys(),
            key=lambda g: group_means[g],
            reverse=True
        )[:top_k_groups]

        # Build matrix
        matrix = np.zeros((len(top_groups), len(tasks)))
        for j, task in enumerate(tasks):
            for i, group in enumerate(top_groups):
                matrix[i, j] = task_group_scores[task].get(group, 0.0)

        fig_h = max(8, len(top_groups) * 0.5)
        fig_w = max(10, len(tasks) * 0.7)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto',
                       vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(tasks)))
        ax.set_xticklabels(tasks, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(np.arange(len(top_groups)))
        ax.set_yticklabels(
            [g.replace('_', ' ').title() for g in top_groups],
            fontsize=9,
        )
        ax.set_title(
            'Functional Group Importance Across Tasks',
            fontsize=14, fontweight='bold',
        )

        plt.colorbar(im, ax=ax, label='Mean Importance Score', fraction=0.02)

        for i in range(len(top_groups)):
            for j in range(len(tasks)):
                val = matrix[i, j]
                ax.text(
                    j, i, f'{val:.2f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if val > 0.65 else 'black',
                )

        plt.tight_layout()
        return self._save(fig, 'fig6_cross_task_heatmap.png')

    # ── Figure 7: Node Feature Importance Radar ──────────────────────────

    def fig7_feature_radar(
        self,
        feature_importance_by_task: Dict[str, np.ndarray],
        feature_groups: Optional[Dict[str, List[int]]] = None,
    ) -> str:
        """
        Radar chart comparing node feature group importance per task.
        Groups the 129 features into categories for readability.
        """
        # Default grouping for F1's 129-dim features
        if feature_groups is None:
            feature_groups = {
                'Atom Type': list(range(44)),
                'Degree': list(range(44, 55)),
                'Valence': list(range(55, 62)),
                'Charge': list(range(62, 67)),
                'Num Hs': list(range(67, 72)),
                'Hybridization': list(range(72, 77)),
                'Aromaticity': [77],
                'Ring': list(range(78, 85)),
                'Chirality': list(range(85, 89)),
            }

        tasks = list(feature_importance_by_task.keys())
        categories = list(feature_groups.keys())
        n_cats = len(categories)

        # Compute grouped importance
        task_cat_scores = {}
        for task, feat_imp in feature_importance_by_task.items():
            scores = []
            for cat, indices in feature_groups.items():
                valid = [i for i in indices if i < len(feat_imp)]
                scores.append(float(np.mean(feat_imp[valid])) if valid else 0.0)
            task_cat_scores[task] = scores

        # Radar angles
        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
        colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(tasks)))

        for (task, scores), color in zip(task_cat_scores.items(), colors):
            vals = scores + scores[:1]
            ax.plot(angles, vals, 'o-', linewidth=2,
                    color=color, label=task, alpha=0.8)
            ax.fill(angles, vals, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(
            'Node Feature Importance by Task',
            fontsize=14, fontweight='bold', pad=25,
        )
        ax.legend(
            loc='upper right',
            bbox_to_anchor=(1.35, 1.1),
            fontsize=9,
        )

        plt.tight_layout()
        return self._save(fig, 'fig7_feature_radar.png')

    # ── Figure 8: Per-Task Metrics Summary ───────────────────────────────

    def fig8_per_task_metrics(
        self,
        metrics_by_task: Dict[str, Dict],
    ) -> str:
        """
        4-panel summary of Fidelity+, Fidelity-, Sparsity, Stability.
        """
        tasks = list(metrics_by_task.keys())
        x = np.arange(len(tasks))

        panels = [
            ('fidelity_plus_mean', 'fidelity_plus_std', 'Fidelity+', '#2ecc71'),
            ('fidelity_minus_mean', 'fidelity_minus_std', 'Fidelity−', '#e74c3c'),
            ('sparsity_mean', 'sparsity_std', 'Sparsity', '#3498db'),
            ('stability_mean', 'stability_std', 'Stability', '#f39c12'),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            'Explanation Quality Metrics by Task',
            fontsize=14, fontweight='bold',
        )

        for ax, (key, std_key, label, color) in zip(axes.flatten(), panels):
            vals = [metrics_by_task[t].get(key, 0) for t in tasks]
            stds = [metrics_by_task[t].get(std_key, 0) for t in tasks]

            bars = ax.bar(x, vals, yerr=stds, color=color,
                          capsize=3, width=0.6,
                          error_kw={'linewidth': 1.5})
            ax.set_xticks(x)
            ax.set_xticklabels(tasks, rotation=45, ha='right', fontsize=8)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.2)
            ax.axhline(0.5, color='grey', linestyle='--', alpha=0.4)
            ax.grid(axis='y', alpha=0.3)

            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{v:.2f}', ha='center', fontsize=7,
                )

        plt.tight_layout()
        return self._save(fig, 'fig8_per_task_metrics.png')

    # ── Figure 9: Correct vs Incorrect ───────────────────────────────────

    def fig9_correct_vs_incorrect(
        self,
        correct_smiles: List[str],
        correct_exps: List[Dict],
        incorrect_smiles: List[str],
        incorrect_exps: List[Dict],
        task_name: str,
        n_show: int = 3,
    ) -> str:
        """
        Side-by-side: correctly vs incorrectly predicted molecules.
        Shows whether explanations differ qualitatively.
        """
        n = min(n_show, len(correct_smiles), len(incorrect_smiles))
        fig, axes = plt.subplots(2, n, figsize=(5 * n, 11))
        fig.suptitle(
            f'Correct vs Incorrect Predictions — {task_name}\n'
            f'(Top row: correct | Bottom row: incorrect)',
            fontsize=13, fontweight='bold',
        )

        for col in range(n):
            # Correct
            ax = axes[0, col] if n > 1 else axes[0]
            try:
                img = self.visualizer.draw_atom_importance(
                    correct_smiles[col],
                    correct_exps[col]['node_importance'],
                )
                if img:
                    ax.imshow(img)
                pred = correct_exps[col].get('prediction', 0.5)
                ax.set_title(
                    f'✓ Correct\nPred: {pred:.3f}',
                    fontsize=10, color='green',
                )
            except Exception:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
            ax.axis('off')

            # Incorrect
            ax = axes[1, col] if n > 1 else axes[1]
            try:
                img = self.visualizer.draw_atom_importance(
                    incorrect_smiles[col],
                    incorrect_exps[col]['node_importance'],
                )
                if img:
                    ax.imshow(img)
                pred = incorrect_exps[col].get('prediction', 0.5)
                ax.set_title(
                    f'✗ Incorrect\nPred: {pred:.3f}',
                    fontsize=10, color='red',
                )
            except Exception:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
            ax.axis('off')

        plt.tight_layout()
        return self._save(fig, 'fig9_correct_vs_incorrect.png')