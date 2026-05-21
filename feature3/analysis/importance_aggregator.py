"""
ImportanceAggregator
====================
Aggregates explanation results across molecules and tasks.
Provides dataset-level insights from GNNExplainer outputs.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class ImportanceAggregator:
    """
    Aggregates atom/edge importance across molecules and tasks.

    Example:
        agg = ImportanceAggregator()
        agg.add_task_explanations(task_idx=0, explanations=exp_list)
        feature_importance = agg.get_feature_importance(task_idx=0)
    """

    def __init__(self):
        # {task_idx: List[Dict]} where each Dict is an explanation
        self._task_explanations: Dict[int, List[Dict]] = {}

    def add_task_explanations(
        self,
        task_idx: int,
        explanations: List[Dict],
    ) -> None:
        """Add explanations for a single task."""
        self._task_explanations[task_idx] = explanations

    def get_feature_importance(
        self,
        task_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Average node feature mask across molecules for a task.

        Returns:
            {
                'mean': [F] mean feature importance,
                'std': [F] std feature importance
            }
        """
        if task_idx not in self._task_explanations:
            return {}

        exps = self._task_explanations[task_idx]
        feat_masks = [
            e['node_feat_mask']
            for e in exps
            if 'node_feat_mask' in e
        ]

        if not feat_masks:
            return {}

        stacked = torch.stack(feat_masks)  # [N_mol, F]
        return {
            'mean': stacked.mean(dim=0),
            'std': stacked.std(dim=0),
        }

    def get_cross_task_feature_importance(
        self,
        task_indices: Optional[List[int]] = None,
        task_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Build matrix [T, F] of mean feature importance per task.

        Useful for understanding which features each task uses.
        """
        tasks = task_indices or list(self._task_explanations.keys())
        matrices = []

        for t in tasks:
            feat = self.get_feature_importance(t)
            if feat:
                matrices.append(feat['mean'].numpy())

        if not matrices:
            return np.array([])

        return np.stack(matrices)  # [T, F]

    def get_average_edge_sparsity(self, task_idx: int) -> float:
        """Average fraction of unimportant edges for a task."""
        if task_idx not in self._task_explanations:
            return 0.0

        sparsities = []
        for exp in self._task_explanations[task_idx]:
            mask = exp.get('edge_mask')
            if mask is not None:
                sparsities.append((mask < 0.5).float().mean().item())

        return float(np.mean(sparsities)) if sparsities else 0.0

    def get_prediction_distribution(
        self,
        task_idx: int,
    ) -> Dict[str, float]:
        """
        Distribution of predictions for explained molecules.

        Returns mean, std, fraction positive.
        """
        if task_idx not in self._task_explanations:
            return {}

        preds = [
            e.get('prediction', 0.5)
            for e in self._task_explanations[task_idx]
        ]

        return {
            'mean': float(np.mean(preds)),
            'std': float(np.std(preds)),
            'fraction_positive': float(np.mean([p > 0.5 for p in preds])),
            'count': len(preds),
        }

    def summary(self) -> Dict[str, Dict]:
        """
        Full summary across all tasks.
        """
        result = {}
        for task_idx, exps in self._task_explanations.items():
            n_success = sum(1 for e in exps if not e.get('failed', False))
            feat = self.get_feature_importance(task_idx)

            result[task_idx] = {
                'n_molecules': len(exps),
                'n_success': n_success,
                'success_rate': n_success / max(len(exps), 1),
                'avg_sparsity': self.get_average_edge_sparsity(task_idx),
                'pred_dist': self.get_prediction_distribution(task_idx),
                'top_feature_idx': (
                    int(feat['mean'].argmax())
                    if feat else -1
                ),
            }

        return result