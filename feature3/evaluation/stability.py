"""
Stability evaluation for GNNExplainer.
Measures how consistent explanations are across random seeds.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from torch_geometric.data import Data


class StabilityEvaluator:
    """
    Measures explanation stability across multiple runs.

    High stability = trustworthy, reproducible explanations.
    Low stability = explanations are unreliable.
    """

    def __init__(self, n_runs: int = 5):
        self.n_runs = n_runs

    def evaluate_single(
        self,
        explainer,
        data: Data,
        task_idx: int,
        device: torch.device = torch.device('cpu'),
    ) -> Dict[str, float]:
        """
        Run explainer n_runs times and measure consistency.

        Returns:
            mean_correlation: High = stable
            std_correlation: Low = consistent
        """
        edge_masks = []
        node_masks = []

        for _ in range(self.n_runs):
            exp = explainer.explain(data, task_idx, device=device)
            edge_masks.append(exp['edge_mask'])
            node_masks.append(exp['node_feat_mask'])

        edge_stack = torch.stack(edge_masks)  # [n_runs, E]
        node_stack = torch.stack(node_masks)  # [n_runs, F]

        correlations = []
        for i in range(self.n_runs):
            for j in range(i + 1, self.n_runs):
                e1 = edge_stack[i].numpy()
                e2 = edge_stack[j].numpy()
                if np.std(e1) > 0 and np.std(e2) > 0:
                    c = np.corrcoef(e1, e2)[0, 1]
                    if not np.isnan(c):
                        correlations.append(c)

        return {
            'mean_correlation': float(np.mean(correlations)) if correlations else 0.0,
            'std_correlation': float(np.std(correlations)) if correlations else 0.0,
            'edge_mask_cv': float(
                (edge_stack.std(dim=0) / (edge_stack.mean(dim=0) + 1e-8)).mean()
            ),
            'n_runs': self.n_runs,
        }

    def evaluate_dataset(
        self,
        explainer,
        data_list: List[Data],
        task_idx: int,
        device: torch.device = torch.device('cpu'),
        max_mols: int = 10,
    ) -> Dict[str, float]:
        """
        Average stability across multiple molecules.
        """
        all_corr = []
        for data in data_list[:max_mols]:
            try:
                result = self.evaluate_single(explainer, data, task_idx, device)
                all_corr.append(result['mean_correlation'])
            except Exception:
                continue

        return {
            'stability_mean': float(np.mean(all_corr)) if all_corr else 0.0,
            'stability_std': float(np.std(all_corr)) if all_corr else 0.0,
            'n_evaluated': len(all_corr),
        }