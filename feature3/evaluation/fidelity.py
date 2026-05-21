"""
Fidelity metrics for explanation quality.
"""

import torch
import numpy as np
from typing import Dict, List
from torch_geometric.data import Data


class FidelityEvaluator:
    """
    Computes fidelity metrics for GNNExplainer explanations.

    Fidelity+: How much does the prediction change when we keep
               only the important subgraph?
               High = important subgraph is sufficient for prediction.

    Fidelity-: How much does the prediction change when we remove
               the important subgraph?
               High = removing important subgraph hurts prediction.
    """

    def __init__(
        self,
        model,
        threshold: float = 0.5,
        device: torch.device = torch.device('cpu'),
    ):
        self.model = model
        self.threshold = threshold
        self.device = device

    def evaluate_single(
        self,
        data: Data,
        edge_mask: torch.Tensor,
        task_idx: int,
    ) -> Dict[str, float]:
        """
        Compute fidelity for a single molecule explanation.

        Args:
            data: PyG Data (single molecule)
            edge_mask: [E] importance in [0,1]
            task_idx: Which task

        Returns:
            Dict with fidelity_plus, fidelity_minus, sparsity
        """
        self.model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            # Original (unmasked) prediction
            orig_logit = self.model(data, task_idx=task_idx)
            orig_pred = torch.sigmoid(orig_logit).item()

            # Binary mask
            binary = (edge_mask.to(self.device) >= self.threshold).float()
            n_important = binary.sum().item()
            n_total = binary.shape[0]

            if n_important > 0:
                # Fidelity+: keep only important edges
                pos_logit = self.model(data, task_idx=task_idx, edge_weight=binary)
                pos_pred = torch.sigmoid(pos_logit).item()
            else:
                pos_pred = 0.5

            # Fidelity-: remove important edges (keep unimportant)
            neg_binary = 1.0 - binary
            neg_logit = self.model(data, task_idx=task_idx, edge_weight=neg_binary)
            neg_pred = torch.sigmoid(neg_logit).item()

        fid_plus = abs(orig_pred - pos_pred)
        fid_minus = abs(orig_pred - neg_pred)
        sparsity = 1.0 - (n_important / max(n_total, 1))

        return {
            'original_pred': orig_pred,
            'pos_pred': pos_pred,
            'neg_pred': neg_pred,
            'fidelity_plus': fid_plus,
            'fidelity_minus': fid_minus,
            'fidelity_combined': fid_plus + fid_minus,
            'sparsity': sparsity,
            'n_important_edges': int(n_important),
            'n_total_edges': n_total,
        }

    def evaluate_dataset(
        self,
        data_list: List[Data],
        edge_masks: List[torch.Tensor],
        task_idx: int,
    ) -> Dict[str, float]:
        """
        Evaluate fidelity across a dataset.

        Returns aggregated statistics.
        """
        all_fp, all_fm, all_sp = [], [], []

        for data, mask in zip(data_list, edge_masks):
            try:
                result = self.evaluate_single(data, mask, task_idx)
                all_fp.append(result['fidelity_plus'])
                all_fm.append(result['fidelity_minus'])
                all_sp.append(result['sparsity'])
            except Exception:
                continue

        if not all_fp:
            return {
                'fidelity_plus_mean': 0.0,
                'fidelity_plus_std': 0.0,
                'fidelity_minus_mean': 0.0,
                'fidelity_minus_std': 0.0,
                'sparsity_mean': 0.0,
                'sparsity_std': 0.0,
                'n_evaluated': 0,
            }

        return {
            'fidelity_plus_mean': float(np.mean(all_fp)),
            'fidelity_plus_std': float(np.std(all_fp)),
            'fidelity_minus_mean': float(np.mean(all_fm)),
            'fidelity_minus_std': float(np.std(all_fm)),
            'sparsity_mean': float(np.mean(all_sp)),
            'sparsity_std': float(np.std(all_sp)),
            'n_evaluated': len(all_fp),
        }