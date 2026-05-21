"""
Quantitative evaluation of explanation quality.

Metrics:
- Fidelity+ / Fidelity-
- Sparsity (fraction of unmasked edges)
- Stability (cross-run consistency)
- AUC-based ranking quality (if ground-truth available)
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from scipy.stats import spearmanr


def compute_sparsity(edge_mask: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Fraction of edges with importance below threshold.
    Higher sparsity = more focused explanation.
    """
    below = (edge_mask < threshold).float().mean()
    return below.item()


def compute_explanation_auc(
    edge_mask: torch.Tensor,
    ground_truth_edges: List[Tuple[int, int]],
    edge_index: torch.Tensor,
) -> float:
    """
    If ground-truth important edges are known (e.g., from mutagenicity),
    compute AUC of explanation vs ground truth.

    ground_truth_edges: list of (i, j) pairs known to be important
    """
    from sklearn.metrics import roc_auc_score

    # Build ground truth binary vector
    src, dst = edge_index.numpy()
    gt = np.zeros(edge_mask.shape[0])
    gt_set = set(
        (min(i, j), max(i, j)) for i, j in ground_truth_edges
    )

    for k in range(src.shape[0]):
        key = (min(src[k], dst[k]), max(src[k], dst[k]))
        if key in gt_set:
            gt[k] = 1.0

    if gt.sum() == 0 or gt.sum() == len(gt):
        return 0.5  # Degenerate case

    try:
        return roc_auc_score(gt, edge_mask.numpy())
    except Exception:
        return 0.5


def compute_all_metrics(
    explainer,
    model,
    data_list: List,
    task_idx: int,
    n_stability_runs: int = 3,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, float]:
    """
    Compute comprehensive explanation quality metrics.

    Args:
        explainer: GNNExplainer instance
        model: trained model
        data_list: list of molecule Data objects
        task_idx: task to explain
        n_stability_runs: runs for stability estimation
        device: computation device

    Returns:
        dict of aggregated metrics
    """
    from feature3.explainer.mask_utils import (
        mask_fidelity, compute_mask_stability
    )

    all_fidelity_plus = []
    all_fidelity_minus = []
    all_sparsity = []
    all_stability = []

    for data in data_list[:20]:  # Evaluate on subset for speed
        # Generate explanation
        exp = explainer.explain(data, task_idx, device=device)
        edge_mask = exp['edge_mask']

        # Fidelity
        fid = mask_fidelity(model, data, edge_mask, task_idx,
                            device=device)
        all_fidelity_plus.append(fid['fidelity_plus'])
        all_fidelity_minus.append(fid['fidelity_minus'])

        # Sparsity
        all_sparsity.append(compute_sparsity(edge_mask))

        # Stability (fewer runs for speed)
        if n_stability_runs > 1:
            stab = compute_mask_stability(
                explainer, data, task_idx, n_runs=n_stability_runs,
                device=device
            )
            all_stability.append(stab['mean_pairwise_correlation'])

    results = {
        'fidelity_plus_mean': float(np.mean(all_fidelity_plus)),
        'fidelity_plus_std': float(np.std(all_fidelity_plus)),
        'fidelity_minus_mean': float(np.mean(all_fidelity_minus)),
        'fidelity_minus_std': float(np.std(all_fidelity_minus)),
        'sparsity_mean': float(np.mean(all_sparsity)),
        'sparsity_std': float(np.std(all_sparsity)),
        'num_molecules_evaluated': len(all_fidelity_plus),
    }

    if all_stability:
        results['stability_mean'] = float(np.mean(all_stability))
        results['stability_std'] = float(np.std(all_stability))

    return results