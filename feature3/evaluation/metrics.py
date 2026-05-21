"""
Explanation quality metrics.
"""

import torch
import numpy as np
from typing import Dict, List
from torch_geometric.data import Data


def compute_fidelity(
    model,
    data: Data,
    edge_mask: torch.Tensor,
    task_idx: int,
    threshold: float = 0.5,
    device: torch.device = torch.device('cpu')
) -> Dict[str, float]:
    """
    Compute fidelity metrics.
    
    Fidelity+: Prediction with important edges only
    Fidelity-: Prediction without important edges
    """
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        # Original prediction
        orig_pred = torch.sigmoid(model(data, task_idx=task_idx)).item()
        
        # Binary mask
        binary_mask = (edge_mask.to(device) > threshold).float()
        
        # Fidelity+: Keep only important edges (mask out others)
        if binary_mask.sum() > 0:
            pos_pred = torch.sigmoid(
                model(data, task_idx=task_idx, edge_weight=binary_mask)
            ).item()
        else:
            pos_pred = 0.5
        
        # Fidelity-: Remove important edges
        neg_mask = 1.0 - binary_mask
        neg_pred = torch.sigmoid(
            model(data, task_idx=task_idx, edge_weight=neg_mask)
        ).item()
    
    return {
        'original': orig_pred,
        'fidelity_plus': abs(orig_pred - pos_pred),
        'fidelity_minus': abs(orig_pred - neg_pred),
        'fidelity_combined': abs(orig_pred - pos_pred) + abs(orig_pred - neg_pred)
    }


def compute_sparsity(edge_mask: torch.Tensor, threshold: float = 0.5) -> float:
    """Fraction of edges with importance below threshold."""
    return (edge_mask < threshold).float().mean().item()


def compute_stability(
    explainer,
    data: Data,
    task_idx: int,
    n_runs: int = 5,
    device: torch.device = torch.device('cpu')
) -> Dict[str, float]:
    """
    Measure explanation consistency across random seeds.
    """
    masks = []
    for _ in range(n_runs):
        exp = explainer.explain(data, task_idx, device=device)
        masks.append(exp['edge_mask'])
    
    masks = torch.stack(masks)  # [n_runs, E]
    
    # Pairwise correlation
    correlations = []
    for i in range(n_runs):
        for j in range(i+1, n_runs):
            m1 = masks[i].numpy()
            m2 = masks[j].numpy()
            if np.std(m1) > 0 and np.std(m2) > 0:
                corr = np.corrcoef(m1, m2)[0,1]
                if not np.isnan(corr):
                    correlations.append(corr)
    
    return {
        'mean_correlation': np.mean(correlations) if correlations else 0.0,
        'std_correlation': np.std(correlations) if correlations else 0.0,
        'min_correlation': np.min(correlations) if correlations else 0.0
    }


def evaluate_explanations(
    model,
    explainer,
    data_list: List[Data],
    smiles_list: List[str],
    task_idx: int,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Full evaluation suite.
    """
    all_fid_plus = []
    all_fid_minus = []
    all_sparsity = []
    
    print(f"Evaluating {len(data_list)} explanations...")
    
    for i, data in enumerate(data_list):
        try:
            # Get explanation
            exp = explainer.explain(data, task_idx, device=device)
            
            # Metrics
            fid = compute_fidelity(model, data, exp['edge_mask'], task_idx, device=device)
            all_fid_plus.append(fid['fidelity_plus'])
            all_fid_minus.append(fid['fidelity_minus'])
            all_sparsity.append(compute_sparsity(exp['edge_mask']))
            
        except Exception as e:
            print(f"Failed on molecule {i}: {e}")
            continue
    
    return {
        'fidelity_plus_mean': np.mean(all_fid_plus) if all_fid_plus else 0.0,
        'fidelity_plus_std': np.std(all_fid_plus) if all_fid_plus else 0.0,
        'fidelity_minus_mean': np.mean(all_fid_minus) if all_fid_minus else 0.0,
        'fidelity_minus_std': np.std(all_fid_minus) if all_fid_minus else 0.0,
        'sparsity_mean': np.mean(all_sparsity) if all_sparsity else 0.0,
        'sparsity_std': np.std(all_sparsity) if all_sparsity else 0.0,
        'n_evaluated': len(all_fid_plus)
    }