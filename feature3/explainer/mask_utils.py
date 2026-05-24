"""
Utility functions for processing GNNExplainer masks.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


def normalize_mask(
    mask: torch.Tensor,
    method: str = 'minmax',
) -> torch.Tensor:
    """
    Normalize mask values to [0,1].

    Args:
        mask: raw importance scores
        method: 'minmax', 'softmax', or 'rank'

    Returns:
        Normalized tensor in [0,1]
    """
    if method == 'minmax':
        lo, hi = mask.min(), mask.max()
        if (hi - lo).abs() < 1e-8:
            return torch.full_like(mask, 0.5)
        return (mask - lo) / (hi - lo)

    elif method == 'softmax':
        return torch.softmax(mask, dim=0)

    elif method == 'rank':
        n = mask.shape[0]
        if n == 1:
            return torch.tensor([0.5])
        order = mask.argsort()
        ranked = torch.zeros_like(mask)
        for rank, idx in enumerate(order):
            ranked[idx] = rank / (n - 1)
        return ranked

    else:
        raise ValueError(f"Unknown method: {method}. Use 'minmax', 'softmax', 'rank'")


def threshold_mask(
    mask: torch.Tensor,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Convert soft mask to binary by threshold or top-K.

    Returns:
        Binary tensor (1.0 = important, 0.0 = not important)
    """
    if top_k is not None:
        k = min(top_k, mask.shape[0])
        binary = torch.zeros_like(mask)
        topk_idx = mask.topk(k).indices
        binary[topk_idx] = 1.0
        return binary
    else:
        return (mask >= threshold).float()


def get_important_edges(
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
    """
    Extract important atom pairs from edge mask.

    Returns:
        pairs: List of (atom_i, atom_j) tuples
        scores: Corresponding importance scores
    """
    binary = threshold_mask(edge_mask, threshold, top_k)
    src, dst = edge_index[0], edge_index[1]

    pairs = []
    scores = []
    seen = set()

    for k in range(binary.shape[0]):
        if binary[k] > 0:
            i, j = src[k].item(), dst[k].item()
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                pairs.append(key)
                scores.append(edge_mask[k].item())

    return pairs, torch.tensor(scores) if scores else torch.tensor([])


def mask_statistics(mask: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics of an importance mask.

    Returns dict with mean, std, min, max, sparsity, entropy.
    """
    eps = 1e-8
    m = mask.float()
    sparsity = (m < 0.5).float().mean().item()
    entropy = -(
        m * torch.log(m + eps) + (1 - m) * torch.log(1 - m + eps)
    ).mean().item()

    return {
        'mean': m.mean().item(),
        'std': m.std().item(),
        'min': m.min().item(),
        'max': m.max().item(),
        'sparsity': sparsity,
        'entropy': entropy,
        'num_elements': m.shape[0],
    }