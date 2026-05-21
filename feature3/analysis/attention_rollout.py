"""
Attention rollout for tracking information flow through EGNN layers.

Computes which atoms receive most information after K message
passing steps — complements GNNExplainer with gradient-free analysis.
"""

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from typing import List, Optional
import numpy as np


class MessagePassingTracer:
    """
    Traces information flow through message passing layers.

    For each layer, computes the attention/influence matrix
    A[i,j] = how much node j influences node i after k hops.

    Works with EGNN by using distance-weighted adjacency.
    """

    def __init__(self, num_layers: int = 4):
        self.num_layers = num_layers

    def compute_influence_matrix(
        self,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        pos: Optional[torch.Tensor],
        num_nodes: int,
        distance_decay: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute influence matrix A where A[i,j] = influence of j on i.

        For EGNN: weight edges by inverse distance (closer = more influence).

        Returns:
            [N, N] influence matrix after num_layers hops
        """
        # Base adjacency (weighted by inverse distance if pos available)
        if pos is not None:
            src, dst = edge_index
            diff = pos[src] - pos[dst]
            dist = diff.norm(dim=1)
            weights = torch.exp(-distance_decay * dist)
        else:
            weights = torch.ones(edge_index.shape[1])

        # Build weighted adjacency matrix
        adj = to_dense_adj(
            edge_index,
            edge_attr=weights,
            max_num_nodes=num_nodes
        ).squeeze(0)  # [N, N]

        # Row-normalize (each node distributes influence equally)
        row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
        adj_norm = adj / row_sum

        # Add self-loops (node retains some of its own information)
        eye = torch.eye(num_nodes, device=adj.device)
        adj_with_self = 0.5 * adj_norm + 0.5 * eye

        # Rollout: multiply matrices across layers
        influence = adj_with_self.clone()
        for _ in range(self.num_layers - 1):
            influence = influence @ adj_with_self

        return influence  # [N, N]

    def get_node_importance(
        self,
        influence_matrix: torch.Tensor,
        target_nodes: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Compute per-node importance from influence matrix.

        If target_nodes given: sum influence received by those nodes.
        Otherwise: sum all received influence.

        Returns: [N] importance scores
        """
        if target_nodes is not None:
            # Importance = how much each node contributes to targets
            target_influence = influence_matrix[target_nodes, :]  # [T, N]
            return target_influence.sum(dim=0)
        else:
            # Total influence received
            return influence_matrix.sum(dim=0)

    def trace_molecule(
        self,
        data,
        target_nodes: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Full trace for a single molecule Data object.

        Returns [N] node importance from message passing rollout.
        """
        num_nodes = data.num_nodes
        pos = data.pos if hasattr(data, 'pos') else None

        influence = self.compute_influence_matrix(
            data.edge_index,
            data.edge_attr,
            pos,
            num_nodes,
        )

        return self.get_node_importance(influence, target_nodes)