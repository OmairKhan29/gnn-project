"""
Prototype-Based Alignment Module.

Learns shared task prototypes (learnable anchor vectors) and encourages
embeddings of each task's positive examples to cluster around their prototype.

This stabilizes multi-task representation learning by providing
explicit "centers" in embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


# ─────────────────────────────────────────────
# Learnable Prototypes
# ─────────────────────────────────────────────

class PrototypeBank(nn.Module):
    """
    A bank of learnable prototype vectors, one per task.

    Each prototype represents a "center" in embedding space.
    Embeddings of molecules positive for task t should cluster near prototype t.
    """

    def __init__(
        self,
        num_prototypes: int,    # number of tasks (17 in Feature 1)
        embedding_dim: int = 128,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.embedding_dim = embedding_dim

        # Learnable prototype vectors [T, D]
        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, embedding_dim) * init_scale
        )

    def get_prototypes(self) -> torch.Tensor:
        """Return L2-normalized prototypes [T, D]."""
        return F.normalize(self.prototypes, dim=-1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between embeddings and each prototype.

        Args:
            embeddings: [B, D]

        Returns:
            similarities: [B, T] (cosine similarity to each prototype)
        """
        emb_norm = F.normalize(embeddings, dim=-1)
        proto_norm = self.get_prototypes()
        return torch.matmul(emb_norm, proto_norm.T)


# ─────────────────────────────────────────────
# Prototype Alignment Loss
# ─────────────────────────────────────────────

class PrototypeAlignment(nn.Module):
    """
    Prototype-based alignment: encourages embeddings to be close to
    the prototype corresponding to their positive task labels.

    Loss strategies:
        - 'distance': L2 distance to prototype (positive labels only)
        - 'contrastive': InfoNCE between embedding and all prototypes
    """

    def __init__(
        self,
        num_prototypes: int,
        embedding_dim: int = 128,
        temperature: float = 0.1,
        strategy: str = "contrastive",  # 'distance' | 'contrastive'
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.strategy = strategy

        self.prototype_bank = PrototypeBank(
            num_prototypes=num_prototypes,
            embedding_dim=embedding_dim,
        )

    def compute_alignment_loss(
        self,
        embeddings: torch.Tensor,    # [B, D]
        labels: torch.Tensor,        # [B, T] — multi-task labels (-1 = missing)
    ) -> torch.Tensor:
        """
        Compute prototype alignment loss.

        For each (mol, task) pair where label == 1:
            Encourage embedding to be close to prototype[task]

        Returns:
            Scalar loss
        """
        if embeddings.shape[0] == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)

        # Find positive (mol_idx, task_idx) pairs
        # We want embeddings of mols positive for task t to be near prototype t
        positive_mask = (labels == 1)  # [B, T]

        if not positive_mask.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        if self.strategy == "distance":
            return self._distance_loss(embeddings, positive_mask)
        elif self.strategy == "contrastive":
            return self._contrastive_loss(embeddings, positive_mask, labels)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _distance_loss(
        self,
        embeddings: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        L2 distance loss: minimize ||emb - prototype||² for positive pairs.
        """
        emb_norm = F.normalize(embeddings, dim=-1)
        protos = self.prototype_bank.get_prototypes()  # [T, D]

        # All pairwise distances [B, T]
        distances = torch.cdist(emb_norm, protos, p=2)  # [B, T]

        # Only count positive pairs
        masked_dist = distances * positive_mask.float()
        n_positives = positive_mask.float().sum().clamp(min=1)
        loss = masked_dist.sum() / n_positives
        return loss

    def _contrastive_loss(
        self,
        embeddings: torch.Tensor,    # [B, D]
        positive_mask: torch.Tensor, # [B, T]
        labels: torch.Tensor,        # [B, T]
    ) -> torch.Tensor:
        """
        Contrastive loss: for each positive (mol, task) pair,
        the prototype[task] should have higher similarity than other prototypes.
        """
        # Similarity of each embedding to all prototypes [B, T]
        similarities = self.prototype_bank(embeddings) / self.temperature

        # Find positive (mol, task) pairs
        pos_indices = positive_mask.nonzero(as_tuple=False)  # [N_pos, 2]

        if pos_indices.shape[0] == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        losses = []
        for i in range(pos_indices.shape[0]):
            mol_idx = pos_indices[i, 0].item()
            task_idx = pos_indices[i, 1].item()

            mol_sims = similarities[mol_idx]   # [T]
            target = torch.tensor(task_idx, device=embeddings.device)
            loss_i = F.cross_entropy(mol_sims.unsqueeze(0), target.unsqueeze(0))
            losses.append(loss_i)

        return torch.stack(losses).mean()

    def get_prototype_similarities(self) -> torch.Tensor:
        """Return inter-prototype cosine similarity matrix [T, T]."""
        protos = self.prototype_bank.get_prototypes()
        return torch.matmul(protos, protos.T)