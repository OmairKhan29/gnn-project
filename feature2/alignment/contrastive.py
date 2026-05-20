"""
Contrastive Alignment Module (InfoNCE).

Encourages embeddings of structurally-similar molecules to be close
in representation space, and dissimilar molecules to be far apart.

Positive pair generation strategies:
    - 'scaffold': Same Bemis-Murcko scaffold
    - 'augmentation': Different 3D conformer of same molecule
    - 'task_label': Same label for shared task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


# ─────────────────────────────────────────────
# Positive Pair Generation
# ─────────────────────────────────────────────

def get_scaffold(smiles: str) -> str:
    """Compute Bemis-Murcko scaffold for a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold
    except Exception:
        return ""


def build_scaffold_groups(smiles_list: List[str]) -> Dict[str, List[int]]:
    """
    Group molecule indices by scaffold.

    Returns:
        {scaffold_smiles: [mol_indices]}
    """
    groups = {}
    for i, smi in enumerate(smiles_list):
        scaffold = get_scaffold(smi)
        if not scaffold:
            continue
        if scaffold not in groups:
            groups[scaffold] = []
        groups[scaffold].append(i)
    return groups


def sample_positive_pair_indices(
    scaffold_groups: Dict[str, List[int]],
    batch_indices: List[int],
    n_pairs: int = 16,
) -> List[Tuple[int, int]]:
    """
    Sample positive pairs from a batch using scaffold matches.

    Returns:
        List of (idx_a, idx_b) pairs where both share a scaffold.
    """
    # Invert: mol_idx → scaffold
    batch_set = set(batch_indices)
    idx_to_scaffold = {}
    for scaffold, mols in scaffold_groups.items():
        for m in mols:
            if m in batch_set:
                idx_to_scaffold[m] = scaffold

    # Find pairs within batch with matching scaffolds
    pairs = []
    seen = set()
    for scaffold, mols in scaffold_groups.items():
        in_batch = [m for m in mols if m in batch_set]
        if len(in_batch) < 2:
            continue
        for i in range(len(in_batch)):
            for j in range(i + 1, len(in_batch)):
                a, b = in_batch[i], in_batch[j]
                key = (min(a, b), max(a, b))
                if key not in seen:
                    pairs.append((a, b))
                    seen.add(key)

    # Random subsample
    if len(pairs) > n_pairs:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(pairs), size=n_pairs, replace=False)
        pairs = [pairs[i] for i in idx]

    return pairs


# ─────────────────────────────────────────────
# InfoNCE Loss
# ─────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss.

    L = -log( exp(sim(a, a+) / τ) / Σ exp(sim(a, all) / τ) )

    Where:
        a = anchor embedding
        a+ = positive (similar) embedding
        all = all batch embeddings (positives + negatives)
        τ = temperature (default 0.1)
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor_emb: torch.Tensor,    # [N, D]
        positive_emb: torch.Tensor,  # [N, D]
        negative_emb: Optional[torch.Tensor] = None,  # [M, D] or None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            anchor_emb: Anchor embeddings [N, D]
            positive_emb: Corresponding positive embeddings [N, D]
            negative_emb: Additional negatives (if None, use in-batch negatives)

        Returns:
            Scalar loss
        """
        # Normalize embeddings
        anchor = F.normalize(anchor_emb, dim=-1)
        positive = F.normalize(positive_emb, dim=-1)

        N = anchor.shape[0]
        if N == 0:
            return torch.tensor(0.0, device=anchor_emb.device, requires_grad=True)

        # All candidates: positives + negatives
        if negative_emb is not None and len(negative_emb) > 0:
            negative = F.normalize(negative_emb, dim=-1)
            candidates = torch.cat([positive, negative], dim=0)  # [N+M, D]
        else:
            candidates = positive  # In-batch only

        # Similarity matrix: anchor × candidates
        sim = torch.matmul(anchor, candidates.T) / self.temperature  # [N, N+M]

        # Positive indices: diagonal (anchor i ↔ positive i)
        labels = torch.arange(N, device=anchor.device)

        loss = F.cross_entropy(sim, labels)
        return loss


# ─────────────────────────────────────────────
# Contrastive Alignment Wrapper
# ─────────────────────────────────────────────

class ContrastiveAlignment(nn.Module):
    """
    Wraps an encoder with contrastive alignment training.

    Usage:
        contrast = ContrastiveAlignment(encoder, temperature=0.1)
        loss = contrast.compute_alignment_loss(batch, scaffold_groups)
    """

    def __init__(
        self,
        projection_dim: int = 64,
        embedding_dim: int = 128,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.projection_dim = projection_dim
        self.embedding_dim = embedding_dim

        # Projection head (MLP) — projects embeddings to contrastive space
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim),
        )

        self.infonce = InfoNCELoss(temperature=temperature)

    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project encoder embeddings to contrastive space."""
        return self.projection(embeddings)

    def compute_alignment_loss(
        self,
        embeddings: torch.Tensor,        # [B, D] from encoder
        positive_pairs: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Compute contrastive loss given embeddings and positive pair indices.

        Args:
            embeddings: [B, D] graph-level embeddings
            positive_pairs: List of (idx_a, idx_b) — indices within the batch

        Returns:
            Scalar contrastive loss
        """
        if not positive_pairs:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Extract anchor and positive embeddings
        anchor_idx = torch.tensor([p[0] for p in positive_pairs],
                                  device=embeddings.device)
        positive_idx = torch.tensor([p[1] for p in positive_pairs],
                                    device=embeddings.device)

        anchor_emb = embeddings[anchor_idx]
        positive_emb = embeddings[positive_idx]

        # Project to contrastive space
        anchor_proj = self.project(anchor_emb)
        positive_proj = self.project(positive_emb)

        # Use rest of batch as negatives
        all_idx = set(range(embeddings.shape[0]))
        used_idx = set(p[0] for p in positive_pairs) | set(p[1] for p in positive_pairs)
        negative_idx = list(all_idx - used_idx)

        negative_emb = None
        if negative_idx:
            negative_idx_t = torch.tensor(negative_idx, device=embeddings.device)
            negative_emb = self.project(embeddings[negative_idx_t])

        return self.infonce(anchor_proj, positive_proj, negative_emb)