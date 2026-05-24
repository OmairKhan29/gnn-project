"""
Transfer Learning Classification Heads for Feature 2.
Implements Linear Probe and Fine-Tuning heads on top of frozen/unfrozen encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from feature2.models.pretrained_encoder import FrozenEncoder


# ─────────────────────────────────────────────
# Base Transfer Model
# ─────────────────────────────────────────────

class BaseTransferModel(nn.Module):
    """Base class for all transfer learning models."""

    def __init__(self, encoder: FrozenEncoder, num_tasks: int):
        super().__init__()
        self.encoder = encoder
        self.num_tasks = num_tasks

    def compute_loss(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, batch) -> torch.Tensor:
        raise NotImplementedError


# ─────────────────────────────────────────────
# Linear Probe
# ─────────────────────────────────────────────

class LinearProbeClassifier(BaseTransferModel):
    """
    Linear probe on frozen encoder.

    Architecture:
        FrozenEncoder → Linear(hidden_dim, num_tasks)

    Training:
        Only the linear head is updated.
        Encoder weights are frozen.

    Use case:
        Measures quality of pretrained representations
        without any adaptation.
    """

    def __init__(
        self,
        encoder: FrozenEncoder,
        num_tasks: int,
        dropout: float = 0.1,
    ):
        super().__init__(encoder, num_tasks)

        embedding_dim = encoder.get_output_dim()

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_tasks),
        )

        # Count trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[LinearProbe] {trainable:,} trainable parameters "
              f"(embedding_dim={embedding_dim}, num_tasks={num_tasks})")

    def forward(self, batch, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Returns:
            logits: [B, num_tasks]
        """
        with torch.no_grad():  # Encoder always frozen
            embeddings = self.encoder(batch, task_id=task_id)

        logits = self.head(embeddings)
        return logits

    def compute_loss(self, batch) -> torch.Tensor:
        """Compute multi-task BCE loss with missing label masking."""
        logits = self.forward(batch)                 # [B, num_tasks]
        labels = batch.y                             # [B, num_tasks]

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)

        # Reshape if needed
        if logits.shape != labels.shape:
            labels = labels.view(logits.shape)

        # Missing label mask (-1 = missing)
        mask = (labels != -1).float()

        loss = F.binary_cross_entropy_with_logits(
            logits, labels.clamp(0, 1), reduction="none"
        )
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss

    def predict(self, batch) -> torch.Tensor:
        """
        Returns:
            probs: [B, num_tasks] sigmoid probabilities
        """
        logits = self.forward(batch)
        return torch.sigmoid(logits)


# ─────────────────────────────────────────────
# Fine-Tuning Classifier
# ─────────────────────────────────────────────

class FineTuneClassifier(BaseTransferModel):
    """
    Fine-tuning model with partially or fully unfrozen encoder.

    Architecture:
        [Partially Unfrozen Encoder] → MLP(hidden_dim, hidden_dim//2, num_tasks)

    Strategies:
        "top_layers": Only unfreeze top N encoder layers
        "full": Unfreeze entire encoder
        "head_only": Equivalent to linear probe (all frozen)
    """

    def __init__(
        self,
        encoder: FrozenEncoder,
        num_tasks: int,
        strategy: str = "top_layers",    # "top_layers" | "full" | "head_only"
        num_unfreeze_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__(encoder, num_tasks)
        self.strategy = strategy

        # Configure encoder freezing
        if strategy == "top_layers":
            encoder.unfreeze_top_layers(num_unfreeze_layers)
        elif strategy == "full":
            encoder.unfreeze_encoder()
        elif strategy == "head_only":
            encoder.freeze_encoder()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        embedding_dim = encoder.get_output_dim()

        # MLP classification head (slightly more expressive than linear probe)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[FineTune-{strategy}] {trainable:,} trainable parameters")

    def forward(self, batch, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Returns:
            logits: [B, num_tasks]
        """
        embeddings = self.encoder(batch, task_id=task_id)
        logits = self.head(embeddings)
        return logits

    def compute_loss(self, batch) -> torch.Tensor:
        """Compute multi-task BCE loss with missing label masking."""
        logits = self.forward(batch)
        labels = batch.y

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        if logits.shape != labels.shape:
            labels = labels.view(logits.shape)

        mask = (labels != -1).float()
        loss = F.binary_cross_entropy_with_logits(
            logits, labels.clamp(0, 1), reduction="none"
        )
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss

    def predict(self, batch) -> torch.Tensor:
        logits = self.forward(batch)
        return torch.sigmoid(logits)


# ─────────────────────────────────────────────
# Scratch Classifier (No Pretraining)
# ─────────────────────────────────────────────

class ScratchClassifier(nn.Module):
    """
    Trains from scratch on transfer dataset.
    Used as the baseline to compare against transfer strategies.
    """

    def __init__(
        self,
        node_dim: int = 129,
        edge_dim: int = 6,
        hidden_dim: int = 128,
        n_layers: int = 4,
        num_tasks: int = 27,
        dropout: float = 0.1,
    ):
        super().__init__()
        from models.egnn import EGNN

        self.encoder = EGNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[ScratchClassifier] {n_params:,} total parameters")

    def forward(self, batch) -> torch.Tensor:
        embeddings = self.encoder(batch)
        return self.head(embeddings)

    def compute_loss(self, batch) -> torch.Tensor:
        logits = self.forward(batch)
        labels = batch.y

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        if logits.shape != labels.shape:
            labels = labels.view(logits.shape)

        mask = (labels != -1).float()
        loss = F.binary_cross_entropy_with_logits(
            logits, labels.clamp(0, 1), reduction="none"
        )
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss

    def predict(self, batch) -> torch.Tensor:
        return torch.sigmoid(self.forward(batch))