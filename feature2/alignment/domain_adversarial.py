"""
Domain-Adversarial Alignment Module (DANN-style with Gradient Reversal).

Goal: Make encoder embeddings invariant to dataset identity.

Training:
    1. Encoder predicts molecular property (task loss)
    2. Domain discriminator predicts source dataset (domain loss)
    3. Gradient Reversal Layer flips gradient during backprop
    4. Encoder learns to "fool" the discriminator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional


# ─────────────────────────────────────────────
# Gradient Reversal Layer
# ─────────────────────────────────────────────

class GradientReversalFunction(Function):
    """
    Forward: identity
    Backward: multiply gradient by -lambda
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps GradientReversalFunction as nn.Module."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        """Update lambda dynamically (used for adaptive scheduling)."""
        self.lambda_ = lambda_


# ─────────────────────────────────────────────
# Domain Discriminator
# ─────────────────────────────────────────────

class DomainDiscriminator(nn.Module):
    """
    MLP that classifies which dataset an embedding came from.

    Input:  embedding [B, D]
    Output: domain logits [B, num_domains]
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_domains: int = 7,   # tox21, clintox, bbbp, bace, hiv, sider, muv
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_domains = num_domains

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_domains),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)


# ─────────────────────────────────────────────
# Domain-Adversarial Alignment
# ─────────────────────────────────────────────

class DomainAdversarialAlignment(nn.Module):
    """
    Full DANN-style alignment module.

    Architecture:
        Encoder → embedding → GRL → DomainDiscriminator → domain logits

    During backprop:
        Discriminator: tries to predict domain
        Encoder: tries to confuse discriminator (via GRL)
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_domains: int = 7,
        lambda_: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=lambda_)
        self.discriminator = DomainDiscriminator(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_domains=num_domains,
            dropout=dropout,
        )
        self.num_domains = num_domains

    def set_lambda(self, lambda_: float):
        """Update GRL lambda (for warm-up scheduling)."""
        self.grl.set_lambda(lambda_)

    def compute_alignment_loss(
        self,
        embeddings: torch.Tensor,        # [B, D]
        domain_labels: torch.Tensor,     # [B] long tensor of domain IDs
    ) -> torch.Tensor:
        """
        Compute domain-adversarial alignment loss.

        Returns:
            Cross-entropy loss between discriminator predictions and true domains.
            Due to GRL, this loss encourages encoder to produce domain-invariant embeddings.
        """
        if embeddings.shape[0] == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Pass through GRL
        reversed_emb = self.grl(embeddings)

        # Predict domain
        domain_logits = self.discriminator(reversed_emb)

        # Cross-entropy
        loss = F.cross_entropy(domain_logits, domain_labels)
        return loss


# ─────────────────────────────────────────────
# Lambda Scheduling
# ─────────────────────────────────────────────

def compute_grl_lambda(
    epoch: int,
    total_epochs: int,
    max_lambda: float = 1.0,
    warmup_fraction: float = 0.5,
) -> float:
    """
    DANN-style lambda scheduling: gradually increase from 0 to max_lambda.

    During first warmup_fraction epochs, lambda grows smoothly.
    After warmup, lambda stays at max_lambda.

    Formula: λ = max_λ × (2 / (1 + exp(-10 × p)) - 1)
    where p = min(epoch / (warmup_fraction × total_epochs), 1.0)
    """
    import math
    p = min(epoch / max(warmup_fraction * total_epochs, 1), 1.0)
    lambda_ = max_lambda * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0)
    return lambda_