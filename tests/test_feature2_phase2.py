"""
Phase 2 Tests: Alignment modules and trainer.
"""

import pytest
import torch
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────
# Contrastive Tests
# ─────────────────────────────────────────────

class TestContrastiveAlignment:

    def test_infonce_basic(self):
        from feature2.alignment.contrastive import InfoNCELoss
        loss_fn = InfoNCELoss(temperature=0.1)
        a = torch.randn(8, 32)
        p = torch.randn(8, 32)
        loss = loss_fn(a, p)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_infonce_perfect_match(self):
        from feature2.alignment.contrastive import InfoNCELoss
        loss_fn = InfoNCELoss(temperature=0.1)
        # Identical anchor and positive — low loss expected
        emb = torch.randn(8, 32)
        loss = loss_fn(emb, emb)
        # Loss not zero (still cross-entropy over batch)
        assert loss.item() >= 0

    def test_scaffold_grouping(self):
        from feature2.alignment.contrastive import build_scaffold_groups
        smiles = ["CCO", "CCCCO", "c1ccccc1", "c1ccccc1C"]
        groups = build_scaffold_groups(smiles)
        assert isinstance(groups, dict)

    def test_contrastive_alignment_forward(self):
        from feature2.alignment.contrastive import ContrastiveAlignment
        model = ContrastiveAlignment(embedding_dim=32, projection_dim=16)
        emb = torch.randn(8, 32)
        pairs = [(0, 1), (2, 3)]
        loss = model.compute_alignment_loss(emb, pairs)
        assert loss.item() >= 0

    def test_contrastive_no_pairs(self):
        from feature2.alignment.contrastive import ContrastiveAlignment
        model = ContrastiveAlignment(embedding_dim=32)
        emb = torch.randn(8, 32)
        loss = model.compute_alignment_loss(emb, [])
        assert loss.item() == 0.0

    def test_projection_dim(self):
        from feature2.alignment.contrastive import ContrastiveAlignment
        model = ContrastiveAlignment(embedding_dim=64, projection_dim=16)
        emb = torch.randn(4, 64)
        proj = model.project(emb)
        assert proj.shape == (4, 16)


# ─────────────────────────────────────────────
# Domain-Adversarial Tests
# ─────────────────────────────────────────────

class TestDomainAdversarial:

    def test_gradient_reversal_forward(self):
        from feature2.alignment.domain_adversarial import GradientReversalLayer
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(4, 16, requires_grad=True)
        y = grl(x)
        assert torch.allclose(x, y)

    def test_gradient_reversal_backward(self):
        from feature2.alignment.domain_adversarial import GradientReversalLayer
        grl = GradientReversalLayer(lambda_=0.5)
        x = torch.randn(4, 16, requires_grad=True)
        y = grl(x)
        loss = y.sum()
        loss.backward()
        # Gradient should be -0.5 (sign flipped, scaled by 0.5)
        assert (x.grad < 0).all()

    def test_domain_discriminator_output_shape(self):
        from feature2.alignment.domain_adversarial import DomainDiscriminator
        disc = DomainDiscriminator(embedding_dim=64, num_domains=5)
        emb = torch.randn(8, 64)
        logits = disc(emb)
        assert logits.shape == (8, 5)

    def test_domain_alignment_loss(self):
        from feature2.alignment.domain_adversarial import DomainAdversarialAlignment
        model = DomainAdversarialAlignment(
            embedding_dim=64, num_domains=5, lambda_=1.0
        )
        emb = torch.randn(8, 64)
        labels = torch.randint(0, 5, (8,))
        loss = model.compute_alignment_loss(emb, labels)
        assert loss.item() > 0

    def test_grl_lambda_scheduling(self):
        from feature2.alignment.domain_adversarial import compute_grl_lambda
        l0 = compute_grl_lambda(0, 100, max_lambda=1.0, warmup_fraction=0.5)
        l50 = compute_grl_lambda(50, 100, max_lambda=1.0, warmup_fraction=0.5)
        l100 = compute_grl_lambda(100, 100, max_lambda=1.0, warmup_fraction=0.5)
        assert l0 < l50 <= l100
        assert 0 <= l0 <= 1
        assert 0 <= l100 <= 1


# ─────────────────────────────────────────────
# Prototype Tests
# ─────────────────────────────────────────────

class TestPrototypeAlignment:

    def test_prototype_bank_init(self):
        from feature2.alignment.prototype import PrototypeBank
        bank = PrototypeBank(num_prototypes=10, embedding_dim=64)
        assert bank.prototypes.shape == (10, 64)

    def test_prototype_bank_forward(self):
        from feature2.alignment.prototype import PrototypeBank
        bank = PrototypeBank(num_prototypes=5, embedding_dim=32)
        emb = torch.randn(8, 32)
        sims = bank(emb)
        assert sims.shape == (8, 5)
        # Cosine similarities ∈ [-1, 1]
        assert (sims >= -1.01).all() and (sims <= 1.01).all()

    def test_prototype_distance_loss(self):
        from feature2.alignment.prototype import PrototypeAlignment
        model = PrototypeAlignment(
            num_prototypes=5, embedding_dim=32, strategy="distance"
        )
        emb = torch.randn(8, 32)
        labels = torch.zeros(8, 5)
        labels[:, 0] = 1  # All molecules positive for task 0
        loss = model.compute_alignment_loss(emb, labels)
        assert loss.item() >= 0

    def test_prototype_contrastive_loss(self):
        from feature2.alignment.prototype import PrototypeAlignment
        model = PrototypeAlignment(
            num_prototypes=5, embedding_dim=32, strategy="contrastive"
        )
        emb = torch.randn(8, 32)
        labels = torch.zeros(8, 5)
        labels[:, 0] = 1
        labels[:, 1] = 1  # Some positive for task 1
        loss = model.compute_alignment_loss(emb, labels)
        assert loss.item() >= 0

    def test_prototype_no_positives(self):
        from feature2.alignment.prototype import PrototypeAlignment
        model = PrototypeAlignment(num_prototypes=5, embedding_dim=32)
        emb = torch.randn(8, 32)
        labels = torch.zeros(8, 5)  # No positives
        loss = model.compute_alignment_loss(emb, labels)
        assert loss.item() == 0.0


# ─────────────────────────────────────────────
# Alignment Factory
# ─────────────────────────────────────────────

class TestAlignmentFactory:

    def test_factory_none(self):
        from feature2.alignment.alignment_trainer import create_alignment_module
        m = create_alignment_module("none")
        assert m is None

    def test_factory_contrastive(self):
        from feature2.alignment.alignment_trainer import create_alignment_module
        m = create_alignment_module("contrastive", embedding_dim=64)
        assert m is not None
        assert hasattr(m, "compute_alignment_loss")

    def test_factory_domain(self):
        from feature2.alignment.alignment_trainer import create_alignment_module
        m = create_alignment_module("domain", embedding_dim=64, num_domains=5)
        assert m is not None

    def test_factory_prototype(self):
        from feature2.alignment.alignment_trainer import create_alignment_module
        m = create_alignment_module("prototype", embedding_dim=64, num_tasks=10)
        assert m is not None

    def test_factory_invalid(self):
        from feature2.alignment.alignment_trainer import create_alignment_module
        with pytest.raises(ValueError):
            create_alignment_module("invalid_strategy")