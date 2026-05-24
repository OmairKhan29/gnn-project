"""
Alignment Trainer for Feature 2 Phase 2.

Trains the multi-task encoder with an additional alignment loss:

    L_total = L_task + λ_align × L_alignment

Supports:
    - 'contrastive': InfoNCE on scaffold-positive pairs
    - 'domain': Domain-adversarial via GRL
    - 'prototype': Learnable prototype alignment
    - 'none': Baseline (no alignment)
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from feature2.alignment.contrastive import (
    ContrastiveAlignment, build_scaffold_groups, sample_positive_pair_indices
)
from feature2.alignment.domain_adversarial import (
    DomainAdversarialAlignment, compute_grl_lambda
)
from feature2.alignment.prototype import PrototypeAlignment
from evaluation.metrics import compute_roc_auc


# ─────────────────────────────────────────────
# Alignment Module Factory
# ─────────────────────────────────────────────

def create_alignment_module(
    strategy: str,
    embedding_dim: int = 128,
    num_tasks: int = 17,
    num_domains: int = 7,
    temperature: float = 0.1,
    grl_lambda: float = 1.0,
) -> Optional[nn.Module]:
    """Factory for alignment modules."""
    if strategy == "none" or strategy is None:
        return None
    elif strategy == "contrastive":
        return ContrastiveAlignment(
            embedding_dim=embedding_dim,
            projection_dim=embedding_dim // 2,
            temperature=temperature,
        )
    elif strategy == "domain":
        return DomainAdversarialAlignment(
            embedding_dim=embedding_dim,
            num_domains=num_domains,
            lambda_=grl_lambda,
        )
    elif strategy == "prototype":
        return PrototypeAlignment(
            num_prototypes=num_tasks,
            embedding_dim=embedding_dim,
            temperature=temperature,
            strategy="contrastive",
        )
    else:
        raise ValueError(f"Unknown alignment strategy: {strategy}")


# ─────────────────────────────────────────────
# Alignment Trainer
# ─────────────────────────────────────────────

class AlignmentTrainer:
    """
    Trains a Feature 1 multi-task model with added alignment loss.

    Total loss: L_total = L_task + λ_align × L_alignment

    Supports per-strategy alignment computation and dynamic λ scheduling.
    """

    def __init__(
        self,
        task_model: nn.Module,            # Feature 1 multi-task classifier
        alignment_module: Optional[nn.Module],
        alignment_strategy: str,
        train_dataset,
        val_dataset,
        test_dataset,
        task_names: List[str],
        smiles_list: List[str],           # for scaffold groups (contrastive)
        config: Dict,
        checkpoint_dir: str = "checkpoints/feature2/alignment",
        result_dir: str = "results/feature2/alignment",
        device: str = "cpu",
        verbose: bool = True,
    ):
        self.task_model = task_model.to(device)
        self.alignment_module = (
            alignment_module.to(device) if alignment_module else None
        )
        self.alignment_strategy = alignment_strategy

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.device = device
        self.verbose = verbose

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        # Training config
        self.lr = config.get("lr", 1e-3)
        self.weight_decay = config.get("weight_decay", 1e-5)
        self.batch_size = config.get("batch_size", 32)
        self.epochs = config.get("epochs", 100)
        self.patience = config.get("patience", 20)
        self.grad_clip = config.get("grad_clip", 1.0)
        self.alignment_weight = config.get("alignment_weight", 0.5)  # λ_align
        self.grl_max_lambda = config.get("grl_max_lambda", 1.0)
        self.warmup_fraction = config.get("warmup_fraction", 0.3)

        # Scaffold groups (precompute for contrastive)
        if alignment_strategy == "contrastive" and smiles_list:
            if verbose:
                print(f"[AlignmentTrainer] Building scaffold groups for "
                      f"{len(smiles_list)} molecules...")
            self.scaffold_groups = build_scaffold_groups(smiles_list)
            multi_groups = {k: v for k, v in self.scaffold_groups.items()
                            if len(v) > 1}
            if verbose:
                print(f"[AlignmentTrainer] Found {len(multi_groups)} "
                      f"scaffolds with ≥2 molecules")
        else:
            self.scaffold_groups = None

        # Build optimizer
        params = list(self.task_model.parameters())
        if self.alignment_module is not None:
            params += list(self.alignment_module.parameters())

        self.optimizer = torch.optim.Adam(
            params, lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=10
        )

        # History
        self.history = {
            "train_task_loss": [],
            "train_align_loss": [],
            "val_auc": [],
            "lr": [],
            "alignment_weight": [],
        }

        self.best_val_auc = -float("inf")
        self.best_state = None
        self.patience_counter = 0

    def _build_loader(self, dataset, shuffle: bool = False):
        return PyGDataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=shuffle, num_workers=0,
        )

    def _extract_embeddings(self, batch) -> torch.Tensor:
        """
        Extract graph-level embeddings from the encoder.
        Works with both task-conditioned and hard-sharing classifiers.
        """
        encoder = self.task_model.encoder
        # Task-conditioned encoder needs task_id; use default task 0
        if hasattr(encoder, "task_embedding"):
            task_ids = torch.zeros(batch.num_graphs, dtype=torch.long,
                                   device=batch.x.device)
            embeddings = encoder(batch, task_ids)
        else:
            embeddings = encoder(batch)
        return embeddings

    def _compute_alignment_loss(
        self,
        batch,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute alignment loss based on the strategy."""
        if self.alignment_module is None:
            return torch.tensor(0.0, device=self.device)

        if self.alignment_strategy == "contrastive":
            # Sample positive pairs from batch using scaffolds
            if not hasattr(batch, "smiles"):
                return torch.tensor(0.0, device=self.device)

            # Build local batch-index scaffold mapping
            batch_smiles = batch.smiles
            if isinstance(batch_smiles, str):
                batch_smiles = [batch_smiles]
            local_groups = build_scaffold_groups(batch_smiles)
            multi_groups = {k: v for k, v in local_groups.items() if len(v) > 1}

            if not multi_groups:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            pairs = sample_positive_pair_indices(
                multi_groups,
                list(range(len(batch_smiles))),
                n_pairs=min(16, len(multi_groups) * 2),
            )

            return self.alignment_module.compute_alignment_loss(
                embeddings, pairs
            )

        elif self.alignment_strategy == "domain":
            # Need domain labels
            if hasattr(batch, "dataset_id"):
                domain_labels = batch.dataset_id.long()
                # Ensure within range
                domain_labels = domain_labels.clamp(
                    0, self.alignment_module.num_domains - 1
                )
            else:
                # Default to domain 0
                domain_labels = torch.zeros(
                    embeddings.shape[0], dtype=torch.long, device=self.device
                )
            return self.alignment_module.compute_alignment_loss(
                embeddings, domain_labels
            )

        elif self.alignment_strategy == "prototype":
            # Need task labels
            labels = batch.y
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            return self.alignment_module.compute_alignment_loss(
                embeddings, labels
            )

        return torch.tensor(0.0, device=self.device)

    def _train_epoch(self, loader, epoch: int) -> Tuple[float, float]:
        self.task_model.train()
        if self.alignment_module is not None:
            self.alignment_module.train()

        # Update GRL lambda if domain strategy
        if self.alignment_strategy == "domain":
            new_lambda = compute_grl_lambda(
                epoch, self.epochs,
                self.grl_max_lambda, self.warmup_fraction
            )
            self.alignment_module.set_lambda(new_lambda)

        total_task_loss = 0.0
        total_align_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Task loss
            task_loss = self.task_model.compute_loss(batch)

            # Alignment loss
            if self.alignment_module is not None:
                embeddings = self._extract_embeddings(batch)
                align_loss = self._compute_alignment_loss(batch, embeddings)
                total_loss = task_loss + self.alignment_weight * align_loss
            else:
                align_loss = torch.tensor(0.0, device=self.device)
                total_loss = task_loss

            total_loss.backward()

            if self.grad_clip > 0:
                params = list(self.task_model.parameters())
                if self.alignment_module is not None:
                    params += list(self.alignment_module.parameters())
                nn.utils.clip_grad_norm_(params, self.grad_clip)

            self.optimizer.step()

            total_task_loss += task_loss.item()
            total_align_loss += align_loss.item()
            n_batches += 1

        return (
            total_task_loss / max(n_batches, 1),
            total_align_loss / max(n_batches, 1),
        )

    @torch.no_grad()
    def _evaluate(self, loader) -> Tuple[float, Dict[str, float]]:
        self.task_model.eval()

        all_probs = []
        all_labels = []

        for batch in loader:
            batch = batch.to(self.device)
            probs = self.task_model.predict(batch)
            labels = batch.y
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            if probs.shape != labels.shape:
                labels = labels.view(probs.shape)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        per_task_auc = {}
        valid_aucs = []

        for t, name in enumerate(self.task_names):
            tl = all_labels[:, t]
            tp = all_probs[:, t]
            mask = tl != -1
            if mask.sum() < 10:
                continue
            tlm = tl[mask]
            tpm = tp[mask]
            if len(np.unique(tlm)) < 2:
                continue
            auc = compute_roc_auc(tlm, tpm)
            if not np.isnan(auc):
                per_task_auc[name] = float(auc)
                valid_aucs.append(auc)

        mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.5
        return mean_auc, per_task_auc

    def train(self, experiment_name: str = "alignment_exp") -> Dict:
        train_loader = self._build_loader(self.train_dataset, shuffle=True)
        val_loader = self._build_loader(self.val_dataset, shuffle=False)
        test_loader = self._build_loader(self.test_dataset, shuffle=False)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Alignment Training: {experiment_name}")
            print(f"  Strategy: {self.alignment_strategy}, "
                  f"λ_align: {self.alignment_weight}")
            print(f"  Train: {len(self.train_dataset)}, "
                  f"Val: {len(self.val_dataset)}, "
                  f"Test: {len(self.test_dataset)}")
            print(f"{'='*60}")

        start_time = time.time()

        for epoch in range(self.epochs):
            task_loss, align_loss = self._train_epoch(train_loader, epoch)
            val_auc, val_per_task = self._evaluate(val_loader)
            self.scheduler.step(val_auc)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_task_loss"].append(task_loss)
            self.history["train_align_loss"].append(align_loss)
            self.history["val_auc"].append(val_auc)
            self.history["lr"].append(current_lr)
            self.history["alignment_weight"].append(self.alignment_weight)

            if self.verbose and epoch % 5 == 0:
                print(f"  Epoch {epoch:3d} | "
                      f"Task L: {task_loss:.4f} | "
                      f"Align L: {align_loss:.4f} | "
                      f"Val AUC: {val_auc:.4f} | "
                      f"LR: {current_lr:.2e}")

            # Early stopping
            if val_auc > self.best_val_auc + 1e-4:
                self.best_val_auc = val_auc
                self.patience_counter = 0
                self.best_state = {
                    k: v.cpu().clone()
                    for k, v in self.task_model.state_dict().items()
                }
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    if self.verbose:
                        print(f"  Early stopping at epoch {epoch}")
                    break

        # Restore best
        if self.best_state is not None:
            self.task_model.load_state_dict(self.best_state)

        # Final test
        test_auc, test_per_task = self._evaluate(test_loader)
        val_auc, val_per_task = self._evaluate(val_loader)

        elapsed = time.time() - start_time

        results = {
            "experiment_name": experiment_name,
            "alignment_strategy": self.alignment_strategy,
            "alignment_weight": self.alignment_weight,
            "val_auc": val_auc,
            "test_auc": test_auc,
            "best_val_auc": self.best_val_auc,
            "val_per_task_auc": val_per_task,
            "test_per_task_auc": test_per_task,
            "elapsed_seconds": elapsed,
            "n_train": len(self.train_dataset),
            "n_val": len(self.val_dataset),
            "n_test": len(self.test_dataset),
            "config": self.config,
        }

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Final: {experiment_name}")
            print(f"  Val AUC:  {val_auc:.4f}")
            print(f"  Test AUC: {test_auc:.4f}")
            print(f"  Time: {elapsed:.1f}s")
            print(f"{'='*60}\n")

        # Save
        result_path = os.path.join(self.result_dir, f"{experiment_name}_results.json")
        save_data = {k: v for k, v in results.items() if k != "history"}
        save_data["history"] = {
            k: [float(x) for x in v] for k, v in self.history.items()
        }
        with open(result_path, "w") as f:
            json.dump(save_data, f, indent=2)

        if self.verbose:
            print(f"  Results saved to {result_path}")

        # Save checkpoint
        ckpt_path = os.path.join(self.checkpoint_dir, f"{experiment_name}_best.pt")
        torch.save({
            "model_state_dict": self.task_model.state_dict(),
            "config": self.config,
            "alignment_strategy": self.alignment_strategy,
            "test_auc": test_auc,
        }, ckpt_path)

        return results