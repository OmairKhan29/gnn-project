"""
Low-Data Trainer for Feature 2 Phase 3.

Trains a model on a fraction of the transfer dataset.
Compares three strategies under limited data:
    1. scratch       — EGNN trained from scratch
    2. linear_probe  — Frozen aligned encoder + linear head
    3. finetune      — Partially unfrozen aligned encoder + MLP head
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from feature2.data.low_data_splits import create_low_data_subset
from feature2.training.transfer_trainer import TransferEarlyStopping
from evaluation.metrics import compute_roc_auc


# ─────────────────────────────────────────────
# Low-Data Training Runner
# ─────────────────────────────────────────────

class LowDataTrainer:
    """
    Unified trainer for low-data experiments.

    Given:
        - Full training dataset
        - Data fraction f ∈ {0.10, 0.25, 0.50, 1.00}
        - Strategy: scratch / linear_probe / finetune

    Runs:
        1. Subsample training set to fraction f
        2. Train model for specified epochs
        3. Evaluate on FULL validation and test sets
        4. Return per-fraction results

    Key Design:
        Validation and test sets are NEVER subsampled.
        Only training is limited to simulate low-data scenarios.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        test_dataset,
        task_names: List[str],
        fraction: float,
        seed: int,
        config: Dict,
        result_dir: str = "results/feature2/low_data",
        device: str = "cpu",
        verbose: bool = True,
    ):
        assert 0.0 < fraction <= 1.0

        self.model = model.to(device)
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.fraction = fraction
        self.seed = seed
        self.config = config
        self.result_dir = result_dir
        self.device = device
        self.verbose = verbose

        os.makedirs(result_dir, exist_ok=True)

        # Subsample training set
        self.train_dataset = create_low_data_subset(
            train_dataset, fraction=fraction, seed=seed
        )

        if verbose:
            print(f"[LowDataTrainer] fraction={fraction:.0%}, "
                  f"train={len(self.train_dataset)}, "
                  f"val={len(val_dataset)}, "
                  f"test={len(test_dataset)}")

        # Training config
        self.lr = config.get("lr", 1e-3)
        self.weight_decay = config.get("weight_decay", 1e-5)
        self.batch_size = config.get("batch_size", 32)
        self.epochs = config.get("epochs", 100)
        self.patience = config.get("patience", 20)
        self.grad_clip = config.get("grad_clip", 1.0)

        # Optimizer — only trainable params
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            trainable = list(model.parameters())

        self.optimizer = torch.optim.Adam(
            trainable, lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=10, verbose=False
        )
        self.early_stopping = TransferEarlyStopping(patience=self.patience)

        # History
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_auc": [],
            "lr": [],
        }

    def _build_loader(self, dataset, shuffle: bool = False):
        return PyGDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    def _train_epoch(self) -> float:
        loader = self._build_loader(self.train_dataset, shuffle=True)
        self.model.train()
        total_loss = 0.0
        n = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model.compute_loss(batch)
            loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
            self.optimizer.step()
            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    @torch.no_grad()
    def _evaluate(self, dataset) -> Tuple[float, Dict[str, float]]:
        loader = self._build_loader(dataset, shuffle=False)
        self.model.eval()

        all_probs, all_labels = [], []
        for batch in loader:
            batch = batch.to(self.device)
            probs = self.model.predict(batch)
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
            if all_labels.ndim > 1:
                tl = all_labels[:, t]
                tp = all_probs[:, t]
            else:
                tl = all_labels
                tp = all_probs

            mask = tl != -1
            if mask.sum() < 10:
                continue
            tlm, tpm = tl[mask], tp[mask]
            if len(np.unique(tlm)) < 2:
                continue
            auc = compute_roc_auc(tlm, tpm)
            if not np.isnan(auc):
                per_task_auc[name] = float(auc)
                valid_aucs.append(auc)

        mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.5
        return mean_auc, per_task_auc

    def train(self, experiment_name: str = "low_data_exp") -> Dict:
        """
        Full low-data training loop.

        Returns:
            results dict with fraction, val_auc, test_auc, per_task_auc, history
        """
        if self.verbose:
            print(f"\n  [{experiment_name}] "
                  f"fraction={self.fraction:.0%}, "
                  f"n_train={len(self.train_dataset)}")

        start = time.time()

        for epoch in range(self.epochs):
            train_loss = self._train_epoch()
            val_auc, _ = self._evaluate(self.val_dataset)
            self.scheduler.step(val_auc)
            lr = self.optimizer.param_groups[0]["lr"]

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(float(train_loss))
            self.history["val_auc"].append(float(val_auc))
            self.history["lr"].append(float(lr))

            if self.verbose and epoch % 20 == 0:
                print(f"    Epoch {epoch:3d} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Val AUC: {val_auc:.4f}")

            if self.early_stopping.step(val_auc, self.model):
                if self.verbose:
                    print(f"    Early stop at epoch {epoch}")
                break

        self.early_stopping.restore_best(self.model)

        # Final evaluation
        val_auc, val_per_task = self._evaluate(self.val_dataset)
        test_auc, test_per_task = self._evaluate(self.test_dataset)
        elapsed = time.time() - start

        results = {
            "experiment_name": experiment_name,
            "fraction": self.fraction,
            "seed": self.seed,
            "n_train": len(self.train_dataset),
            "n_val": len(self.val_dataset),
            "n_test": len(self.test_dataset),
            "val_auc": float(val_auc),
            "test_auc": float(test_auc),
            "best_val_auc": float(self.early_stopping.best_score or val_auc),
            "val_per_task_auc": val_per_task,
            "test_per_task_auc": test_per_task,
            "elapsed_seconds": float(elapsed),
            "config": self.config,
        }

        if self.verbose:
            print(f"    Result: val={val_auc:.4f}, "
                  f"test={test_auc:.4f}, "
                  f"time={elapsed:.1f}s")

        # Save
        result_path = os.path.join(
            self.result_dir, f"{experiment_name}_results.json"
        )
        save_data = dict(results)
        save_data["history"] = {
            k: [float(x) for x in v]
            for k, v in self.history.items()
        }
        with open(result_path, "w") as f:
            json.dump(save_data, f, indent=2)

        return results