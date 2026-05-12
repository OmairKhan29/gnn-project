"""
Pretrained Encoder Loader for Feature 2.
Loads Feature 1 checkpoints and exposes them for transfer learning.
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.task_conditioned_egnn import (
    TaskConditionedEGNN,
    MultiTaskClassifier,
    HardSharingClassifier,
)
from models.egnn import EGNN, EGNNClassifier


# ─────────────────────────────────────────────
# Checkpoint Loading
# ─────────────────────────────────────────────

def load_feature1_checkpoint(
    checkpoint_path: str,
    model_type: str = "task_conditioned",   # "task_conditioned" | "hard_sharing" | "single_task"
    device: str = "cpu",
    verbose: bool = True,
) -> nn.Module:
    """
    Load a Feature 1 trained model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        model_type: Architecture type used during Feature 1 training
        device: Device to load onto
        verbose: Print loading info

    Returns:
        Loaded nn.Module (full classifier)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run Feature 1 training first:\n"
            f"  python scripts/train_multitask.py --model task_conditioned --device cpu"
        )

    if verbose:
        print(f"[checkpoint] Loading {model_type} from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config if saved
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})

    hidden_dim = model_config.get("hidden_dim", 128)
    n_layers = model_config.get("n_layers", 4)
    task_dim = model_config.get("task_dim", 64)
    dropout = model_config.get("dropout", 0.1)
    num_tasks = model_config.get("num_tasks", 17)

    # Build the correct architecture
    if model_type == "task_conditioned":
        model = MultiTaskClassifier(
            node_dim=129,
            edge_dim=6,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            num_tasks=num_tasks,
            task_dim=task_dim,
            dropout=dropout,
        )
    elif model_type == "hard_sharing":
        model = HardSharingClassifier(
            node_dim=129,
            edge_dim=6,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            num_tasks=num_tasks,
            dropout=dropout,
        )
    elif model_type == "single_task":
        model = EGNNClassifier(
            node_dim=129,
            edge_dim=6,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Load weights
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[checkpoint] Loaded {n_params:,} parameters")
        print(f"[checkpoint] hidden_dim={hidden_dim}, n_layers={n_layers}, "
              f"task_dim={task_dim}, num_tasks={num_tasks}")

    return model


# ─────────────────────────────────────────────
# Encoder Extractor
# ─────────────────────────────────────────────

class FrozenEncoder(nn.Module):
    """
    Wraps a Feature 1 encoder with frozen parameters.
    Used for linear probe evaluation.

    Exposes:
        forward(batch) → graph_embeddings [B, hidden_dim]
    """

    def __init__(self, full_model: nn.Module, model_type: str = "task_conditioned"):
        super().__init__()
        self.model_type = model_type
        self.full_model = full_model

        # Extract encoder sub-module
        if model_type == "task_conditioned":
            self.encoder = full_model.encoder     # TaskConditionedEGNN
        elif model_type == "hard_sharing":
            self.encoder = full_model.encoder     # EGNN
        elif model_type == "single_task":
            self.encoder = full_model.encoder     # EGNN
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Freeze all encoder parameters
        self.freeze_encoder()

    def freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

        frozen_count = sum(
            p.numel() for p in self.encoder.parameters() if not p.requires_grad
        )
        print(f"[FrozenEncoder] Frozen {frozen_count:,} parameters")

    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters (for fine-tuning)."""
        for param in self.encoder.parameters():
            param.requires_grad = True

        unfrozen_count = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        print(f"[FrozenEncoder] Unfrozen {unfrozen_count:,} parameters")

    def unfreeze_top_layers(self, num_layers: int = 2):
        """
        Unfreeze only the top N EGNN layers (for partial fine-tuning).
        Lower layers remain frozen.
        """
        # First freeze everything
        self.freeze_encoder()

        # Unfreeze top layers
        if hasattr(self.encoder, "layers"):
            total_layers = len(self.encoder.layers)
            for i in range(total_layers - num_layers, total_layers):
                for param in self.encoder.layers[i].parameters():
                    param.requires_grad = True

        unfrozen = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[FrozenEncoder] Top {num_layers} layers unfrozen ({unfrozen:,} params)")

    def get_output_dim(self) -> int:
        """Return the embedding output dimension."""
        if hasattr(self.encoder, "hidden_dim"):
            return self.encoder.hidden_dim
        # Fallback: run a dummy forward pass
        return 128

    def forward(self, batch, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Extract graph-level embeddings.

        Args:
            batch: PyG Batch object
            task_id: Task ID for task-conditioned encoder (optional)

        Returns:
            embeddings: [B, hidden_dim]
        """
        if self.model_type == "task_conditioned":
            # Need task_id for task-conditioned encoder
            if task_id is None:
                # Use a default task embedding (task 0)
                task_id = 0
            task_ids = torch.zeros(batch.num_graphs, dtype=torch.long,
                                   device=batch.x.device) + task_id
            return self.encoder(batch, task_ids)
        else:
            return self.encoder(batch)


# ─────────────────────────────────────────────
# Mock Checkpoint Generator (for testing without trained models)
# ─────────────────────────────────────────────

def create_mock_checkpoint(
    save_path: str,
    model_type: str = "task_conditioned",
    hidden_dim: int = 64,
    n_layers: int = 2,
    task_dim: int = 32,
    num_tasks: int = 17,
):
    """
    Create a random-weight checkpoint for testing transfer pipeline
    when no trained Feature 1 checkpoint exists.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if model_type == "task_conditioned":
        model = MultiTaskClassifier(
            node_dim=129,
            edge_dim=6,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            num_tasks=num_tasks,
            task_dim=task_dim,
        )
    elif model_type == "hard_sharing":
        model = HardSharingClassifier(
            node_dim=129,
            edge_dim=6,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            num_tasks=num_tasks,
        )
    else:
        model = EGNNClassifier(
            node_dim=129,
            edge_dim=6,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "model": {
                "hidden_dim": hidden_dim,
                "n_layers": n_layers,
                "task_dim": task_dim,
                "num_tasks": num_tasks,
                "dropout": 0.1,
            }
        },
        "epoch": 0,
        "val_auc": 0.5,
        "is_mock": True,
    }

    torch.save(checkpoint, save_path)
    print(f"[mock] Saved mock {model_type} checkpoint to {save_path}")
    return save_path