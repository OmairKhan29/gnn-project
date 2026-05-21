"""
Checkpoint loader for Feature 3.

Loads F1 or F2 trained models and wraps them
for explanation without any modification.
"""

import os
import torch
from typing import Optional, Dict, Any
from feature3.models.maskable_wrapper import MaskableModelWrapper


# Lazy import to avoid circular deps
def _load_multitask_classifier():
    from models.task_conditioned_egnn import MultiTaskClassifier
    return MultiTaskClassifier


class CheckpointLoader:
    """
    Loads F1/F2 model checkpoints for Feature 3 explanation.

    Supports:
        - Feature 1 base checkpoints
        - Feature 2 aligned/fine-tuned checkpoints
    """

    # Default F1 architecture config
    DEFAULT_F1_CONFIG = {
        'node_dim': 129,
        'edge_dim': 6,
        'hidden_dim': 128,
        'task_dim': 64,
        'num_tasks': 17,
        'num_layers': 4,
        'dropout': 0.1,
    }

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device = torch.device('cpu'),
        model_config: Optional[Dict[str, Any]] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model_config = model_config or self.DEFAULT_F1_CONFIG

    def load(self) -> MaskableModelWrapper:
        """
        Load checkpoint and return wrapped model ready for F3.

        Returns:
            MaskableModelWrapper with frozen base model
        """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"Train F1 model first with:\n"
                f"  python scripts/train_multitask.py "
                f"--model task_conditioned --pcgrad"
            )

        print(f"[CheckpointLoader] Loading: {self.checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device
        )

        # Extract state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                epoch = checkpoint.get('epoch', '?')
                auc = checkpoint.get('val_auc', '?')
                print(f"[CheckpointLoader] Epoch={epoch}, Val AUC={auc}")
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Build base model
        MultiTaskClassifier = _load_multitask_classifier()
        base_model = MultiTaskClassifier(**self.model_config)
        base_model.load_state_dict(state_dict, strict=False)
        base_model.to(self.device)
        base_model.eval()

        n_params = sum(p.numel() for p in base_model.parameters())
        print(f"[CheckpointLoader] Parameters: {n_params:,}")

        # Wrap for Feature 3
        wrapped = MaskableModelWrapper(base_model)

        print(f"[CheckpointLoader] Ready. Model frozen for explanation.")
        return wrapped

    @classmethod
    def from_f1(
        cls,
        checkpoint_dir: str = 'checkpoints',
        device: torch.device = torch.device('cpu'),
    ) -> MaskableModelWrapper:
        """
        Convenience: load best F1 checkpoint.
        """
        path = os.path.join(checkpoint_dir, 'best_model.pt')
        return cls(path, device).load()

    @classmethod
    def from_f2(
        cls,
        alignment: str = 'prototype',
        checkpoint_dir: str = 'checkpoints/feature2/alignment',
        device: torch.device = torch.device('cpu'),
    ) -> MaskableModelWrapper:
        """
        Convenience: load best F2 aligned checkpoint.

        Args:
            alignment: 'contrastive', 'domain', or 'prototype'
        """
        path = os.path.join(checkpoint_dir, f'{alignment}_best.pt')
        return cls(path, device).load()