"""
MaskableModelWrapper
====================
Wraps any trained F1/F2 model to support edge_weight masking
without modifying original model code.

Strategy:
    Since F1 EGNNLayer uses edge_attr in message passing,
    multiplying edge_attr by edge_weight effectively scales
    each bond's contribution to message passing.
    
    edge_attr_masked = edge_attr * edge_weight[:, None]
    
    When edge_weight[i] = 0.0 → bond i contributes nothing
    When edge_weight[i] = 1.0 → bond i contributes normally
    When edge_weight[i] = 0.5 → bond i contributes at half strength

This is mathematically equivalent to soft edge removal.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import Optional


class MaskableModelWrapper(nn.Module):
    """
    Wraps a trained F1/F2 MultiTaskClassifier to enable
    edge masking for GNNExplainer.
    
    Args:
        base_model: Trained MultiTaskClassifier from Feature 1 or 2
        
    Example:
        # Load F1 model
        base_model = MultiTaskClassifier(...)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Wrap for explanation
        model = MaskableModelWrapper(base_model)
        
        # Use normally (no mask)
        pred = model(data, task_idx=0)
        
        # Use with mask (for GNNExplainer)
        pred = model(data, task_idx=0, edge_weight=mask)
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()

        # Store base model
        self.base_model = base_model

        # Freeze ALL base model parameters
        # Feature 3 never trains the original model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.eval()

    def forward(
        self,
        data: Data,
        task_idx: Optional[int] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional edge masking.

        Args:
            data: PyG Data (x, edge_index, edge_attr, pos, batch)
            task_idx: Task head to use (0-16 for F1 model)
            edge_weight: [E] soft mask in [0, 1]
                         None = standard forward pass
                         tensor = masked forward pass

        Returns:
            logits: [batch_size, 1] prediction logits
        """
        if edge_weight is None:
            # Standard pass - no masking needed
            return self.base_model(data, task_idx=task_idx)

        # Build masked data object
        # We scale edge_attr by edge_weight to simulate
        # soft edge removal without modifying model code
        masked_edge_attr = None
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            # edge_attr: [E, F]
            # edge_weight: [E]
            # Result: [E, F] where each row scaled by weight
            masked_edge_attr = data.edge_attr * edge_weight.unsqueeze(1)

        # Build new Data object with masked edges
        masked_data = Data(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=masked_edge_attr,
            pos=data.pos if hasattr(data, 'pos') else None,
            batch=(
                data.batch
                if hasattr(data, 'batch') and data.batch is not None
                else torch.zeros(
                    data.x.size(0),
                    dtype=torch.long,
                    device=data.x.device
                )
            ),
        )

        return self.base_model(masked_data, task_idx=task_idx)

    @property
    def num_tasks(self) -> int:
        """Expose num_tasks from wrapped model."""
        if hasattr(self.base_model, 'num_tasks'):
            return self.base_model.num_tasks
        return 17  # F1 default

    @property
    def task_names(self):
        """Task names from F1."""
        return [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
            'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
            'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53',
            'ClinTox_CT', 'ClinTox_FDA',
            'BBBP', 'BACE', 'HIV_active',
        ]

    def get_task_name(self, task_idx: int) -> str:
        """Get human-readable task name."""
        names = self.task_names
        if 0 <= task_idx < len(names):
            return names[task_idx]
        return f'Task_{task_idx}'

    def train(self, mode: bool = True):
        """Keep base model in eval mode always."""
        super().train(mode)
        self.base_model.eval()
        return self

    def to(self, device):
        """Move to device."""
        self.base_model = self.base_model.to(device)
        return self

    def parameters(self, recurse: bool = True):
        """Return empty iterator - wrapper has no trainable params."""
        return iter([])

    def __repr__(self):
        return (
            f'MaskableModelWrapper(\n'
            f'  base_model={self.base_model.__class__.__name__},\n'
            f'  num_tasks={self.num_tasks},\n'
            f'  trainable_params=0 (frozen)\n'
            f')'
        )