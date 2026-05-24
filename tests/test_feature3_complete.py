"""
Complete Feature 3 tests - validates integration without modifying F1.
"""

import pytest
import torch
from torch_geometric.data import Data

# Feature 3 imports
from feature3.models.maskable_wrapper import MaskableModelWrapper
from feature3.explainer.gnn_explainer import GNNExplainer
from feature3.analysis.substructure_mapper import SubstructureMapper


def create_mock_f1_model():
    """Create a mock F1-like model for testing."""
    class MockF1Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_tasks = 17
            self.linear = torch.nn.Linear(10, 1)
        
        def forward(self, data, task_idx=None):
            # Simple forward that uses edge_attr if present
            x = data.x.mean(dim=0, keepdim=True)
            return torch.sigmoid(self.linear(x))
    
    return MockF1Model()


class TestMaskableWrapper:
    def test_wrapper_creation(self):
        base = create_mock_f1_model()
        wrapped = MaskableModelWrapper(base)
        assert wrapped.num_tasks == 17
    
    def test_forward_without_mask(self):
        base = create_mock_f1_model()
        wrapped = MaskableModelWrapper(base)
        
        data = Data(
            x=torch.randn(5, 10),
            edge_index=torch.tensor([[0,1,1,2], [1,0,2,1]]),
            edge_attr=torch.randn(4, 6)
        )
        
        out = wrapped(data, task_idx=0)
        assert out.shape == (1, 1)
    
    def test_forward_with_mask(self):
        """Critical test: edge_weight changes output."""
        base = create_mock_f1_model()
        wrapped = MaskableModelWrapper(base)
        
        data = Data(
            x=torch.randn(5, 10),
            edge_index=torch.tensor([[0,1,1,2], [1,0,2,1]]),
            edge_attr=torch.ones(4, 6)  # All ones for clear effect
        )
        
        # Without mask
        out1 = wrapped(data, task_idx=0, edge_weight=None)
        
        # With zero mask (should zero out edge_attr)
        out2 = wrapped(data, task_idx=0, edge_weight=torch.zeros(4))
        
        # With ones mask (should be same as no mask)
        out3 = wrapped(data, task_idx=0, edge_weight=torch.ones(4))
        
        # out2 should differ from out1/out3 because edges are masked
        assert not torch.allclose(out1, out2, atol=1e-4)
        assert torch.allclose(out1, out3, atol=1e-4)


class TestGNNExplainerIntegration:
    def test_explainer_runs(self):
        base = create_mock_f1_model()
        wrapped = MaskableModelWrapper(base)
        explainer = GNNExplainer(wrapped, epochs=5)  # Fast test
        
        data = Data(
            x=torch.randn(3, 10),
            edge_index=torch.tensor([[0,1,1,2], [1,0,2,1]]),
            edge_attr=torch.randn(4, 6),
            pos=torch.randn(3, 3)
        )
        
        result = explainer.explain(data, task_idx=0)
        
        assert 'edge_mask' in result
        assert 'node_importance' in result
        assert result['edge_mask'].shape == (4,)
        assert result['node_importance'].shape == (3,)
        assert 0 <= result['edge_mask'].min() <= result['edge_mask'].max() <= 1


class TestSubstructureMapper:
    def test_nitro_detection(self):
        mapper = SubstructureMapper()
        results = mapper.map_to_substructures(
            'O=[N+]([O-])c1ccccc1',  # Nitrobenzene
            torch.tensor([0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
        )
        
        assert results['nitro']['present']
        assert results['nitro']['score'] > 0.5  # High importance on nitro atoms
    
    def test_benzene_detection(self):
        mapper = SubstructureMapper()
        results = mapper.map_to_substructures('c1ccccc1', torch.ones(6) * 0.8)
        assert results['benzene']['present']
        assert results['benzene']['score'] == pytest.approx(0.8, abs=0.01)