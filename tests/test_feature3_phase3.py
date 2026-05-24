"""Tests for Feature 3 Phase 3: Visualization."""

import pytest
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import os

from feature3.visualization.mol_visualizer import MoleculeVisualizer, _importance_to_rgb
from feature3.visualization.figure_builder import FigureBuilder


class TestImportanceToRgb:

    def test_returns_tuple_of_three(self):
        c = _importance_to_rgb(0.5)
        assert len(c) == 3

    def test_range_valid(self):
        for val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            c = _importance_to_rgb(val)
            assert all(0.0 <= ch <= 1.0 for ch in c)

    def test_low_differs_from_high(self):
        c_low = _importance_to_rgb(0.0)
        c_high = _importance_to_rgb(1.0)
        assert c_low != c_high

    def test_clipping(self):
        c_below = _importance_to_rgb(-0.5)
        c_min = _importance_to_rgb(0.0)
        c_above = _importance_to_rgb(1.5)
        c_max = _importance_to_rgb(1.0)
        assert c_below == c_min
        assert c_above == c_max


class TestMoleculeVisualizer:

    @pytest.fixture
    def viz(self):
        return MoleculeVisualizer(img_size=(200, 200))

    def test_invalid_smiles_returns_none(self, viz):
        result = viz.draw_atom_importance('INVALID', torch.ones(5))
        assert result is None

    def test_draw_benzene(self, viz):
        imp = torch.rand(6)
        result = viz.draw_atom_importance('c1ccccc1', imp)
        # Result is either PIL Image or None (if RDKit Cairo not available)
        # We just verify it doesn't crash
        assert result is None or hasattr(result, 'size')

    def test_draw_edge_importance_invalid(self, viz):
        edge_index = torch.tensor([[0,1],[1,0]])
        mask = torch.tensor([0.8, 0.8])
        result = viz.draw_edge_importance('INVALID', edge_index, mask)
        assert result is None

    def test_draw_edge_importance_valid(self, viz):
        edge_index = torch.tensor([[0,1,1,2],[1,0,2,1]])
        mask = torch.rand(4)
        result = viz.draw_edge_importance('CCC', edge_index, mask)
        assert result is None or hasattr(result, 'size')

    def test_normalize_flag(self, viz):
        """Should not crash with various normalization settings."""
        imp_large = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        viz.draw_atom_importance('c1ccccc1', imp_large, normalize=True)

    def test_threshold_flag(self, viz):
        imp = torch.rand(6)
        viz.draw_atom_importance('c1ccccc1', imp, threshold=0.8)


class TestFigureBuilder:

    @pytest.fixture
    def builder(self, tmp_path):
        return FigureBuilder(
            output_dir=str(tmp_path / 'figures'),
            dpi=72,  # Low DPI for fast tests
        )

    def make_exp(self, n_atoms=5, n_edges=6):
        return {
            'node_importance': torch.rand(n_atoms),
            'edge_mask': torch.rand(n_edges),
            'prediction': float(torch.rand(1)),
            'target': float(torch.randint(0, 2, (1,))),
        }

    def test_fig3_substructure_importance(self, builder):
        summary = {
            'nitro': {'mean': 0.9, 'std': 0.1, 'count': 5, 'frequency': 0.5},
            'benzene': {'mean': 0.7, 'std': 0.1, 'count': 10, 'frequency': 0.8},
            'amine': {'mean': 0.3, 'std': 0.05, 'count': 0, 'frequency': 0.0},
        }
        path = builder.fig3_substructure_importance(summary, 'NR-AR')
        if path:
            assert os.path.exists(path)

    def test_fig4_fidelity_comparison(self, builder):
        metrics = {
            'NR-AR': {
                'fidelity_plus_mean': 0.7,
                'fidelity_plus_std': 0.05,
                'fidelity_minus_mean': 0.6,
                'fidelity_minus_std': 0.04,
            },
            'SR-MMP': {
                'fidelity_plus_mean': 0.65,
                'fidelity_plus_std': 0.06,
                'fidelity_minus_mean': 0.55,
                'fidelity_minus_std': 0.05,
            },
        }
        path = builder.fig4_fidelity_comparison(metrics)
        assert os.path.exists(path)

    def test_fig5_stability(self, builder):
        stability = {'NR-AR': 0.85, 'SR-MMP': 0.72, 'BBBP': 0.91}
        path = builder.fig5_stability(stability)
        assert os.path.exists(path)

    def test_fig6_cross_task_heatmap(self, builder):
        scores = {
            'NR-AR': {'nitro': 0.9, 'benzene': 0.7, 'amine': 0.3},
            'BBBP': {'nitro': 0.4, 'benzene': 0.8, 'amine': 0.6},
        }
        path = builder.fig6_cross_task_heatmap(scores, top_k_groups=3)
        assert os.path.exists(path)

    def test_fig8_per_task_metrics(self, builder):
        metrics = {
            'NR-AR': {
                'fidelity_plus_mean': 0.7, 'fidelity_plus_std': 0.05,
                'fidelity_minus_mean': 0.6, 'fidelity_minus_std': 0.04,
                'sparsity_mean': 0.75, 'sparsity_std': 0.06,
                'stability_mean': 0.82, 'stability_std': 0.03,
            },
        }
        path = builder.fig8_per_task_metrics(metrics)
        assert os.path.exists(path)

    def test_output_dir_created(self, tmp_path):
        new_dir = str(tmp_path / 'new' / 'nested' / 'dir')
        builder = FigureBuilder(output_dir=new_dir, dpi=72)
        assert os.path.exists(new_dir)