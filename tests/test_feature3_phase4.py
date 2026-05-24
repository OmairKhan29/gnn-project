"""Tests for Feature 3 Phase 4: Evaluation + Tables."""

import pytest
import torch
import numpy as np
import os
from torch_geometric.data import Data
import torch.nn as nn

from feature3.evaluation.fidelity import FidelityEvaluator
from feature3.evaluation.stability import StabilityEvaluator
from feature3.tables.latex_tables import LatexTableGenerator
from feature3.models.maskable_wrapper import MaskableModelWrapper


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_tasks = 4
        self.lin = nn.Linear(13, 1)

    def forward(self, data, task_idx=None):
        x = data.x.mean(0)
        ea = data.edge_attr.mean(0) if data.edge_attr is not None else torch.zeros(3)
        return self.lin(torch.cat([x, ea]).unsqueeze(0))


@pytest.fixture
def model():
    base = SimpleModel()
    return MaskableModelWrapper(base)


@pytest.fixture
def simple_data():
    return Data(
        x=torch.randn(4, 10),
        edge_index=torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]], dtype=torch.long),
        edge_attr=torch.ones(6, 3),
        pos=torch.randn(4, 3),
    )


# ── FidelityEvaluator Tests ───────────────────────────────────────────────

class TestFidelityEvaluator:

    def test_evaluate_single_returns_dict(self, model, simple_data):
        ev = FidelityEvaluator(model, threshold=0.5)
        mask = torch.rand(6)
        result = ev.evaluate_single(simple_data, mask, task_idx=0)
        assert isinstance(result, dict)

    def test_required_keys(self, model, simple_data):
        ev = FidelityEvaluator(model, threshold=0.5)
        mask = torch.rand(6)
        result = ev.evaluate_single(simple_data, mask, task_idx=0)
        for key in ['fidelity_plus', 'fidelity_minus', 'sparsity',
                    'original_pred', 'pos_pred', 'neg_pred']:
            assert key in result

    def test_fidelity_non_negative(self, model, simple_data):
        ev = FidelityEvaluator(model, threshold=0.5)
        mask = torch.rand(6)
        result = ev.evaluate_single(simple_data, mask, task_idx=0)
        assert result['fidelity_plus'] >= 0
        assert result['fidelity_minus'] >= 0

    def test_sparsity_range(self, model, simple_data):
        ev = FidelityEvaluator(model, threshold=0.5)
        for _ in range(5):
            mask = torch.rand(6)
            result = ev.evaluate_single(simple_data, mask, task_idx=0)
            assert 0.0 <= result['sparsity'] <= 1.0

    def test_all_zeros_mask_high_sparsity(self, model, simple_data):
        ev = FidelityEvaluator(model, threshold=0.5)
        mask = torch.zeros(6)
        result = ev.evaluate_single(simple_data, mask, task_idx=0)
        assert result['sparsity'] == 1.0

    def test_all_ones_mask_zero_sparsity(self, model, simple_data):
        ev = FidelityEvaluator(model, threshold=0.5)
        mask = torch.ones(6)
        result = ev.evaluate_single(simple_data, mask, task_idx=0)
        assert result['sparsity'] == 0.0

    def test_evaluate_dataset(self, model, simple_data):
        ev = FidelityEvaluator(model, threshold=0.5)
        masks = [torch.rand(6) for _ in range(3)]
        result = ev.evaluate_dataset([simple_data]*3, masks, task_idx=0)
        assert 'fidelity_plus_mean' in result
        assert 'n_evaluated' in result
        assert result['n_evaluated'] == 3

    def test_empty_dataset(self, model):
        ev = FidelityEvaluator(model, threshold=0.5)
        result = ev.evaluate_dataset([], [], task_idx=0)
        assert result['n_evaluated'] == 0
        assert result['fidelity_plus_mean'] == 0.0


# ── StabilityEvaluator Tests ─────────────────────────────────────────────

class TestStabilityEvaluator:

    @pytest.fixture
    def fast_explainer(self, model):
        from feature3.explainer.gnn_explainer import GNNExplainer
        return GNNExplainer(model, epochs=3)

    def test_evaluate_single_keys(self, fast_explainer, simple_data):
        ev = StabilityEvaluator(n_runs=2)
        result = ev.evaluate_single(fast_explainer, simple_data, task_idx=0)
        assert 'mean_correlation' in result
        assert 'n_runs' in result

    def test_correlation_range(self, fast_explainer, simple_data):
        ev = StabilityEvaluator(n_runs=3)
        result = ev.evaluate_single(fast_explainer, simple_data, task_idx=0)
        c = result['mean_correlation']
        assert -1.0 <= c <= 1.0

    def test_evaluate_dataset(self, fast_explainer, simple_data):
        ev = StabilityEvaluator(n_runs=2)
        result = ev.evaluate_dataset(
            fast_explainer, [simple_data], task_idx=0, max_mols=1
        )
        assert 'stability_mean' in result
        assert result['n_evaluated'] >= 0


# ── LaTeX Table Tests ─────────────────────────────────────────────────────

class TestLatexTableGenerator:

    @pytest.fixture
    def gen(self, tmp_path):
        return LatexTableGenerator(output_dir=str(tmp_path / 'tables'))

    @pytest.fixture
    def sample_metrics(self):
        return {
            'NR-AR': {
                'fidelity_plus_mean': 0.72, 'fidelity_plus_std': 0.05,
                'fidelity_minus_mean': 0.68, 'fidelity_minus_std': 0.04,
                'sparsity_mean': 0.75, 'sparsity_std': 0.08,
                'stability_mean': 0.81, 'stability_std': 0.03,
            },
            'SR-MMP': {
                'fidelity_plus_mean': 0.65, 'fidelity_plus_std': 0.06,
                'fidelity_minus_mean': 0.61, 'fidelity_minus_std': 0.05,
                'sparsity_mean': 0.70, 'sparsity_std': 0.09,
                'stability_mean': 0.77, 'stability_std': 0.04,
            },
        }

    # Table 1
    def test_table1_is_latex(self, gen, sample_metrics):
        latex = gen.table1_explanation_metrics(sample_metrics)
        assert r'\begin{table}' in latex
        assert r'\end{table}' in latex
        assert r'\toprule' in latex
        assert r'\midrule' in latex
        assert r'\bottomrule' in latex

    def test_table1_contains_tasks(self, gen, sample_metrics):
        latex = gen.table1_explanation_metrics(sample_metrics)
        assert 'NR-AR' in latex
        assert 'SR-MMP' in latex

    def test_table1_saved_to_disk(self, gen, sample_metrics, tmp_path):
        gen.table1_explanation_metrics(sample_metrics)
        path = tmp_path / 'tables' / 'table1_explanation_metrics.tex'
        assert path.exists()
        content = path.read_text()
        assert r'\begin{table}' in content

    def test_table1_contains_values(self, gen, sample_metrics):
        latex = gen.table1_explanation_metrics(sample_metrics)
        assert '0.720' in latex or '0.72' in latex

    # Table 2
    def test_table2_is_latex(self, gen):
        subs = {
            'NR-AR': [('nitro', 0.9), ('benzene', 0.8)],
            'BBBP': [('alcohol', 0.7), ('amide', 0.6)],
        }
        latex = gen.table2_top_substructures(subs, top_k=2)
        assert r'\begin{table}' in latex
        assert 'Top-1' in latex
        assert 'NR-AR' in latex

    def test_table2_saved(self, gen, tmp_path):
        subs = {'Task': [('nitro', 0.9)]}
        gen.table2_top_substructures(subs, top_k=1)
        path = tmp_path / 'tables' / 'table2_top_substructures.tex'
        assert path.exists()

    # Table 3
    def test_table3_is_latex(self, gen):
        ablation = {
            'default': {
                'fidelity_plus_mean': 0.72,
                'fidelity_minus_mean': 0.68,
                'sparsity_mean': 0.75,
                'params': {'edge_size': 0.005,
                           'edge_entropy': 1.0, 'epochs': 100},
            },
            'high_entropy': {
                'fidelity_plus_mean': 0.69,
                'fidelity_minus_mean': 0.65,
                'sparsity_mean': 0.80,
                'params': {'edge_size': 0.005,
                           'edge_entropy': 2.0, 'epochs': 100},
            },
        }
        latex = gen.table3_ablation(ablation)
        assert r'\begin{table}' in latex
        assert 'default' in latex
        assert r'\star' in latex  # Best config marker

    def test_table3_saved(self, gen, tmp_path):
        ablation = {
            'cfg1': {
                'fidelity_plus_mean': 0.7,
                'fidelity_minus_mean': 0.6,
                'sparsity_mean': 0.75,
                'params': {'edge_size': 0.005,
                           'edge_entropy': 1.0, 'epochs': 100},
            }
        }
        gen.table3_ablation(ablation)
        path = tmp_path / 'tables' / 'table3_ablation.tex'
        assert path.exists()