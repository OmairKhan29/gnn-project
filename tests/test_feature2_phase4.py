"""
Tests for Feature 2 Phase 4: Ablations, Visualizations, LaTeX Tables.
"""

import pytest
import torch
import numpy as np
import os
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def mock_comparison_results(tmp_path):
    """Create mock transfer comparison results."""
    data = {
        "unaligned": {
            "sider": {"linear_probe": {"test_auc_mean": 0.65, "test_auc_std": 0.01, "n_seeds": 3, "per_seed": [0.64, 0.65, 0.66]}},
        },
        "contrastive": {
            "sider": {"linear_probe": {"test_auc_mean": 0.68, "test_auc_std": 0.01, "n_seeds": 3, "per_seed": [0.67, 0.68, 0.69]}},
        },
    }
    out_dir = tmp_path / "transfer_comparison"
    out_dir.mkdir(parents=True)
    with open(out_dir / "full_comparison.json", "w") as f:
        json.dump(data, f)
    return data


@pytest.fixture
def mock_ablation_results(tmp_path):
    """Create mock ablation results."""
    data = {
        "no_alignment": {
            "0": {"test_auc": 0.65},
            "1": {"test_auc": 0.66},
            "2": {"test_auc": 0.64},
        },
        "lambda_0.5": {
            "0": {"test_auc": 0.68},
            "1": {"test_auc": 0.69},
            "2": {"test_auc": 0.67},
        },
    }
    out_dir = tmp_path / "ablations"
    out_dir.mkdir(parents=True)
    with open(out_dir / "all_ablation_results.json", "w") as f:
        json.dump(data, f)
    return data


@pytest.fixture
def mock_low_data_results(tmp_path):
    """Create mock low-data results."""
    data = [
        {"dataset": "sider", "strategy": "scratch", "fraction": 0.5, "test_auc": 0.62, "error": ""},
        {"dataset": "sider", "strategy": "linear_probe", "fraction": 0.5, "test_auc": 0.68, "error": ""},
        {"dataset": "muv", "strategy": "scratch", "fraction": 0.5, "test_auc": 0.60, "error": ""},
    ]
    out_dir = tmp_path / "low_data"
    out_dir.mkdir(parents=True)
    with open(out_dir / "all_low_data_results.json", "w") as f:
        json.dump(data, f)
    return data


# ─────────────────────────────────────────────
# Test 1: Statistical Significance
# ─────────────────────────────────────────────

class TestStatisticalSignificance:

    def test_paired_t_test_two_sided(self):
        from feature2.statistical.significance import paired_t_test
        a = [0.65, 0.66, 0.67]
        b = [0.62, 0.63, 0.64]
        result = paired_t_test(a, b, alternative="two-sided")
        assert result["can_compute"]
        assert "t_statistic" in result
        assert result["p_value"] >= 0.0 and result["p_value"] <= 1.0

    def test_paired_t_test_greater_alternative(self):
        from feature2.statistical.significance import paired_t_test
        a = [0.70, 0.71, 0.72]
        b = [0.65, 0.66, 0.67]
        result = paired_t_test(a, b, alternative="greater")
        assert result["can_compute"]
        assert result["significant"]  # Should be significant one-tailed

    def test_cohens_d_large_effect(self):
        from feature2.statistical.significance import cohen_d
        a = [0.80, 0.81, 0.82]
        b = [0.65, 0.66, 0.67]
        result = cohen_d(a, b)
        assert result["interpretation"] == "large"
        assert result["abs_cohen_d"] > 0.8

    def test_bootstrap_ci(self):
        from feature2.statistical.significance import bootstrap_confidence_interval
        values = [0.65, 0.67, 0.68, 0.66, 0.69]
        result = bootstrap_confidence_interval(values, n_bootstrap=1000)
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] < result["estimate"] < result["ci_upper"]

    def test_format_significance_stars(self):
        from feature2.statistical.significance import format_significance_stars
        assert format_significance_stars(0.001) == "***"
        assert format_significance_stars(0.01) == "**"
        assert format_significance_stars(0.05) == "*"
        assert format_significance_stars(0.1) == "."
        assert format_significance_stars(0.5) == ""

    def test_analyze_ablations(self, mock_ablation_results, tmp_path):
        from feature2.statistical.significance import analyze_ablations
        results = mock_ablation_results
        analysis = analyze_ablations(results)
        assert "lambda_0.5" in analysis
        assert "comparison_vs_baseline" in analysis["lambda_0.5"]


# ─────────────────────────────────────────────
# Test 2: LaTeX Table Generation
# ─────────────────────────────────────────────

class TestLatexTableGeneration:

    def test_main_results_table_generation(self, mock_comparison_results, tmp_path):
        from feature2.tables.latex_generator import generate_main_results_table
        save_path = str(tmp_path / "main_results.tex")
        tex = generate_main_results_table(mock_comparison_results, save_path)
        assert os.path.exists(save_path)
        assert "\\begin{table}" in tex
        assert "\\caption" in tex
        assert "Strategy" in tex

    def test_ablation_table_generation(self, mock_ablation_results, tmp_path):
        from feature2.tables.latex_generator import generate_ablation_table
        from feature2.statistical.significance import analyze_ablations
        results = mock_ablation_results
        analysis = analyze_ablation_results(results)
        save_path = str(tmp_path / "ablation.tex")
        tex = generate_ablation_table(analysis, save_path)
        assert os.path.exists(save_path)
        assert "Ablation" in tex

    def test_low_data_table_generation(self, mock_low_data_results, tmp_path):
        from feature2.tables.latex_generator import generate_low_data_table
        save_path = str(tmp_path / "low_data_sider.tex")
        tex = generate_low_data_table(mock_low_data_results, "sider", save_path)
        assert os.path.exists(save_path)
        assert "Training Data" in tex
        assert "SCRATCH" in tex

    def test_invalid_dataset_no_crash(self, tmp_path):
        from feature2.tables.latex_generator import generate_low_data_table
        save_path = str(tmp_path / "empty.tex")
        result = generate_low_data_table([], "nonexistent", save_path)
        assert os.path.exists(save_path)


# ─────────────────────────────────────────────
# Test 3: Visualization Functions
# ─────────────────────────────────────────────

class TestVisualizationFunctions:

    def test_fig_transfer_gain_bars_creates_file(self, mock_comparison_results, tmp_path):
        from feature2.visualization.figure_builder import fig_transfer_gain_bars
        save_path = str(tmp_path / "test_gain.png")
        fig_transfer_gain_bars(mock_comparison_results, save_path)
        assert os.path.exists(save_path)

    def test_fig_learning_curves_creates_file(self, mock_low_data_results, tmp_path):
        from feature2.visualization.figure_builder import fig_learning_curves
        save_path = str(tmp_path / "test_curve.png")
        low_data_dir = str(tmp_path / "low_data")
        os.makedirs(low_data_dir, exist_ok=True)
        # Write dummy result files
        for r in mock_low_data_results:
            fname = f"lowdata_{r['dataset']}_{r['strategy']}_frac{int(r['fraction']*100)}.json"
            with open(os.path.join(low_data_dir, fname), "w") as f:
                json.dump(r, f)
        
        fig_learning_curves(low_data_dir, "sider", save_path)
        assert os.path.exists(save_path)

    def test_fig_ablation_plot_creates_file(self, tmp_path):
        from feature2.visualization.figure_builder import fig_ablation_plot
        analysis = {
            "lambda_0.5": {
                "summary": {"mean": 0.68, "std": 0.01},
            },
            "proj_dim_64": {
                "summary": {"mean": 0.67, "std": 0.01},
            },
        }
        save_path = str(tmp_path / "test_ablation.png")
        fig_ablation_plot(analysis, save_path)
        assert os.path.exists(save_path)


# ─────────────────────────────────────────────
# Test 4: Integration
# ─────────────────────────────────────────────

class TestIntegrationPhase4:

    def test_full_pipeline_no_errors(self, mock_comparison_results, tmp_path):
        """Verify all Phase 4 functions run without crashing."""
        from feature2.visualization.figure_builder import fig_transfer_gain_bars, fig_ablation_plot
        from feature2.tables.latex_generator import generate_main_results_table
        from feature2.statistical.significance import analyze_ablations

        # Figure generation
        fig_transfer_gain_bars(mock_comparison_results, str(tmp_path / "fig.png"))
        assert True  # If we reach here, it didn't crash

        # Table generation
        generate_main_results_table(mock_comparison_results, str(tmp_path / "table.tex"))
        assert os.path.exists(tmp_path / "table.tex")

        # Statistical analysis
        ablation = {
            "baseline": {"0": {"test_auc": 0.65}, "1": {"test_auc": 0.66}},
            "test": {"0": {"test_auc": 0.68}, "1": {"test_auc": 0.69}},
        }
        analysis = analyze_ablations(ablation)
        assert "test" in analysis


# ─────────────────────────────────────────────
# Test 5: Edge Cases
# ─────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_comparison_results(self, tmp_path):
        from feature2.visualization.figure_builder import fig_transfer_gain_bars
        save_path = str(tmp_path / "empty.png")
        fig_transfer_gain_bars({}, save_path)
        # Should handle gracefully (may produce empty plot)

    def test_single_sample_ttest(self):
        from feature2.statistical.significance import paired_t_test
        a = [0.65]
        b = [0.60]
        result = paired_t_test(a, b)
        assert result["can_compute"] is False  # Need ≥2 samples

    def test_zero_variance_cohens_d(self):
        from feature2.statistical.significance import cohen_d
        a = [0.65, 0.65, 0.65]
        b = [0.60, 0.60, 0.60]
        result = cohen_d(a, b)
        assert result["cohen_d"] is not None