"""
Tests for Feature 2 Phase 3: Transfer + Low-Data Learning.
"""

import pytest
import torch
import numpy as np
import os
import json
import sys
from pathlib import Path
from torch_geometric.data import Data, Batch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────
# Shared Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def mock_checkpoint(tmp_path):
    from feature2.models.pretrained_encoder import create_mock_checkpoint
    path = str(tmp_path / "mock.pt")
    create_mock_checkpoint(path, "task_conditioned",
                           hidden_dim=32, n_layers=2, task_dim=16, num_tasks=17)
    return path


@pytest.fixture
def tiny_dataset():
    """Tiny synthetic dataset (no SMILES needed)."""
    class TinyDS:
        def __init__(self, n=20, num_tasks=5):
            self.data_list = [
                Data(
                    x=torch.randn(8, 129),
                    edge_index=torch.randint(0, 8, (2, 16)),
                    edge_attr=torch.randn(16, 6),
                    pos=torch.randn(8, 3),
                    y=torch.randint(0, 2, (num_tasks,)).float(),
                ) for _ in range(n)
            ]
            self.task_names = [f"task_{i}" for i in range(num_tasks)]
            self.num_tasks = num_tasks

        def __len__(self): return len(self.data_list)
        def __getitem__(self, i): return self.data_list[i]

    return TinyDS


@pytest.fixture
def base_config():
    return {
        "lr": 1e-3,
        "weight_decay": 0,
        "batch_size": 4,
        "epochs": 2,
        "patience": 5,
        "grad_clip": 1.0,
        "hidden_dim": 32,
        "num_unfreeze_layers": 1,
    }


# ─────────────────────────────────────────────
# Test 1: Low-Data Trainer
# ─────────────────────────────────────────────

class TestLowDataTrainer:

    def test_trainer_subsamples_correctly(self, tiny_dataset, base_config, tmp_path):
        from feature2.training.low_data_trainer import LowDataTrainer
        from feature2.models.transfer_heads import ScratchClassifier

        ds = tiny_dataset(n=40, num_tasks=3)
        model = ScratchClassifier(
            node_dim=129, edge_dim=6,
            hidden_dim=32, n_layers=2, num_tasks=3
        )

        trainer = LowDataTrainer(
            model=model,
            train_dataset=ds,
            val_dataset=tiny_dataset(n=10, num_tasks=3)(),
            test_dataset=tiny_dataset(n=10, num_tasks=3)(),
            task_names=["t0", "t1", "t2"],
            fraction=0.5,
            seed=42,
            config=base_config,
            result_dir=str(tmp_path),
            device="cpu",
            verbose=False,
        )

        # Training set should be subsampled
        n_expected = max(32, int(0.5 * 40))
        assert len(trainer.train_dataset) == n_expected

    def test_trainer_runs_and_returns_results(
        self, tiny_dataset, base_config, tmp_path
    ):
        from feature2.training.low_data_trainer import LowDataTrainer
        from feature2.models.transfer_heads import ScratchClassifier

        ds_cls = tiny_dataset

        model = ScratchClassifier(
            node_dim=129, edge_dim=6,
            hidden_dim=32, n_layers=2, num_tasks=3
        )
        trainer = LowDataTrainer(
            model=model,
            train_dataset=ds_cls(n=40, num_tasks=3)(),
            val_dataset=ds_cls(n=10, num_tasks=3)(),
            test_dataset=ds_cls(n=10, num_tasks=3)(),
            task_names=["t0", "t1", "t2"],
            fraction=1.0,
            seed=0,
            config=base_config,
            result_dir=str(tmp_path),
            device="cpu",
            verbose=False,
        )
        results = trainer.train(experiment_name="test")
        assert "test_auc" in results
        assert "val_auc" in results
        assert "fraction" in results
        assert 0.0 <= results["test_auc"] <= 1.0

    def test_fraction_10pct(self, tiny_dataset, base_config, tmp_path):
        from feature2.training.low_data_trainer import LowDataTrainer
        from feature2.models.transfer_heads import ScratchClassifier

        ds_cls = tiny_dataset
        model = ScratchClassifier(
            node_dim=129, edge_dim=6,
            hidden_dim=32, n_layers=2, num_tasks=2
        )
        trainer = LowDataTrainer(
            model=model,
            train_dataset=ds_cls(n=50, num_tasks=2)(),
            val_dataset=ds_cls(n=10, num_tasks=2)(),
            test_dataset=ds_cls(n=10, num_tasks=2)(),
            task_names=["t0", "t1"],
            fraction=0.10,
            seed=0,
            config=base_config,
            result_dir=str(tmp_path),
            device="cpu",
            verbose=False,
        )
        results = trainer.train("frac10")
        assert results["fraction"] == pytest.approx(0.10)
        assert results["n_train"] >= 1

    def test_result_json_saved(self, tiny_dataset, base_config, tmp_path):
        from feature2.training.low_data_trainer import LowDataTrainer
        from feature2.models.transfer_heads import ScratchClassifier

        ds_cls = tiny_dataset
        model = ScratchClassifier(
            node_dim=129, edge_dim=6,
            hidden_dim=32, n_layers=2, num_tasks=2
        )
        trainer = LowDataTrainer(
            model=model,
            train_dataset=ds_cls(n=40, num_tasks=2)(),
            val_dataset=ds_cls(n=10, num_tasks=2)(),
            test_dataset=ds_cls(n=10, num_tasks=2)(),
            task_names=["t0", "t1"],
            fraction=0.5,
            seed=0,
            config=base_config,
            result_dir=str(tmp_path),
            device="cpu",
            verbose=False,
        )
        trainer.train("json_test")
        saved = list(tmp_path.glob("*json_test*results.json"))
        assert len(saved) == 1


# ─────────────────────────────────────────────
# Test 2: Transfer Comparison
# ─────────────────────────────────────────────

class TestTransferComparison:

    def test_compute_transfer_gain_positive(self):
        from feature2.evaluation.transfer_comparison import (
            compute_transfer_gain_table
        )
        comparison = {
            "unaligned": {
                "sider": {"linear_probe": {"test_auc_mean": 0.65, "test_auc_std": 0.01, "n_seeds": 3, "per_seed": [0.65]}},
            },
            "contrastive": {
                "sider": {"linear_probe": {"test_auc_mean": 0.70, "test_auc_std": 0.01, "n_seeds": 3, "per_seed": [0.70]}},
            },
        }
        gain_table = compute_transfer_gain_table(comparison)
        assert "contrastive" in gain_table
        g = gain_table["contrastive"]["sider"]["linear_probe"]
        assert g["absolute_gain"] == pytest.approx(0.05, abs=1e-4)
        assert g["is_positive"] is True

    def test_compute_transfer_gain_negative(self):
        from feature2.evaluation.transfer_comparison import (
            compute_transfer_gain_table
        )
        comparison = {
            "unaligned": {
                "muv": {"linear_probe": {"test_auc_mean": 0.70, "test_auc_std": 0.01, "n_seeds": 3, "per_seed": [0.70]}},
            },
            "domain": {
                "muv": {"linear_probe": {"test_auc_mean": 0.65, "test_auc_std": 0.01, "n_seeds": 3, "per_seed": [0.65]}},
            },
        }
        gain_table = compute_transfer_gain_table(comparison)
        g = gain_table["domain"]["muv"]["linear_probe"]
        assert g["is_positive"] is False

    def test_resolve_checkpoint_fallback(self, tmp_path, mock_checkpoint):
        from feature2.evaluation.transfer_comparison import resolve_checkpoint

        # Direct path works
        result = resolve_checkpoint(mock_checkpoint, seed=0)
        assert result is not None

    def test_evaluate_encoder_on_transfer_mock(
        self, mock_checkpoint, tiny_dataset, tmp_path, base_config
    ):
        """Test transfer evaluation with mock checkpoint and synthetic dataset."""
        from feature2.evaluation.transfer_comparison import (
            evaluate_encoder_on_transfer
        )
        from unittest.mock import patch

        ds_cls = tiny_dataset

        # Patch create_transfer_datasets to return tiny datasets
        with patch(
            "feature2.evaluation.transfer_comparison.create_transfer_datasets"
        ) as mock_create, patch(
            "feature2.evaluation.transfer_comparison.get_transfer_task_names"
        ) as mock_names:
            mock_create.return_value = (
                ds_cls(n=20, num_tasks=3)(),
                ds_cls(n=10, num_tasks=3)(),
                ds_cls(n=10, num_tasks=3)(),
            )
            mock_names.return_value = ["t0", "t1", "t2"]

            result = evaluate_encoder_on_transfer(
                checkpoint_path=mock_checkpoint,
                alignment_name="test",
                dataset_name="sider",
                transfer_strategy="linear_probe",
                model_type="task_conditioned",
                seed=0,
                config={**base_config, "epochs": 2},
                data_dir=str(tmp_path),
                result_dir=str(tmp_path),
                device="cpu",
                verbose=False,
            )

        assert "test_auc" in result
        assert 0.0 <= result["test_auc"] <= 1.0


# ─────────────────────────────────────────────
# Test 3: Low-Data Curves
# ─────────────────────────────────────────────

class TestLowDataCurves:

    def test_plot_low_data_curves(self, tmp_path):
        from feature2.evaluation.low_data_curves import plot_low_data_curves

        curve_data = {
            "scratch": {
                0.10: {"mean": 0.60, "std": 0.02, "n": 3},
                0.50: {"mean": 0.68, "std": 0.01, "n": 3},
                1.00: {"mean": 0.72, "std": 0.01, "n": 3},
            },
            "linear_probe": {
                0.10: {"mean": 0.65, "std": 0.02, "n": 3},
                0.50: {"mean": 0.71, "std": 0.01, "n": 3},
                1.00: {"mean": 0.75, "std": 0.01, "n": 3},
            },
        }
        save_path = str(tmp_path / "test_curves.png")
        plot_low_data_curves(curve_data, "sider", save_path)
        assert os.path.exists(save_path)

    def test_degradation_table_runs(self, capsys):
        from feature2.evaluation.low_data_curves import print_degradation_table

        curve_data = {
            "scratch": {
                1.0: {"mean": 0.72, "std": 0.01, "n": 3},
                0.5: {"mean": 0.68, "std": 0.01, "n": 3},
                0.1: {"mean": 0.60, "std": 0.02, "n": 3},
            },
        }
        print_degradation_table(curve_data, "sider")
        captured = capsys.readouterr()
        assert "scratch" in captured.out
        assert "sider" in captured.out.upper()

    def test_data_efficiency_score(self):
        from feature2.evaluation.low_data_curves import compute_data_efficiency_score

        curve_data = {
            "scratch": {
                0.10: {"mean": 0.60},
                0.50: {"mean": 0.68},
                1.00: {"mean": 0.72},
            },
            "linear_probe": {
                0.10: {"mean": 0.65},
                0.50: {"mean": 0.71},
                1.00: {"mean": 0.75},
            },
        }
        scores = compute_data_efficiency_score(curve_data)
        assert "scratch" in scores
        assert "linear_probe" in scores
        # Linear probe should have higher efficiency
        assert scores["linear_probe"] >= scores["scratch"]

    def test_aggregate_low_data_results(self, tmp_path):
        from feature2.evaluation.low_data_curves import aggregate_low_data_results

        # Write dummy result files
        for frac in [0.5, 1.0]:
            r = {
                "experiment_name": f"lowdata_sider_none_scratch_frac{int(frac*100)}_seed0",
                "strategy": "scratch",
                "fraction": frac,
                "test_auc": 0.65 + frac * 0.05,
            }
            with open(tmp_path / f"lowdata_sider_scratch_{frac}_results.json", "w") as f:
                json.dump(r, f)

        result = aggregate_low_data_results(str(tmp_path), "sider")
        assert "scratch" in result
        assert len(result["scratch"]) == 2


# ─────────────────────────────────────────────
# Test 4: End-to-End Integration
# ─────────────────────────────────────────────

class TestEndToEndPhase3:

    def test_full_low_data_pipeline_scratch(
        self, tiny_dataset, base_config, tmp_path
    ):
        """
        Full pipeline: scratch model, 2 fractions, 1 seed.
        Verifies the entire training + evaluation loop works.
        """
        from feature2.training.low_data_trainer import LowDataTrainer
        from feature2.models.transfer_heads import ScratchClassifier
        from feature2.evaluation.low_data_curves import compute_data_efficiency_score

        ds_cls = tiny_dataset
        results_by_fraction = {}

        for fraction in [0.5, 1.0]:
            model = ScratchClassifier(
                node_dim=129, edge_dim=6,
                hidden_dim=32, n_layers=2, num_tasks=2
            )
            trainer = LowDataTrainer(
                model=model,
                train_dataset=ds_cls(n=40, num_tasks=2)(),
                val_dataset=ds_cls(n=10, num_tasks=2)(),
                test_dataset=ds_cls(n=10, num_tasks=2)(),
                task_names=["t0", "t1"],
                fraction=fraction,
                seed=0,
                config=base_config,
                result_dir=str(tmp_path),
                device="cpu",
                verbose=False,
            )
            r = trainer.train(f"e2e_{fraction}")
            results_by_fraction[fraction] = {
                "mean": r["test_auc"],
                "std": 0.0,
                "n": 1,
            }

        eff = compute_data_efficiency_score(
            {"scratch": results_by_fraction}
        )
        assert "scratch" in eff
        assert eff["scratch"] > 0

    def test_full_transfer_with_linear_probe(
        self, mock_checkpoint, tiny_dataset, base_config, tmp_path
    ):
        """
        Linear probe on frozen encoder across 2 fractions.
        """
        from feature2.models.pretrained_encoder import (
            load_feature1_checkpoint, FrozenEncoder
        )
        from feature2.models.transfer_heads import LinearProbeClassifier
        from feature2.training.low_data_trainer import LowDataTrainer

        ds_cls = tiny_dataset
        full_model = load_feature1_checkpoint(
            mock_checkpoint, "task_conditioned", "cpu", verbose=False
        )
        encoder = FrozenEncoder(full_model, "task_conditioned")

        for fraction in [0.5, 1.0]:
            model = LinearProbeClassifier(encoder=encoder, num_tasks=2)
            trainer = LowDataTrainer(
                model=model,
                train_dataset=ds_cls(n=40, num_tasks=2)(),
                val_dataset=ds_cls(n=10, num_tasks=2)(),
                test_dataset=ds_cls(n=10, num_tasks=2)(),
                task_names=["t0", "t1"],
                fraction=fraction,
                seed=0,
                config=base_config,
                result_dir=str(tmp_path),
                device="cpu",
                verbose=False,
            )
            r = trainer.train(f"lp_{fraction}")
            assert "test_auc" in r