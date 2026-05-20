"""
Transfer Comparison: aligned vs unaligned encoders.

Given checkpoints from:
    - Phase 1 (unaligned Feature 1 encoder)
    - Phase 2 (aligned encoders: contrastive / domain / prototype)

Measures improvement in transfer performance on SIDER + MUV.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from feature2.models.pretrained_encoder import (
    load_feature1_checkpoint, FrozenEncoder, create_mock_checkpoint
)
from feature2.models.transfer_heads import (
    LinearProbeClassifier, FineTuneClassifier, ScratchClassifier
)
from feature2.training.transfer_trainer import TransferTrainer
from feature2.data.transfer_datasets import (
    create_transfer_datasets, get_transfer_task_names
)


# ─────────────────────────────────────────────
# Checkpoint Registry
# ─────────────────────────────────────────────

CHECKPOINT_REGISTRY = {
    "unaligned": "checkpoints/feature2/alignment/task_conditioned_align_none_seed{seed}_best.pt",
    "contrastive": "checkpoints/feature2/alignment/task_conditioned_align_contrastive_seed{seed}_best.pt",
    "domain": "checkpoints/feature2/alignment/task_conditioned_align_domain_seed{seed}_best.pt",
    "prototype": "checkpoints/feature2/alignment/task_conditioned_align_prototype_seed{seed}_best.pt",
}


def resolve_checkpoint(
    alignment_name: str,
    seed: int,
    model_type: str = "task_conditioned",
) -> Optional[str]:
    """Resolve checkpoint path for a given alignment strategy and seed."""
    if alignment_name in CHECKPOINT_REGISTRY:
        path = CHECKPOINT_REGISTRY[alignment_name].format(seed=seed)
    else:
        path = alignment_name  # Treat as direct path

    if os.path.exists(path):
        return path

    # Fallback: search common locations
    fallbacks = [
        f"checkpoints/{alignment_name}_best.pt",
        f"checkpoints/multitask_{alignment_name}_best.pt",
    ]
    for fb in fallbacks:
        if os.path.exists(fb):
            return fb

    return None


# ─────────────────────────────────────────────
# Single Transfer Evaluation
# ─────────────────────────────────────────────

def evaluate_encoder_on_transfer(
    checkpoint_path: Optional[str],
    alignment_name: str,
    dataset_name: str,
    transfer_strategy: str,     # "linear_probe" | "top_layers" | "full"
    model_type: str,
    seed: int,
    config: Dict,
    data_dir: str,
    result_dir: str,
    device: str,
    verbose: bool = False,
) -> Dict:
    """
    Load an encoder checkpoint and evaluate on a transfer dataset.

    Returns:
        result dict with test_auc, val_auc, transfer_strategy, alignment_name
    """
    from training.trainer import set_seed
    set_seed(seed)

    # Load datasets
    train_ds, val_ds, test_ds = create_transfer_datasets(
        dataset_name, data_dir, verbose=False
    )
    task_names = get_transfer_task_names(dataset_name)
    num_tasks = len(task_names)

    # Load encoder
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        # Use mock
        mock_path = f"checkpoints/feature2/mock_{model_type}_seed{seed}.pt"
        os.makedirs("checkpoints/feature2", exist_ok=True)
        create_mock_checkpoint(mock_path, model_type)
        checkpoint_path = mock_path
        if verbose:
            print(f"  [WARNING] Using mock checkpoint for {alignment_name}")

    full_model = load_feature1_checkpoint(
        checkpoint_path, model_type, device, verbose=False
    )
    encoder = FrozenEncoder(full_model, model_type)

    # Build transfer model
    if transfer_strategy == "linear_probe":
        model = LinearProbeClassifier(encoder=encoder, num_tasks=num_tasks)
    elif transfer_strategy in ["top_layers", "full"]:
        model = FineTuneClassifier(
            encoder=encoder,
            num_tasks=num_tasks,
            strategy=transfer_strategy,
            num_unfreeze_layers=config.get("num_unfreeze_layers", 2),
            hidden_dim=config.get("hidden_dim", 128),
        )
    else:
        raise ValueError(f"Unknown transfer_strategy: {transfer_strategy}")

    # Train
    exp_name = (f"transfer_{alignment_name}_{dataset_name}"
                f"_{transfer_strategy}_seed{seed}")

    trainer = TransferTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        task_names=task_names,
        config=config,
        result_dir=result_dir,
        device=device,
        verbose=verbose,
    )

    results = trainer.train(experiment_name=exp_name)
    results["alignment_name"] = alignment_name
    results["transfer_strategy"] = transfer_strategy
    results["dataset_name"] = dataset_name
    results["seed"] = seed
    return results


# ─────────────────────────────────────────────
# Full Comparison Runner
# ─────────────────────────────────────────────

class TransferComparisonRunner:
    """
    Runs all (alignment × dataset × transfer_strategy × seed) combinations.
    Aggregates and saves comparison table.
    """

    def __init__(
        self,
        alignment_names: List[str],
        datasets: List[str],
        transfer_strategies: List[str],
        seeds: List[int],
        model_type: str = "task_conditioned",
        config: Dict = None,
        data_dir: str = "data/transfer",
        result_dir: str = "results/feature2/transfer_comparison",
        device: str = "cpu",
        verbose: bool = True,
    ):
        self.alignment_names = alignment_names
        self.datasets = datasets
        self.transfer_strategies = transfer_strategies
        self.seeds = seeds
        self.model_type = model_type
        self.config = config or {
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": 32,
            "epochs": 80,
            "patience": 15,
            "grad_clip": 1.0,
            "hidden_dim": 128,
            "num_unfreeze_layers": 2,
        }
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.device = device
        self.verbose = verbose

        os.makedirs(result_dir, exist_ok=True)

    def run(self) -> Dict:
        """
        Execute all experiments.

        Returns:
            Nested dict: {alignment: {dataset: {strategy: aggregated_result}}}
        """
        all_results = {}
        total = (len(self.alignment_names) * len(self.datasets) *
                 len(self.transfer_strategies) * len(self.seeds))
        done = 0

        print(f"\n[TransferComparison] Running {total} experiments...")
        print(f"  Alignments: {self.alignment_names}")
        print(f"  Datasets: {self.datasets}")
        print(f"  Strategies: {self.transfer_strategies}")
        print(f"  Seeds: {self.seeds}")
        print()

        for alignment in self.alignment_names:
            all_results[alignment] = {}
            for dataset in self.datasets:
                all_results[alignment][dataset] = {}
                for strategy in self.transfer_strategies:
                    seed_results = []
                    for seed in self.seeds:
                        ckpt = resolve_checkpoint(alignment, seed, self.model_type)
                        done += 1
                        print(f"  [{done}/{total}] "
                              f"{alignment}|{dataset}|{strategy}|seed{seed}")

                        try:
                            r = evaluate_encoder_on_transfer(
                                checkpoint_path=ckpt,
                                alignment_name=alignment,
                                dataset_name=dataset,
                                transfer_strategy=strategy,
                                model_type=self.model_type,
                                seed=seed,
                                config=self.config,
                                data_dir=self.data_dir,
                                result_dir=self.result_dir,
                                device=self.device,
                                verbose=False,
                            )
                            seed_results.append(r)
                        except Exception as e:
                            print(f"    FAILED: {e}")
                            seed_results.append({"test_auc": 0.5, "val_auc": 0.5})

                    # Aggregate across seeds
                    test_aucs = [r.get("test_auc", 0.5) for r in seed_results]
                    all_results[alignment][dataset][strategy] = {
                        "test_auc_mean": float(np.mean(test_aucs)),
                        "test_auc_std": float(np.std(test_aucs)),
                        "n_seeds": len(test_aucs),
                        "per_seed": [r.get("test_auc", 0.5) for r in seed_results],
                    }

        # Save
        out_path = os.path.join(self.result_dir, "full_comparison.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[TransferComparison] Results saved to {out_path}")

        return all_results


# ─────────────────────────────────────────────
# Gain Analysis
# ─────────────────────────────────────────────

def compute_transfer_gain_table(comparison_results: Dict) -> Dict:
    """
    Compute transfer gain of aligned vs unaligned encoder.

    Args:
        comparison_results: output of TransferComparisonRunner.run()

    Returns:
        gain_table: {alignment: {dataset: {strategy: gain}}}
    """
    gain_table = {}
    baseline_key = "unaligned"

    for alignment, datasets in comparison_results.items():
        if alignment == baseline_key:
            continue
        gain_table[alignment] = {}
        for dataset, strategies in datasets.items():
            gain_table[alignment][dataset] = {}
            for strategy, result in strategies.items():
                baseline = (comparison_results
                            .get(baseline_key, {})
                            .get(dataset, {})
                            .get(strategy, {})
                            .get("test_auc_mean", 0.5))
                aligned = result["test_auc_mean"]
                gain_table[alignment][dataset][strategy] = {
                    "baseline_auc": float(baseline),
                    "aligned_auc": float(aligned),
                    "absolute_gain": float(aligned - baseline),
                    "relative_gain_pct": float(
                        (aligned - baseline) / max(baseline, 1e-8) * 100
                    ),
                    "is_positive": aligned > baseline,
                }

    return gain_table


def print_gain_table(gain_table: Dict):
    """Print formatted gain table."""
    print("\n" + "=" * 75)
    print("Transfer Gain: Aligned vs Unaligned Encoder")
    print("=" * 75)

    header = (f"{'Alignment':<14} {'Dataset':<8} {'Strategy':<14} "
              f"{'Baseline':>10} {'Aligned':>10} {'Δ AUC':>8} {'%':>8}")
    print(header)
    print("-" * 75)

    for alignment, datasets in gain_table.items():
        for dataset, strategies in datasets.items():
            for strategy, g in strategies.items():
                sign = "✅" if g["is_positive"] else "❌"
                print(
                    f"{alignment:<14} {dataset:<8} {strategy:<14} "
                    f"{g['baseline_auc']:>10.4f} "
                    f"{g['aligned_auc']:>10.4f} "
                    f"{g['absolute_gain']:>+8.4f} "
                    f"{g['relative_gain_pct']:>+7.1f}%  {sign}"
                )