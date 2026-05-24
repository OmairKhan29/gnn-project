"""
Low-Data Split Generator for Feature 2.
Simulates limited data scenarios by subsampling training splits.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.featurizer import Molecule3DFeaturizer


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# Fractions to simulate low-data scenarios
LOW_DATA_FRACTIONS = [1.0, 0.5, 0.25, 0.10]

# Minimum molecules per fraction to ensure valid training
MIN_MOLECULES = 32


# ─────────────────────────────────────────────
# Low-Data Subsetting
# ─────────────────────────────────────────────

def create_low_data_subset(
    dataset: Dataset,
    fraction: float,
    seed: int = 42,
    stratify_by_task: int = 0,
) -> Dataset:
    """
    Create a subsampled version of a dataset.

    Args:
        dataset: Full training dataset
        fraction: Fraction of data to keep (0.0 - 1.0]
        seed: Random seed for reproducibility
        stratify_by_task: Task index to stratify sampling by (if applicable)

    Returns:
        Subsampled Dataset
    """
    assert 0.0 < fraction <= 1.0, f"Fraction must be in (0, 1], got {fraction}"

    n_total = len(dataset)
    n_keep = max(MIN_MOLECULES, int(n_total * fraction))
    n_keep = min(n_keep, n_total)  # Can't keep more than available

    rng = np.random.RandomState(seed)
    indices = rng.choice(n_total, size=n_keep, replace=False)
    indices = sorted(indices.tolist())

    return Subset(dataset, indices)


def create_all_low_data_splits(
    dataset: Dataset,
    fractions: List[float] = None,
    seed: int = 42,
) -> Dict[float, Dataset]:
    """
    Create multiple low-data splits for a dataset.

    Returns:
        Dict mapping fraction → subsampled Dataset
    """
    if fractions is None:
        fractions = LOW_DATA_FRACTIONS

    splits = {}
    for frac in fractions:
        splits[frac] = create_low_data_subset(dataset, frac, seed=seed)

    return splits


# ─────────────────────────────────────────────
# Low-Data Experiment Config
# ─────────────────────────────────────────────

class LowDataConfig:
    """
    Configuration for a low-data experiment run.
    Specifies which dataset, fraction, seed, and strategy.
    """

    def __init__(
        self,
        dataset_name: str,
        fraction: float,
        seed: int,
        strategy: str,             # "scratch", "linear_probe", "finetune"
        pretrained_checkpoint: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.fraction = fraction
        self.seed = seed
        self.strategy = strategy
        self.pretrained_checkpoint = pretrained_checkpoint

    def __repr__(self) -> str:
        return (
            f"LowDataConfig("
            f"dataset={self.dataset_name}, "
            f"frac={self.fraction}, "
            f"seed={self.seed}, "
            f"strategy={self.strategy})"
        )

    def to_dict(self) -> Dict:
        return {
            "dataset_name": self.dataset_name,
            "fraction": self.fraction,
            "seed": self.seed,
            "strategy": self.strategy,
            "pretrained_checkpoint": self.pretrained_checkpoint,
        }


# ─────────────────────────────────────────────
# Low-Data Experiment Grid
# ─────────────────────────────────────────────

def generate_low_data_experiment_grid(
    datasets: List[str],
    fractions: List[float] = None,
    seeds: List[int] = None,
    strategies: List[str] = None,
    pretrained_checkpoint: Optional[str] = None,
) -> List[LowDataConfig]:
    """
    Generate all (dataset, fraction, seed, strategy) combinations.

    Returns:
        List of LowDataConfig objects defining all experiments
    """
    if fractions is None:
        fractions = LOW_DATA_FRACTIONS
    if seeds is None:
        seeds = [0, 1, 2]
    if strategies is None:
        strategies = ["scratch", "linear_probe", "finetune"]

    configs = []
    for dataset in datasets:
        for fraction in fractions:
            for seed in seeds:
                for strategy in strategies:
                    configs.append(
                        LowDataConfig(
                            dataset_name=dataset,
                            fraction=fraction,
                            seed=seed,
                            strategy=strategy,
                            pretrained_checkpoint=pretrained_checkpoint,
                        )
                    )

    return configs


# ─────────────────────────────────────────────
# Reporting Utilities
# ─────────────────────────────────────────────

def summarize_low_data_splits(splits: Dict[float, Dataset]) -> str:
    """Print a summary table of low-data split sizes."""
    lines = ["Low-Data Split Summary", "=" * 40]
    lines.append(f"{'Fraction':>10} {'N Molecules':>12} {'% of Full':>10}")
    lines.append("-" * 40)

    full_size = None
    for frac in sorted(splits.keys(), reverse=True):
        n = len(splits[frac])
        if full_size is None:
            full_size = n
        pct = 100.0 * n / full_size if full_size > 0 else 0
        lines.append(f"{frac:>10.0%} {n:>12} {pct:>9.1f}%")

    return "\n".join(lines)