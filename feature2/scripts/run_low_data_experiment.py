"""
Phase 3: Low-Data Experiment Runner.

Tests all strategies at fractions: 10%, 25%, 50%, 100%.
Uses both aligned and unaligned encoders.
"""

import sys
import os
import json
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from feature2.data.transfer_datasets import (
    create_transfer_datasets, get_transfer_task_names
)
from feature2.data.low_data_splits import LOW_DATA_FRACTIONS
from feature2.models.pretrained_encoder import (
    load_feature1_checkpoint, FrozenEncoder, create_mock_checkpoint
)
from feature2.models.transfer_heads import (
    LinearProbeClassifier, FineTuneClassifier, ScratchClassifier
)
from feature2.training.low_data_trainer import LowDataTrainer
from feature2.evaluation.transfer_comparison import resolve_checkpoint
from training.trainer import set_seed


# ─────────────────────────────────────────────
# Build Model for Strategy
# ─────────────────────────────────────────────

def build_model_for_strategy(
    strategy: str,
    alignment_name: str,
    num_tasks: int,
    seed: int,
    model_type: str,
    config: Dict,
    device: str,
):
    """Build the right model for a given (strategy, alignment) combination."""
    from typing import Dict

    if strategy == "scratch":
        model = ScratchClassifier(
            node_dim=129, edge_dim=6,
            hidden_dim=config.get("hidden_dim", 128),
            n_layers=4, num_tasks=num_tasks,
        )
        return model

    # Load encoder checkpoint
    ckpt = resolve_checkpoint(alignment_name, seed, model_type)
    if ckpt is None:
        mock_path = f"checkpoints/feature2/mock_{model_type}_seed{seed}.pt"
        os.makedirs("checkpoints/feature2", exist_ok=True)
        create_mock_checkpoint(mock_path, model_type)
        ckpt = mock_path

    full_model = load_feature1_checkpoint(ckpt, model_type, device, verbose=False)
    encoder = FrozenEncoder(full_model, model_type)

    if strategy == "linear_probe":
        return LinearProbeClassifier(encoder=encoder, num_tasks=num_tasks)
    elif strategy in ["top_layers", "full"]:
        return FineTuneClassifier(
            encoder=encoder,
            num_tasks=num_tasks,
            strategy=strategy,
            num_unfreeze_layers=config.get("num_unfreeze_layers", 2),
            hidden_dim=config.get("hidden_dim", 128),
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignments", nargs="+",
                        default=["unaligned", "contrastive", "prototype"])
    parser.add_argument("--strategies", nargs="+",
                        default=["scratch", "linear_probe", "top_layers"])
    parser.add_argument("--datasets", nargs="+", default=["sider", "muv"])
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=[0.10, 0.25, 0.50, 1.00])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--model_type", type=str, default="task_conditioned")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_dir", type=str, default="data/transfer")
    args = parser.parse_args()

    config = {
        "lr": args.lr,
        "weight_decay": 1e-5,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": 20,
        "grad_clip": 1.0,
        "hidden_dim": 128,
        "num_unfreeze_layers": 2,
    }

    result_dir = "results/feature2/low_data"
    os.makedirs(result_dir, exist_ok=True)

    total_runs = (len(args.alignments) * len(args.strategies) *
                  len(args.datasets) * len(args.fractions) * len(args.seeds))

    print("=" * 70)
    print(f"Phase 3: Low-Data Experiment ({total_runs} total runs)")
    print("=" * 70)
    print(f"  Alignments: {args.alignments}")
    print(f"  Strategies: {args.strategies}")
    print(f"  Datasets:   {args.datasets}")
    print(f"  Fractions:  {args.fractions}")
    print(f"  Seeds:      {args.seeds}")

    all_results = []
    done = 0

    for dataset_name in args.datasets:
        # Load full datasets once
        train_ds_full, val_ds, test_ds = create_transfer_datasets(
            dataset_name, args.data_dir, verbose=True
        )
        task_names = get_transfer_task_names(dataset_name)
        num_tasks = len(task_names)

        for alignment_name in args.alignments:
            for strategy in args.strategies:

                # Skip invalid combos
                # (scratch doesn't depend on alignment — run once only)
                if strategy == "scratch" and alignment_name != args.alignments[0]:
                    continue

                for fraction in args.fractions:
                    for seed in args.seeds:
                        done += 1
                        exp_name = (f"lowdata_{dataset_name}_{alignment_name}"
                                    f"_{strategy}_frac{int(fraction*100)}_seed{seed}")

                        print(f"\n  [{done}/{total_runs}] {exp_name}")

                        set_seed(seed)
                        try:
                            model = build_model_for_strategy(
                                strategy=strategy,
                                alignment_name=alignment_name,
                                num_tasks=num_tasks,
                                seed=seed,
                                model_type=args.model_type,
                                config=config,
                                device=args.device,
                            )

                            trainer = LowDataTrainer(
                                model=model,
                                train_dataset=train_ds_full,
                                val_dataset=val_ds,
                                test_dataset=test_ds,
                                task_names=task_names,
                                fraction=fraction,
                                seed=seed,
                                config=config,
                                result_dir=result_dir,
                                device=args.device,
                                verbose=True,
                            )

                            result = trainer.train(experiment_name=exp_name)
                            result["dataset"] = dataset_name
                            result["alignment"] = alignment_name
                            result["strategy"] = strategy
                            all_results.append(result)

                        except Exception as e:
                            print(f"    FAILED: {e}")
                            all_results.append({
                                "experiment_name": exp_name,
                                "dataset": dataset_name,
                                "alignment": alignment_name,
                                "strategy": strategy,
                                "fraction": fraction,
                                "seed": seed,
                                "test_auc": 0.5,
                                "error": str(e),
                            })

    # Save all results
    all_path = os.path.join(result_dir, "all_low_data_results.json")
    save_list = []
    for r in all_results:
        save_r = {k: v for k, v in r.items()
                  if k not in ["val_per_task_auc", "test_per_task_auc"]}
        save_list.append(save_r)

    with open(all_path, "w") as f:
        json.dump(save_list, f, indent=2)
    print(f"\nAll results saved to {all_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Low-Data Summary (mean test AUC)")
    print("=" * 70)

    for dataset_name in args.datasets:
        print(f"\n  {dataset_name.upper()}")
        print(f"  {'Strategy':<25} "
              f"{'10%':>8} {'25%':>8} {'50%':>8} {'100%':>8}")
        print(f"  {'-'*60}")

        for strategy in args.strategies:
            for alignment in args.alignments:
                label = (f"{alignment}/{strategy}"
                         if strategy != "scratch" else "scratch")

                row = f"  {label:<25}"
                for frac in [0.10, 0.25, 0.50, 1.00]:
                    matching = [
                        r["test_auc"] for r in all_results
                        if (r.get("dataset") == dataset_name and
                            r.get("strategy") == strategy and
                            r.get("alignment") == alignment and
                            abs(r.get("fraction", -1) - frac) < 0.01 and
                            "error" not in r)
                    ]
                    if matching:
                        row += f" {np.mean(matching):>8.4f}"
                    else:
                        row += f" {'N/A':>8}"
                print(row)


if __name__ == "__main__":
    main()