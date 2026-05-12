"""
Phase 1: Linear Probe Training on Transfer Datasets.
Trains a linear classification head on frozen Feature 1 encoder.
"""

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from feature2.data.transfer_datasets import (
    create_transfer_datasets,
    get_transfer_task_names,
    get_num_transfer_tasks,
)
from feature2.models.pretrained_encoder import (
    load_feature1_checkpoint,
    FrozenEncoder,
    create_mock_checkpoint,
)
from feature2.models.transfer_heads import LinearProbeClassifier
from feature2.training.transfer_trainer import TransferTrainer
from training.trainer import set_seed


def run_linear_probe(
    dataset_name: str,
    checkpoint_path: str,
    model_type: str,
    seed: int,
    device: str,
    config: dict,
    data_dir: str,
    result_dir: str,
    use_mock: bool = False,
):
    set_seed(seed)

    # Load datasets
    train_ds, val_ds, test_ds = create_transfer_datasets(
        dataset_name, data_dir, verbose=False
    )
    task_names = get_transfer_task_names(dataset_name)
    num_tasks = get_num_transfer_tasks(dataset_name)

    # Load encoder
    if use_mock or not os.path.exists(checkpoint_path):
        mock_path = f"checkpoints/feature2/mock_{model_type}.pt"
        os.makedirs("checkpoints/feature2", exist_ok=True)
        create_mock_checkpoint(mock_path, model_type)
        checkpoint_path = mock_path

    full_model = load_feature1_checkpoint(checkpoint_path, model_type, device, verbose=False)
    encoder = FrozenEncoder(full_model, model_type)

    # Build linear probe
    model = LinearProbeClassifier(encoder=encoder, num_tasks=num_tasks)

    # Train
    exp_name = f"linear_probe_{dataset_name}_seed{seed}"
    trainer = TransferTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        task_names=task_names,
        config=config,
        checkpoint_dir="checkpoints/feature2",
        result_dir=result_dir,
        device=device,
        verbose=True,
    )

    results = trainer.train(experiment_name=exp_name)
    results["seed"] = seed
    results["strategy"] = "linear_probe"
    results["dataset"] = dataset_name

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/multitask_task_conditioned_best.pt")
    parser.add_argument("--model_type", type=str, default="task_conditioned")
    parser.add_argument("--datasets", nargs="+", default=["sider", "muv"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_dir", type=str, default="data/transfer")
    parser.add_argument("--use_mock", action="store_true")
    args = parser.parse_args()

    config = {
        "lr": args.lr,
        "weight_decay": 1e-5,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": 20,
        "grad_clip": 1.0,
    }

    result_dir = "results/feature2/transfer_baselines"
    os.makedirs(result_dir, exist_ok=True)

    all_results = {}

    for dataset_name in args.datasets:
        dataset_results = []
        print(f"\n{'='*60}")
        print(f"Linear Probe: {dataset_name.upper()}")
        print(f"{'='*60}")

        for seed in args.seeds:
            print(f"\n  Seed {seed}:")
            result = run_linear_probe(
                dataset_name=dataset_name,
                checkpoint_path=args.checkpoint,
                model_type=args.model_type,
                seed=seed,
                device=args.device,
                config=config,
                data_dir=args.data_dir,
                result_dir=result_dir,
                use_mock=args.use_mock,
            )
            dataset_results.append(result)

        # Aggregate
        import numpy as np
        test_aucs = [r["test_auc"] for r in dataset_results]
        all_results[dataset_name] = {
            "strategy": "linear_probe",
            "dataset": dataset_name,
            "mean_test_auc": float(np.mean(test_aucs)),
            "std_test_auc": float(np.std(test_aucs)),
            "seeds": args.seeds,
            "per_seed_results": dataset_results,
        }
        print(f"\n  {dataset_name}: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")

    # Save aggregated results
    agg_path = os.path.join(result_dir, "linear_probe_aggregated.json")
    with open(agg_path, "w") as f:
        save_data = {}
        for ds, r in all_results.items():
            save_r = {k: v for k, v in r.items() if k != "per_seed_results"}
            save_r["per_seed_aucs"] = [
                s["test_auc"] for s in r["per_seed_results"]
            ]
            save_data[ds] = save_r
        json.dump(save_data, f, indent=2)

    print(f"\nAggregated results saved to {agg_path}")


if __name__ == "__main__":
    main()