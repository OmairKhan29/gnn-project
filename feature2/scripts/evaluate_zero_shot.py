"""
Phase 1: Zero-Shot Transfer Evaluation.
Evaluates Feature 1 encoder on SIDER/MUV WITHOUT any fine-tuning.
This is the hardest transfer scenario and establishes the baseline ceiling.
"""

import sys
import os
import json
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from feature2.data.transfer_datasets import create_transfer_datasets
from feature2.models.pretrained_encoder import (
    load_feature1_checkpoint,
    FrozenEncoder,
    create_mock_checkpoint,
)
from feature2.models.transfer_heads import LinearProbeClassifier
from torch_geometric.loader import DataLoader as PyGDataLoader
from evaluation.metrics import compute_roc_auc


def evaluate_zero_shot(
    encoder: FrozenEncoder,
    test_dataset,
    task_names,
    device: str = "cpu",
    batch_size: int = 32,
):
    """
    Evaluate encoder on test set without any head training.
    Uses random linear head (zero-shot).
    """
    model = LinearProbeClassifier(
        encoder=encoder,
        num_tasks=len(task_names),
        dropout=0.0,
    ).to(device)
    model.eval()

    loader = PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probs = model.predict(batch)
            labels = batch.y
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            if probs.shape != labels.shape:
                labels = labels.view(probs.shape)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    per_task_auc = {}
    valid_aucs = []

    for t, name in enumerate(task_names):
        task_labels = all_labels[:, t]
        task_probs = all_probs[:, t]
        mask = task_labels != -1
        if mask.sum() < 10:
            continue
        tl = task_labels[mask]
        tp = task_probs[mask]
        if len(np.unique(tl)) < 2:
            continue
        auc = compute_roc_auc(tl, tp)
        if not np.isnan(auc):
            per_task_auc[name] = float(auc)
            valid_aucs.append(auc)

    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.5
    return mean_auc, per_task_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/multitask_task_conditioned_best.pt")
    parser.add_argument("--model_type", type=str, default="task_conditioned",
                        choices=["task_conditioned", "hard_sharing", "single_task"])
    parser.add_argument("--datasets", nargs="+", default=["sider", "muv"])
    parser.add_argument("--data_dir", type=str, default="data/transfer")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use_mock", action="store_true",
                        help="Use mock checkpoint for testing (no trained model needed)")
    args = parser.parse_args()

    os.makedirs("results/feature2/transfer_baselines", exist_ok=True)

    print("=" * 60)
    print("Phase 1: Zero-Shot Transfer Evaluation")
    print("=" * 60)

    # Load or create mock checkpoint
    if args.use_mock or not os.path.exists(args.checkpoint):
        print("[WARNING] No checkpoint found. Using mock (random) weights.")
        mock_path = "checkpoints/feature2/mock_checkpoint.pt"
        os.makedirs("checkpoints/feature2", exist_ok=True)
        create_mock_checkpoint(mock_path, args.model_type)
        checkpoint_path = mock_path
    else:
        checkpoint_path = args.checkpoint

    # Load encoder
    full_model = load_feature1_checkpoint(
        checkpoint_path, args.model_type, args.device
    )
    encoder = FrozenEncoder(full_model, args.model_type)

    all_results = {}

    for dataset_name in args.datasets:
        print(f"\n[Evaluating zero-shot on {dataset_name.upper()}]")
        try:
            _, _, test_dataset = create_transfer_datasets(
                dataset_name, args.data_dir, verbose=True
            )

            from feature2.data.transfer_datasets import get_transfer_task_names
            task_names = get_transfer_task_names(dataset_name)

            mean_auc, per_task_auc = evaluate_zero_shot(
                encoder, test_dataset, task_names, args.device
            )

            print(f"  Mean ROC-AUC (zero-shot): {mean_auc:.4f}")
            print(f"  Valid tasks evaluated: {len(per_task_auc)}")

            all_results[dataset_name] = {
                "experiment_name": f"zero_shot_{dataset_name}",
                "strategy": "zero_shot",
                "dataset": dataset_name,
                "test_auc": mean_auc,
                "per_task_auc": per_task_auc,
                "num_tasks": len(task_names),
            }

        except Exception as e:
            print(f"  Failed: {e}")
            all_results[dataset_name] = {"error": str(e)}

    # Save results
    out_path = "results/feature2/transfer_baselines/zero_shot_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("Zero-Shot Transfer Summary")
    print("=" * 60)
    print(f"{'Dataset':<15} {'Mean AUC':>10}")
    print("-" * 30)
    for name, r in all_results.items():
        auc = r.get("test_auc", 0.0)
        print(f"{name:<15} {auc:>10.4f}")


if __name__ == "__main__":
    main()