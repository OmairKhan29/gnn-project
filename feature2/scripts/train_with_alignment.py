"""
Phase 2 main training script: Multi-task training with alignment.

Trains Feature 1 multi-task model with one of:
    - none (baseline)
    - contrastive
    - domain
    - prototype
"""

import sys
import os
import json
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.multitask_dataset import MultiTaskDataset, get_all_task_names, get_num_tasks
from models.task_conditioned_egnn import MultiTaskClassifier, HardSharingClassifier
from feature2.alignment.alignment_trainer import (
    AlignmentTrainer, create_alignment_module
)
from training.trainer import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment", type=str, default="none",
                        choices=["none", "contrastive", "domain", "prototype"])
    parser.add_argument("--model_type", type=str, default="task_conditioned",
                        choices=["task_conditioned", "hard_sharing"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--alignment_weight", type=float, default=0.5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--task_dim", type=int, default=64)
    parser.add_argument("--data_dir", type=str, default="data/processed")
    args = parser.parse_args()

    set_seed(args.seed)

    config = {
        "lr": args.lr,
        "weight_decay": 1e-5,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": 25,
        "grad_clip": 1.0,
        "alignment_weight": args.alignment_weight,
        "grl_max_lambda": 1.0,
        "warmup_fraction": 0.3,
    }

    print("=" * 60)
    print(f"Phase 2: Alignment Training [{args.alignment}]")
    print("=" * 60)

    # ── Load multi-task dataset ───────────────────────
    print("\n[1/4] Loading multi-task dataset...")
    train_ds = MultiTaskDataset(
        split="train",
        data_dir=args.data_dir,
        datasets=["tox21", "clintox", "bbbp", "bace", "hiv"],
    )
    val_ds = MultiTaskDataset(
        split="valid",
        data_dir=args.data_dir,
        datasets=["tox21", "clintox", "bbbp", "bace", "hiv"],
    )
    test_ds = MultiTaskDataset(
        split="test",
        data_dir=args.data_dir,
        datasets=["tox21", "clintox", "bbbp", "bace", "hiv"],
    )

    task_names = get_all_task_names()
    num_tasks = len(task_names)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  Num tasks: {num_tasks}")

    # ── Extract SMILES (for contrastive) ──────────────
    smiles_list = []
    if args.alignment == "contrastive":
        print("\n[2/4] Extracting SMILES from training set...")
        for i in range(len(train_ds)):
            d = train_ds[i]
            if hasattr(d, "smiles"):
                smiles_list.append(d.smiles)
            else:
                smiles_list.append("")
        print(f"  Extracted {len(smiles_list)} SMILES")
    else:
        smiles_list = None

    # ── Build task model ──────────────────────────────
    print("\n[3/4] Building model...")
    if args.model_type == "task_conditioned":
        task_model = MultiTaskClassifier(
            node_dim=129,
            edge_dim=6,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            num_tasks=num_tasks,
            task_dim=args.task_dim,
        )
    else:
        task_model = HardSharingClassifier(
            node_dim=129,
            edge_dim=6,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            num_tasks=num_tasks,
        )

    # Build alignment module
    alignment_module = create_alignment_module(
        strategy=args.alignment,
        embedding_dim=args.hidden_dim,
        num_tasks=num_tasks,
        num_domains=5,  # tox21, clintox, bbbp, bace, hiv
    )

    # ── Train ──────────────────────────────────────────
    print(f"\n[4/4] Training with alignment={args.alignment}...")
    exp_name = f"{args.model_type}_align_{args.alignment}_seed{args.seed}"

    trainer = AlignmentTrainer(
        task_model=task_model,
        alignment_module=alignment_module,
        alignment_strategy=args.alignment,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        task_names=task_names,
        smiles_list=smiles_list,
        config=config,
        device=args.device,
        verbose=True,
    )

    results = trainer.train(experiment_name=exp_name)

    print("\n" + "=" * 60)
    print(f"Strategy: {args.alignment}")
    print(f"Test AUC: {results['test_auc']:.4f}")
    print(f"Val AUC:  {results['val_auc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()