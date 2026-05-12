"""
scripts/analyze_pcgrad.py
Post-training PCGrad analysis script.

Run AFTER training is complete to generate all paper analysis figures.

Usage:
    python scripts/analyze_pcgrad.py \
        --ckpt checkpoints/ablation_strategy_task_conditioned_pcgrad_seed0.pt \
        --datasets bbbp bace hiv clintox tox21 \
        --device cuda
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, ".")

from data.multitask_dataset import (
    MultiTaskDataset,
    get_all_task_names,
    get_num_tasks,
)
from models.task_conditioned_egnn import MultiTaskClassifier
from evaluation.pcgrad_analysis import (
    run_full_pcgrad_analysis,
    plot_conflict_vs_auc_improvement,
    plot_conflict_ratio_over_training,
)
from torch_geometric.loader import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/ablation_strategy_task_conditioned_pcgrad_seed0.pt",
        help="Path to trained PCGrad model checkpoint",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["bbbp", "bace", "hiv", "clintox", "tox21"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pcgrad_analysis",
    )
    return parser.parse_args()


def load_model(ckpt_path: str, device: torch.device) -> MultiTaskClassifier:
    """Load trained model from checkpoint."""
    num_tasks = get_num_tasks()

    model = MultiTaskClassifier(
        node_dim=129,
        edge_dim=6,
        hidden_dim=128,
        num_layers=4,
        num_tasks=num_tasks,
        task_dim=64,
        dropout=0.0,  # No dropout during analysis
    )

    state = torch.load(ckpt_path, map_location=device)

    # Handle both full checkpoint and bare state_dict
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    model = model.to(device)
    model.eval()

    print(f"Loaded model from: {ckpt_path}")
    return model


def load_training_stats(ckpt_path: str) -> list:
    """Extract PCGrad stats history from checkpoint."""
    state = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state, dict) and "history" in state:
        return state["history"].get("pcgrad_stats", [])

    return []


def compute_per_task_conflict_scores(
    cosine_matrix,
    labels,
) -> dict:
    """
    For each task, compute average conflict with all other tasks.
    Higher score = more conflicting task.
    """
    n = len(labels)
    scores = {}

    for i, label in enumerate(labels):
        # Count negative cosines (conflicts) with other tasks
        conflict_count = sum(
            1 for j in range(n)
            if i != j and cosine_matrix[i, j] < 0
        )
        avg_conflict = conflict_count / max(n - 1, 1)
        scores[label] = avg_conflict

    return scores


def main():
    args = parse_args()

    print("=" * 80)
    print("PCGRAD POST-TRAINING ANALYSIS")
    print("=" * 80)

    # Device setup
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    if not os.path.exists(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}")
        print("Available checkpoints:")
        for f in os.listdir("checkpoints"):
            print(f"  checkpoints/{f}")
        return

    model = load_model(args.ckpt, device)

    # ── Load training data ────────────────────────────────────────────────
    print("\nLoading training data for gradient analysis...")
    train_ds = MultiTaskDataset(args.datasets, "train")
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    task_names = get_all_task_names()

    # ── Run full gradient analysis ─────────────────────────────────────────
    print(f"\nRunning gradient analysis ({args.num_batches} batches)...")
    stats, cosine_before, cosine_after, labels = run_full_pcgrad_analysis(
        model=model,
        loader=loader,
        device=device,
        task_names=task_names,
        output_dir=args.output_dir,
        num_batches=args.num_batches,
    )

    # ── Training-time conflict curve ───────────────────────────────────────
    print("\nExtracting training-time conflict statistics...")
    pcgrad_history = load_training_stats(args.ckpt)

    if pcgrad_history:
        plot_conflict_ratio_over_training(
            pcgrad_history,
            save_path=os.path.join(args.output_dir, "conflict_over_training.png"),
        )
        print(f"  Epochs with PCGrad stats: {len(pcgrad_history)}")
    else:
        print("  No training stats found in checkpoint.")

    # ── Conflict vs AUC improvement ────────────────────────────────────────
    print("\nComputing conflict vs AUC improvement correlation...")

    # Load ablation strategy results
    ablation_path = "results/ablations/ablation_strategy.json"

    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            ablation_data = json.load(f)

        # Get per-task AUC for hard_sharing and task_conditioned_pcgrad
        baseline_aucs = {}
        pcgrad_aucs = {}

        for strategy, results_list in [
            ("hard_sharing", baseline_aucs),
            ("task_conditioned_pcgrad", pcgrad_aucs),
        ]:
            if strategy in ablation_data["results"]:
                per_seed = ablation_data["results"][strategy]["per_seed"]
                # Average across seeds
                all_tasks = set()
                for seed_r in per_seed:
                    all_tasks.update(seed_r["test_auc_per_task"].keys())

                for task in all_tasks:
                    vals = [
                        s["test_auc_per_task"][task]
                        for s in per_seed
                        if task in s["test_auc_per_task"]
                    ]
                    results_list[task] = float(sum(vals) / len(vals)) if vals else 0.0

        # Compute AUC deltas
        auc_deltas = {
            task: pcgrad_aucs[task] - baseline_aucs[task]
            for task in set(baseline_aucs.keys()) & set(pcgrad_aucs.keys())
        }

        # Get conflict scores per task (map short label → full task name)
        per_task_conflict = compute_per_task_conflict_scores(cosine_before, labels)

        # Align naming (labels are short names, auc_deltas use full names)
        conflict_aligned = {}
        for full_task, delta in auc_deltas.items():
            short = "_".join(full_task.split("_")[1:])
            if short in per_task_conflict:
                conflict_aligned[full_task] = per_task_conflict[short]

        if conflict_aligned and auc_deltas:
            plot_conflict_vs_auc_improvement(
                conflict_aligned,
                auc_deltas,
                save_path=os.path.join(args.output_dir, "conflict_vs_auc.png"),
            )

        # Save delta summary
        delta_summary = {
            "auc_deltas_pcgrad_vs_baseline": auc_deltas,
            "mean_delta": float(sum(auc_deltas.values()) / len(auc_deltas)),
            "tasks_improved": int(sum(1 for d in auc_deltas.values() if d > 0)),
            "tasks_total": len(auc_deltas),
        }

        with open(os.path.join(args.output_dir, "auc_delta_summary.json"), "w") as f:
            json.dump(delta_summary, f, indent=2)

        print(f"  Mean Δ AUC (PCGrad vs baseline): {delta_summary['mean_delta']:+.4f}")
        print(f"  Tasks improved: {delta_summary['tasks_improved']}/{delta_summary['tasks_total']}")

    else:
        print(f"  Ablation results not found: {ablation_path}")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nKey findings:")
    print(f"  Conflict ratio before PCGrad: {stats['conflict_ratio_before']:.1%}")
    print(f"  Conflict ratio after PCGrad:  {stats['conflict_ratio_after']:.1%}")
    print(f"  Conflict reduction:           {stats['conflict_reduction_pct']:.1f}%")
    print(f"  Mean cosine similarity shift: {stats['avg_cosine_before']:.4f} → {stats['avg_cosine_after']:.4f}")

    print(f"\nOutput files:")
    for f in sorted(os.listdir(args.output_dir)):
        size_kb = os.path.getsize(os.path.join(args.output_dir, f)) / 1024
        print(f"  {f} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()