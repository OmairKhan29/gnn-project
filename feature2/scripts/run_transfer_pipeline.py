"""
Phase 3 Main Script: Full Transfer Pipeline.

Evaluates all alignment strategies on SIDER + MUV using:
    - linear_probe
    - top_layers fine-tuning
    - full fine-tuning

Compares aligned vs unaligned encoder transfer performance.
"""

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from feature2.evaluation.transfer_comparison import (
    TransferComparisonRunner,
    compute_transfer_gain_table,
    print_gain_table,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignments", nargs="+",
                        default=["unaligned", "contrastive", "domain", "prototype"])
    parser.add_argument("--datasets", nargs="+", default=["sider", "muv"])
    parser.add_argument("--strategies", nargs="+",
                        default=["linear_probe", "top_layers"])
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
        "patience": 15,
        "grad_clip": 1.0,
        "hidden_dim": 128,
        "num_unfreeze_layers": 2,
    }

    result_dir = "results/feature2/transfer_comparison"
    os.makedirs(result_dir, exist_ok=True)

    print("=" * 70)
    print("Phase 3: Cross-Dataset Transfer Pipeline")
    print("=" * 70)

    # Run all experiments
    runner = TransferComparisonRunner(
        alignment_names=args.alignments,
        datasets=args.datasets,
        transfer_strategies=args.strategies,
        seeds=args.seeds,
        model_type=args.model_type,
        config=config,
        data_dir=args.data_dir,
        result_dir=result_dir,
        device=args.device,
        verbose=True,
    )

    comparison = runner.run()

    # Compute and print gain table
    gain_table = compute_transfer_gain_table(comparison)
    print_gain_table(gain_table)

    # Save gain table
    gain_path = os.path.join(result_dir, "transfer_gain_table.json")
    with open(gain_path, "w") as f:
        json.dump(gain_table, f, indent=2)
    print(f"\nGain table saved to {gain_path}")

    # Summary
    print("\n" + "=" * 70)
    print("Best Alignment per Dataset")
    print("=" * 70)
    for dataset in args.datasets:
        best_alignment = None
        best_gain = -999
        for alignment, ds_data in gain_table.items():
            if dataset in ds_data:
                for strategy, g in ds_data[dataset].items():
                    if g["absolute_gain"] > best_gain:
                        best_gain = g["absolute_gain"]
                        best_alignment = f"{alignment}/{strategy}"
        if best_alignment:
            print(f"  {dataset.upper()}: {best_alignment} "
                  f"(Δ = {best_gain:+.4f})")


if __name__ == "__main__":
    main()