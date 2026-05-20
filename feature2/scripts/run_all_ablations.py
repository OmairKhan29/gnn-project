"""
Run all ablation experiments for Feature 2 Phase 4.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from feature2.ablation.runner import AblationRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations", nargs="+",
                       default=[
                           "no_task_conditioning", "no_alignment",
                           "lambda_0.0", "lambda_0.3", "lambda_0.5",
                           "proj_dim_32", "proj_dim_64",
                           "temp_0.1", "temp_0.2",
                       ])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    config = {
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 64,
        "epochs": args.epochs,
        "patience": 20,
        "grad_clip": 1.0,
        "hidden_dim": 128,
        "n_layers": 4,
        "task_dim": 64,
    }

    runner = AblationRunner(config=config, device=args.device, verbose=True)
    runner.run_all_ablations(args.ablations, args.seeds)


if __name__ == "__main__":
    import argparse
    main()