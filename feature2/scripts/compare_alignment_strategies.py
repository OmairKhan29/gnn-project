"""
Compare alignment strategies across multiple seeds.
Generates summary table: no_alignment vs contrastive vs domain vs prototype.
"""

import sys
import os
import json
import glob
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_alignment_results(result_dir: str):
    """Load all alignment experiment results, grouped by strategy."""
    by_strategy = {}

    for fpath in glob.glob(os.path.join(result_dir, "*_results.json")):
        with open(fpath) as f:
            r = json.load(f)
        strategy = r.get("alignment_strategy", "unknown")
        if strategy not in by_strategy:
            by_strategy[strategy] = []
        by_strategy[strategy].append(r)

    return by_strategy


def main():
    result_dir = "results/feature2/alignment"

    if not os.path.exists(result_dir):
        print(f"No results at {result_dir}")
        return

    print("=" * 70)
    print("Phase 2: Alignment Strategy Comparison")
    print("=" * 70)

    by_strategy = load_alignment_results(result_dir)

    if not by_strategy:
        print("No results found.")
        return

    # Aggregate per strategy
    print(f"\n{'Strategy':<20} {'Val AUC':>15} {'Test AUC':>15} {'N Seeds':>10}")
    print("-" * 65)

    strategy_order = ["none", "contrastive", "domain", "prototype"]
    summary = {}

    for strat in strategy_order:
        if strat not in by_strategy:
            continue
        results = by_strategy[strat]
        val_aucs = [r["val_auc"] for r in results]
        test_aucs = [r["test_auc"] for r in results]

        summary[strat] = {
            "val_mean": float(np.mean(val_aucs)),
            "val_std": float(np.std(val_aucs)),
            "test_mean": float(np.mean(test_aucs)),
            "test_std": float(np.std(test_aucs)),
            "n_seeds": len(results),
        }

        print(f"{strat:<20} "
              f"{summary[strat]['val_mean']:>8.4f}±{summary[strat]['val_std']:<5.3f} "
              f"{summary[strat]['test_mean']:>8.4f}±{summary[strat]['test_std']:<5.3f} "
              f"{summary[strat]['n_seeds']:>10}")

    # Compute deltas vs baseline
    if "none" in summary:
        baseline = summary["none"]["test_mean"]
        print(f"\n{'Strategy':<20} {'Δ vs baseline':>20}")
        print("-" * 45)
        for strat, s in summary.items():
            delta = s["test_mean"] - baseline
            sign = "+" if delta >= 0 else ""
            print(f"{strat:<20} {sign}{delta:>15.4f}")

    # Save summary
    summary_path = os.path.join(result_dir, "alignment_comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()