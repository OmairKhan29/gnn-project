"""
Phase 1: Baseline Report Generator.
Summarizes all transfer learning results from Phase 1.
"""

import sys
import os
import json
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from feature2.evaluation.transfer_metrics import build_comparison_table


def main():
    result_dir = "results/feature2/transfer_baselines"

    if not os.path.exists(result_dir):
        print(f"No results found at {result_dir}")
        print("Run evaluation scripts first.")
        return

    print("=" * 70)
    print("Feature 2 Phase 1: Transfer Learning Baseline Report")
    print("=" * 70)

    # Load all result files
    all_results = {}
    for fpath in glob.glob(os.path.join(result_dir, "*.json")):
        fname = os.path.basename(fpath)
        with open(fpath) as f:
            data = json.load(f)

        # Handle different result formats
        if isinstance(data, dict):
            if "experiment_name" in data:
                # Single experiment result
                all_results[data["experiment_name"]] = data
            else:
                # Aggregated results (dataset → result)
                for key, val in data.items():
                    if isinstance(val, dict) and "mean_test_auc" in val:
                        exp_name = f"{val.get('strategy', 'unknown')}_{key}"
                        all_results[exp_name] = {
                            "experiment_name": exp_name,
                            "val_auc": val.get("mean_test_auc", 0.0),
                            "test_auc": val.get("mean_test_auc", 0.0),
                            "std": val.get("std_test_auc", 0.0),
                            "strategy": val.get("strategy", "unknown"),
                            "dataset": key,
                        }

    if not all_results:
        print("No results found. Run the evaluation scripts first.")
        return

    # Summary by strategy and dataset
    strategies = {}
    for name, r in all_results.items():
        strategy = r.get("strategy", "unknown")
        dataset = r.get("dataset", "unknown")
        if strategy not in strategies:
            strategies[strategy] = {}
        strategies[strategy][dataset] = r.get("test_auc", 0.0)

    print("\nTransfer Performance by Strategy:")
    print("-" * 50)
    print(f"{'Strategy':<25} {'SIDER':>10} {'MUV':>10}")
    print("-" * 50)

    strategy_order = ["zero_shot", "linear_probe", "top_layers", "full", "scratch"]
    for strategy in strategy_order:
        if strategy in strategies:
            sider_auc = strategies[strategy].get("sider", 0.0)
            muv_auc = strategies[strategy].get("muv", 0.0)
            print(f"{strategy:<25} {sider_auc:>10.4f} {muv_auc:>10.4f}")

    print("\n" + "=" * 70)
    print("Phase 1 Complete. Next: Phase 2 (Alignment Module)")
    print("=" * 70)


if __name__ == "__main__":
    main()