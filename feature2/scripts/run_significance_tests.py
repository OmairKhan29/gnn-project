"""
Run statistical significance tests for Feature 2.
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from feature2.statistical.significance import (
    paired_t_test,
    cohen_d,
    format_significance_stars,
)


def main():
    print("\n" + "=" * 70)
    print("Running Significance Tests for Feature 2")
    print("=" * 70)

    # Load comparison results
    comp_path = "results/feature2/transfer_comparison/full_comparison.json"
    if not os.path.exists(comp_path):
        print(f"No comparison results at {comp_path}")
        return

    with open(comp_path) as f:
        comparison = json.load(f)

    # Paired t-tests: aligned vs unaligned
    results = {}
    alignments = ["contrastive", "domain", "prototype"]
    datasets = ["sider", "muv"]

    for alignment in alignments:
        for dataset in datasets:
            unaligned_data = comparison.get("unaligned", {}).get(dataset, {}).get("linear_probe", {}).get("per_seed", [])
            aligned_data = comparison.get(alignment, {}).get(dataset, {}).get("linear_probe", {}).get("per_seed", [])

            if len(unaligned_data) >= 2 and len(aligned_data) >= 2:
                t_res = paired_t_test(aligned_data, unaligned_data, alternative="greater")
                effect = cohen_d(aligned_data, unaligned_data)

                results[f"{alignment}_{dataset}"] = {
                    "t_stat": t_res["t_statistic"],
                    "p_value": t_res["p_value"],
                    "stars": format_significance_stars(t_res["p_value"]),
                    "cohen_d": effect["cohen_d"],
                    "interpretation": effect["interpretation"],
                    "mean_aligned": t_res["mean_a"],
                    "mean_unaligned": t_res["mean_b"],
                    "significant": t_res["significant"],
                }

    # Save
    out_path = "results/feature2/significance_tests.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'Comparison':<25} {'T-Stat':>10} {'P-Value':>10} {'Stars':>8} {'Cohen\'s d':>12} {'Significant?':>12}")
    print("-" * 85)
    for key, r in results.items():
        sig_flag = "YES ✅" if r["significant"] else "NO ❌"
        print(f"{key:<25} {r['t_stat']:>10.3f} {r['p_value']:>10.4f} {r['stars']:>8} {r['cohen_d']:>12.3f} {sig_flag:>12}")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()