"""
Phase 3 Final Report.
Consolidates all Phase 3 results into a single summary.
"""

import sys
import os
import json
import glob
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_json_safe(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    print("\n" + "=" * 70)
    print("FEATURE 2 — PHASE 3 FINAL REPORT")
    print("Cross-Dataset Transfer + Low-Data Learning")
    print("=" * 70)

    base = "results/feature2"

    # ── Section 1: Transfer comparison ─────────────────
    print("\n[1] Transfer Comparison: Aligned vs Unaligned")
    print("-" * 55)

    gain_path = os.path.join(base, "transfer_comparison", "transfer_gain_table.json")
    if os.path.exists(gain_path):
        gain_table = load_json_safe(gain_path)
        for alignment, ds_data in gain_table.items():
            for dataset, strat_data in ds_data.items():
                for strategy, g in strat_data.items():
                    flag = "✅" if g.get("is_positive") else "❌"
                    print(f"  {alignment:<14} → {dataset.upper():<6} "
                          f"[{strategy:<12}] "
                          f"Δ = {g.get('absolute_gain', 0):+.4f}  {flag}")
    else:
        print("  No transfer comparison results found.")
        print("  Run: python feature2/scripts/run_transfer_pipeline.py")

    # ── Section 2: Low-data summary ─────────────────────
    print("\n[2] Low-Data Performance Summary")
    print("-" * 55)

    all_low_data_path = os.path.join(base, "low_data", "all_low_data_results.json")
    if os.path.exists(all_low_data_path):
        all_results = load_json_safe(all_low_data_path)
        if isinstance(all_results, list) and all_results:
            # Get unique combinations
            combos = set()
            for r in all_results:
                combos.add((
                    r.get("dataset", ""),
                    r.get("strategy", ""),
                    r.get("alignment", ""),
                ))

            # Print per fraction
            for dataset in ["sider", "muv"]:
                print(f"\n  {dataset.upper()}")
                print(f"  {'Strategy':<35} "
                      f"{'10%':>8} {'50%':>8} {'100%':>8}")
                print(f"  {'-' * 60}")

                relevant = [
                    (s, a) for (d, s, a) in combos if d == dataset
                ]
                for strategy, alignment in sorted(relevant):
                    label = (f"{alignment}/{strategy}"
                             if strategy != "scratch" else "scratch")

                    row = f"  {label:<35}"
                    for frac in [0.10, 0.50, 1.00]:
                        vals = [
                            r.get("test_auc", 0.5)
                            for r in all_results
                            if (r.get("dataset") == dataset and
                                r.get("strategy") == strategy and
                                r.get("alignment") == alignment and
                                abs(r.get("fraction", -1) - frac) < 0.01 and
                                "error" not in r)
                        ]
                        row += (f" {np.mean(vals):>8.4f}"
                                if vals else f" {'N/A':>8}")
                    print(row)
        else:
            print("  No results found in all_low_data_results.json")
    else:
        print("  No low-data results found.")
        print("  Run: python feature2/scripts/run_low_data_experiment.py")

    # ── Section 3: Generated files ──────────────────────
    print("\n[3] Generated Files")
    print("-" * 55)
    for fpath in sorted(glob.glob(os.path.join(base, "**/*"), recursive=True)):
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            rel = os.path.relpath(fpath, base)
            print(f"  {rel:<55} {size:>8} bytes")

    print("\n" + "=" * 70)
    print("Phase 3 Complete.")
    print("Next: Phase 4 — Ablations + Visualizations + LaTeX Tables")
    print("=" * 70)


if __name__ == "__main__":
    main()