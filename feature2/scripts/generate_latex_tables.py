"""
Generate all LaTeX tables for Feature 2 publication.
"""

import sys
import os
import json
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from feature2.tables.latex_generator import (
    generate_main_results_table,
    generate_ablation_table,
    generate_low_data_table,
    generate_embedding_metrics_table,
)


def main():
    print("\n" + "=" * 70)
    print("Generating LaTeX Tables for Feature 2")
    print("=" * 70)

    base = "results/feature2"
    os.makedirs(os.path.join(base, "tables"), exist_ok=True)

    # Table 1: Main Results
    comp_path = os.path.join(base, "transfer_comparison", "full_comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            comparison = json.load(f)
        generate_main_results_table(comparison)
    else:
        print("Skipping main results table (no comparison data)")

    # Table 2: Ablation Study
    abl_path = os.path.join(base, "ablations", "all_ablation_results.json")
    if os.path.exists(abl_path):
        with open(abl_path) as f:
            ablation = json.load(f)
        from feature2.statistical.significance import analyze_ablations
        analysis = analyze_ablations(ablation)
        generate_ablation_table(analysis)
    else:
        print("Skipping ablation table (no ablation data)")

    # Tables 3 & 4: Low-Data Performance
    low_data_path = os.path.join(base, "low_data", "all_low_data_results.json")
    if os.path.exists(low_data_path):
        with open(low_data_path) as f:
            low_data = json.load(f)
        for dataset in ["sider", "muv"]:
            generate_low_data_table(low_data, dataset)
    else:
        print("Skipping low-data tables (no low-data results)")

    # Table 5: Embedding Metrics
    metric_files = glob.glob(os.path.join(base, "embeddings", "*.json"))
    if metric_files:
        generate_embedding_metrics_table(metric_files)
    else:
        print("Skipping embedding metrics table (no metric data)")

    print("\nAll LaTeX tables generated in: results/feature2/tables/")


if __name__ == "__main__":
    main()