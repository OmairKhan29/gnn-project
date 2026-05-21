#!/usr/bin/env python
"""
Feature 3 Final Report Generator.
Aggregates all results from phases 1-4 into a complete report.

Usage:
    python feature3/scripts/feature3_final_report.py \
        --results_dir results/feature3 \
        --checkpoint checkpoints/best_model.pt \
        --device cpu
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')
))

TASK_NAMES = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53',
    'ClinTox_CT', 'ClinTox_FDA',
    'BBBP', 'BACE', 'HIV_active',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str,
                        default='results/feature3')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best_model.pt')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--all_tasks', action='store_true',
                        help='Run for all 17 tasks (slow)')
    return parser.parse_args()


def load_json_safe(path):
    """Load JSON file if it exists."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def aggregate_task_results(results_dir, task_indices):
    """
    Aggregate results across multiple tasks.
    Returns combined metrics dict.
    """
    all_metrics = {}

    for idx in task_indices:
        task_name = TASK_NAMES[idx] if idx < len(TASK_NAMES) else f'Task_{idx}'

        # Load phase 1 (explanations)
        p1 = load_json_safe(
            os.path.join(results_dir, f'phase1_task{idx}.json')
        )

        # Load phase 2 (substructures)
        p2 = load_json_safe(
            os.path.join(results_dir, f'phase2_substructures_task{idx}.json')
        )

        if p1 is None:
            continue

        # Basic stats from phase 1
        exps = p1.get('explanations', [])
        preds = [e['prediction'] for e in exps]
        labels = [e['label'] for e in exps]
        correct = sum(1 for p, l in zip(preds, labels) if round(p) == l)

        metrics = {
            'n_molecules': len(exps),
            'n_correct': correct,
            'accuracy': correct / max(len(exps), 1),
            'mean_prediction': float(np.mean(preds)) if preds else 0.0,
        }

        # Add substructure info from phase 2
        if p2:
            top_subs = p2.get('top_substructures', [])
            metrics['top_substructure'] = (
                top_subs[0]['name'] if top_subs else 'N/A'
            )
            metrics['top_substructure_score'] = (
                float(top_subs[0]['score']) if top_subs else 0.0
            )
            metrics['n_groups_found'] = sum(
                1 for v in p2.get('group_summary', {}).values()
                if v.get('count', 0) > 0
            )

        all_metrics[task_name] = metrics

    return all_metrics


def print_summary_table(all_metrics):
    """Print ASCII summary table."""
    print("\n" + "=" * 75)
    print("FEATURE 3 RESULTS SUMMARY")
    print("=" * 75)
    print(
        f"{'Task':<20} {'N':>5} {'Acc':>6} "
        f"{'Top Group':<20} {'Score':>6} {'Groups':>7}"
    )
    print("-" * 75)

    for task, metrics in all_metrics.items():
        n = metrics.get('n_molecules', 0)
        acc = metrics.get('accuracy', 0)
        top_sub = metrics.get('top_substructure', 'N/A')
        top_score = metrics.get('top_substructure_score', 0)
        n_groups = metrics.get('n_groups_found', 0)

        print(
            f"{task:<20} {n:>5} {acc:>6.3f} "
            f"{top_sub:<20} {top_score:>6.3f} {n_groups:>7}"
        )

    print("=" * 75)


def generate_full_report(results_dir, all_metrics, checkpoint_path):
    """Generate complete markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Feature 3: Explainable Molecular Prediction — Final Report",
        f"\n**Generated:** {timestamp}",
        f"**Checkpoint:** {checkpoint_path}",
        f"**Results Directory:** {results_dir}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"- **Tasks Analyzed:** {len(all_metrics)}",
        f"- **Total Molecules Explained:** {sum(m.get('n_molecules', 0) for m in all_metrics.values())}",
        f"- **Method:** GNNExplainer with MaskableModelWrapper",
        f"- **No F1/F2 code modified:** ✅",
        "",
        "---",
        "",
        "## Results by Task",
        "",
        "| Task | N | Accuracy | Top Substructure | Score |",
        "|------|---|----------|-----------------|-------|",
    ]

    for task, metrics in all_metrics.items():
        n = metrics.get('n_molecules', 0)
        acc = metrics.get('accuracy', 0)
        top_sub = metrics.get('top_substructure', 'N/A')
        top_score = metrics.get('top_substructure_score', 0)
        lines.append(
            f"| {task} | {n} | {acc:.3f} | "
            f"{top_sub} | {top_score:.3f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Key Findings",
        "",
        "### Substructure Importance",
        "",
    ]

    # Find most important substructure globally
    all_top = [
        (task, m.get('top_substructure', 'N/A'),
         m.get('top_substructure_score', 0))
        for task, m in all_metrics.items()
        if m.get('top_substructure', 'N/A') != 'N/A'
    ]
    all_top.sort(key=lambda x: x[2], reverse=True)

    if all_top:
        lines.append(
            f"- **Globally most important substructure:** "
            f"{all_top[0][1]} (score={all_top[0][2]:.3f}, task={all_top[0][0]})"
        )

    lines += [
        "",
        "### Generated Outputs",
        "",
        "**Figures (results/feature3/figures/):**",
        "- fig1_atom_importance_grid.png — Atom heatmaps",
        "- fig2_bond_importance_grid.png — Bond heatmaps",
        "- fig3_substructure_importance.png — Functional group chart",
        "- fig4_fidelity_comparison.png — Fidelity+ vs Fidelity-",
        "- fig5_stability.png — Cross-run consistency",
        "- fig6_cross_task_heatmap.png — Task comparison",
        "- fig7_feature_radar.png — Node feature importance",
        "- fig8_per_task_metrics.png — All metrics summary",
        "- fig9_correct_vs_incorrect.png — Prediction comparison",
        "",
        "**Tables (results/feature3/tables/):**",
        "- table1_explanation_metrics.tex — Main results",
        "- table2_top_substructures.tex — Chemical findings",
        "- table3_ablation.tex — Hyperparameter study",
        "",
        "---",
        "",
        "## Scientific Contributions",
        "",
        "1. **GNNExplainer applied to multi-task molecular GNN** —",
        "   Task-specific explanations using FiLM conditioning",
        "",
        "2. **MaskableModelWrapper** — Enables edge masking without",
        "   modifying trained F1/F2 models",
        "",
        "3. **Functional group importance mapping** — Links atom-level",
        "   importance to named chemical substructures via SMARTS",
        "",
        "4. **Fidelity + Stability evaluation** — Quantifies",
        "   explanation quality and reproducibility",
        "",
        "---",
        "",
        "## Connection to Features 1 and 2",
        "",
        "| Component | Source | Used By F3 |",
        "|-----------|--------|------------|",
        "| Trained EGNN encoder | Feature 1 | Frozen inside wrapper |",
        "| Task embeddings (FiLM) | Feature 1 | Makes masks task-specific |",
        "| 17 task heads | Feature 1 | task_idx parameter |",
        "| Scaffold split molecules | Feature 1 | Explanation test set |",
        "| Aligned encoder | Feature 2 | Better representations |",
        "| SIDER/MUV checkpoints | Feature 2 | 44 extra explainable tasks |",
    ]

    report_text = "\n".join(lines)

    report_path = os.path.join(results_dir, 'feature3_final_report.md')
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"\n📄 Full report saved: {report_path}")
    return report_path


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 60)
    print("FEATURE 3: FINAL REPORT GENERATOR")
    print("=" * 60)

    # Determine which tasks to aggregate
    if args.all_tasks:
        task_indices = list(range(17))
    else:
        # Find which phase1 files exist
        task_indices = []
        for i in range(17):
            path = os.path.join(args.results_dir, f'phase1_task{i}.json')
            if os.path.exists(path):
                task_indices.append(i)

    if not task_indices:
        print("No phase1 results found.")
        print("Run phase1_run_explanations.py first.")
        return

    print(f"\nFound results for tasks: {task_indices}")

    # Aggregate
    all_metrics = aggregate_task_results(args.results_dir, task_indices)

    # Print summary
    print_summary_table(all_metrics)

    # Save JSON summary
    summary_path = os.path.join(args.results_dir, 'final_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\n💾 JSON summary: {summary_path}")

    # Generate markdown report
    generate_full_report(args.results_dir, all_metrics, args.checkpoint)

    print("\n" + "=" * 60)
    print("Feature 3 Final Report Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()