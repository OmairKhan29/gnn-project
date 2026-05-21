#!/usr/bin/env python
"""
Phase 2: Substructure analysis from Phase 1 explanations.
Loads explanation JSONs and identifies important chemical groups.

Usage:
    python feature3/scripts/phase2_substructure_analysis.py \
        --explanation_dir results/feature3 \
        --task_idx 0
"""

import argparse
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')
))

from feature3.analysis.substructure_mapper import SubstructureMapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--explanation_dir', type=str,
                        default='results/feature3')
    parser.add_argument('--task_idx', type=int, default=0)
    parser.add_argument('--output_dir', type=str,
                        default='results/feature3')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Feature 3 Phase 2: Substructure Analysis")
    print("=" * 60)

    # Load Phase 1 output
    phase1_path = os.path.join(
        args.explanation_dir, f'phase1_task{args.task_idx}.json'
    )

    if not os.path.exists(phase1_path):
        print(f"ERROR: Phase 1 output not found: {phase1_path}")
        print("Run phase1_run_explanations.py first")
        return

    with open(phase1_path) as f:
        phase1_data = json.load(f)

    task_name = phase1_data['task_name']
    explanations = phase1_data['explanations']

    print(f"\nTask: {task_name}")
    print(f"Explanations: {len(explanations)} molecules")

    # Rebuild explanations with torch tensors
    mapper = SubstructureMapper()

    smiles_list = [e['smiles'] for e in explanations]
    reconstructed_exps = [
        {
            'node_importance': torch.tensor(e['node_importance']),
            'prediction': e['prediction'],
            'label': e['label'],
        }
        for e in explanations
    ]

    # Dataset-level substructure analysis
    print("\nComputing dataset-level substructure importance...")
    summary = mapper.dataset_summary(smiles_list, reconstructed_exps)

    # Print results
    present = {k: v for k, v in summary.items() if v['count'] > 0}
    sorted_groups = sorted(present.items(), key=lambda x: x[1]['mean'], reverse=True)

    print(f"\nTop 10 Important Substructures ({task_name}):")
    print(f"{'Group':<25} {'Mean':>6} {'Std':>6} {'Count':>6} {'Freq':>6}")
    print("-" * 55)
    for name, stats in sorted_groups[:10]:
        print(
            f"{name:<25} "
            f"{stats['mean']:>6.3f} "
            f"{stats['std']:>6.3f} "
            f"{stats['count']:>6d} "
            f"{stats['frequency']:>6.2f}"
        )

    # Per-molecule analysis
    print("\nPer-molecule top substructures:")
    for exp_data, exp_rec in zip(explanations[:5], reconstructed_exps[:5]):
        sub = mapper.map_to_substructures(
            exp_data['smiles'], exp_rec['node_importance']
        )
        ranked = mapper.rank_substructures(sub, top_k=3)
        print(
            f"  {exp_data['description'][:25]:<25} "
            f"top: {[f'{n}({s:.2f})' for n, s in ranked[:3]]}"
        )

    # Save
    output = {
        'task_idx': args.task_idx,
        'task_name': task_name,
        'group_summary': {
            k: {
                'mean': float(v['mean']),
                'std': float(v['std']),
                'count': int(v['count']),
                'frequency': float(v['frequency']),
            }
            for k, v in summary.items()
        },
        'top_substructures': [
            {'name': name, 'score': float(score)}
            for name, score in sorted_groups[:10]
        ],
    }

    out_path = os.path.join(
        args.output_dir, f'phase2_substructures_task{args.task_idx}.json'
    )
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Saved substructure analysis to: {out_path}")


if __name__ == '__main__':
    main()