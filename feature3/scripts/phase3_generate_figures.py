#!/usr/bin/env python
"""
Phase 3: Generate all 9 publication figures.
Loads Phase 1 and Phase 2 results.

Usage:
    python feature3/scripts/phase3_generate_figures.py \
        --checkpoint checkpoints/best_model.pt \
        --task_idx 0 \
        --results_dir results/feature3
"""

import argparse
import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')
))

from models.task_conditioned_egnn import MultiTaskClassifier
from data.featurizer import Molecule3DFeaturizer
from feature3.models.maskable_wrapper import MaskableModelWrapper
from feature3.explainer.gnn_explainer import GNNExplainer
from feature3.visualization.figure_builder import FigureBuilder
from feature3.evaluation.fidelity import FidelityEvaluator
from feature3.evaluation.stability import StabilityEvaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--task_idx', type=int, default=0)
    parser.add_argument('--results_dir', type=str,
                        default='results/feature3')
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()


def load_phase1(results_dir, task_idx):
    path = os.path.join(results_dir, f'phase1_task{task_idx}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_phase2(results_dir, task_idx):
    path = os.path.join(results_dir, f'phase2_substructures_task{task_idx}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    args = parse_args()
    device = torch.device(args.device)

    figures_dir = os.path.join(args.results_dir, 'figures')
    builder = FigureBuilder(output_dir=figures_dir, dpi=300)

    print("=" * 60)
    print("Feature 3 Phase 3: Generating Publication Figures")
    print("=" * 60)

    # Load previous results
    phase1 = load_phase1(args.results_dir, args.task_idx)
    phase2 = load_phase2(args.results_dir, args.task_idx)

    if not phase1:
        print(f"ERROR: Phase 1 results not found. Run phase1 first.")
        return

    task_name = phase1['task_name']
    explanations_raw = phase1['explanations']

    # Rebuild explanation dicts with tensors
    explanations = [
        {
            'node_importance': torch.tensor(e['node_importance']),
            'edge_mask': torch.tensor(e['edge_mask']),
            'prediction': e['prediction'],
            'target': float(e['label']),
            'converged': e.get('converged', False),
        }
        for e in explanations_raw
    ]
    smiles_list = [e['smiles'] for e in explanations_raw]

    # Load model for fidelity evaluation
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    base_model = MultiTaskClassifier(
        node_dim=129, edge_dim=6, hidden_dim=128,
        task_dim=64, num_tasks=17, num_layers=4,
    )
    base_model.load_state_dict(state_dict, strict=False)
    model = MaskableModelWrapper(base_model)
    model.to(device)

    explainer = GNNExplainer(model=model, epochs=50)
    featurizer = Molecule3DFeaturizer()

    # Rebuild data_list
    data_list = []
    valid_smiles = []
    valid_explanations = []
    for smi, exp in zip(smiles_list, explanations):
        data = featurizer.featurize(smi)
        if data:
            data_list.append(data)
            valid_smiles.append(smi)
            valid_explanations.append(exp)

    print(f"\nGenerating figures for {task_name}...")
    saved = []

    # Figure 1: Atom importance grid
    print("\n[Fig 1] Atom importance grid...")
    path = builder.fig1_atom_importance_grid(
        valid_smiles, valid_explanations, task_name
    )
    saved.append(('Figure 1: Atom Importance Grid', path))

    # Figure 2: Bond importance grid
    print("\n[Fig 2] Bond importance grid...")
    path = builder.fig2_bond_importance_grid(
        valid_smiles, valid_explanations, data_list, task_name
    )
    saved.append(('Figure 2: Bond Importance Grid', path))

    # Figure 3: Substructure importance
    if phase2:
        print("\n[Fig 3] Substructure importance...")
        path = builder.fig3_substructure_importance(
            phase2['group_summary'], task_name
        )
        if path:
            saved.append(('Figure 3: Substructure Importance', path))

    # Figure 4: Fidelity comparison
    print("\n[Fig 4] Fidelity comparison...")
    fid_eval = FidelityEvaluator(model, device=device)
    fid_result = fid_eval.evaluate_dataset(
        data_list[:10],
        [e['edge_mask'] for e in valid_explanations[:10]],
        args.task_idx,
    )
    metrics_by_task = {task_name: fid_result}
    path = builder.fig4_fidelity_comparison(metrics_by_task)
    saved.append(('Figure 4: Fidelity Comparison', path))

    # Figure 5: Stability
    print("\n[Fig 5] Stability evaluation (slow)...")
    stab_eval = StabilityEvaluator(n_runs=3)
    stab_result = stab_eval.evaluate_dataset(
        explainer, data_list, args.task_idx,
        device=device, max_mols=5,
    )
    stability_by_task = {task_name: stab_result['stability_mean']}
    path = builder.fig5_stability(stability_by_task)
    saved.append(('Figure 5: Stability', path))

    # Figure 6: Cross-task heatmap (single task still useful)
    if phase2:
        print("\n[Fig 6] Cross-task substructure heatmap...")
        task_group_scores = {
            task_name: {
                g: v['mean']
                for g, v in phase2['group_summary'].items()
                if v['count'] > 0
            }
        }
        path = builder.fig6_cross_task_heatmap(task_group_scores)
        saved.append(('Figure 6: Cross-task Heatmap', path))

    # Figure 7: Feature radar
    print("\n[Fig 7] Node feature importance radar...")
    feat_imps = {
        task_name: np.array([e['node_importance'].mean().item()
                             for e in valid_explanations])
    }
    feat_imp_arr = np.stack([
        e['node_importance'].numpy()[:89]  # Clip to feature groups
        for e in valid_explanations
    ])
    mean_feat_imp = feat_imp_arr.mean(axis=0) if len(feat_imp_arr) > 0 else np.zeros(89)
    path = builder.fig7_feature_radar({task_name: mean_feat_imp})
    saved.append(('Figure 7: Feature Radar', path))

    # Figure 8: Per-task metrics
    print("\n[Fig 8] Per-task metrics summary...")
    combined_metrics = {
        task_name: {
            **fid_result,
            'stability_mean': stab_result['stability_mean'],
            'stability_std': stab_result['stability_std'],
        }
    }
    path = builder.fig8_per_task_metrics(combined_metrics)
    saved.append(('Figure 8: Per-Task Metrics', path))

    # Figure 9: Correct vs Incorrect
    print("\n[Fig 9] Correct vs Incorrect explanations...")
    correct_smi, correct_exp = [], []
    incorrect_smi, incorrect_exp = [], []

    for smi, exp, raw in zip(valid_smiles, valid_explanations, explanations_raw):
        if round(exp['prediction']) == raw['label']:
            correct_smi.append(smi)
            correct_exp.append(exp)
        else:
            incorrect_smi.append(smi)
            incorrect_exp.append(exp)

    if correct_smi and incorrect_smi:
        path = builder.fig9_correct_vs_incorrect(
            correct_smi, correct_exp,
            incorrect_smi, incorrect_exp,
            task_name,
        )
        saved.append(('Figure 9: Correct vs Incorrect', path))
    else:
        print("  Skipping Fig 9 (need both correct and incorrect predictions)")

    # Summary
    print("\n" + "=" * 60)
    print("Figure Generation Complete!")
    print("=" * 60)
    for desc, path in saved:
        if path:
            print(f"  ✅ {desc}")
            print(f"     {path}")


if __name__ == '__main__':
    main()