#!/usr/bin/env python
"""
Phase 4: Generate LaTeX tables and run ablation.

Usage:
    python feature3/scripts/phase4_generate_tables.py \
        --checkpoint checkpoints/best_model.pt \
        --results_dir results/feature3 \
        --task_idx 0
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
from feature3.evaluation.fidelity import FidelityEvaluator
from feature3.evaluation.stability import StabilityEvaluator
from feature3.analysis.substructure_mapper import SubstructureMapper
from feature3.tables.latex_tables import LatexTableGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--task_idx', type=int, default=0)
    parser.add_argument('--results_dir', type=str, default='results/feature3')
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()


ABLATION_CONFIGS = [
    {
        'name': 'default',
        'edge_size': 0.005,
        'edge_entropy': 1.0,
        'epochs': 100,
    },
    {
        'name': 'high_entropy',
        'edge_size': 0.005,
        'edge_entropy': 2.0,
        'epochs': 100,
    },
    {
        'name': 'low_edge_size',
        'edge_size': 0.001,
        'edge_entropy': 1.0,
        'epochs': 100,
    },
    {
        'name': 'high_edge_size',
        'edge_size': 0.01,
        'edge_entropy': 1.0,
        'epochs': 100,
    },
    {
        'name': 'more_epochs',
        'edge_size': 0.005,
        'edge_entropy': 1.0,
        'epochs': 200,
    },
]


def main():
    args = parse_args()
    device = torch.device(args.device)
    tables_dir = os.path.join(args.results_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    print("=" * 60)
    print("Feature 3 Phase 4: Tables & Ablation Study")
    print("=" * 60)

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    base_model = MultiTaskClassifier(
        node_dim=129, edge_dim=6, hidden_dim=128,
        task_dim=64, num_tasks=17, num_layers=4,
    )
    base_model.load_state_dict(state_dict, strict=False)
    model = MaskableModelWrapper(base_model)
    model.to(device)

    # Load Phase 1 data
    phase1_path = os.path.join(args.results_dir, f'phase1_task{args.task_idx}.json')
    phase2_path = os.path.join(args.results_dir, f'phase2_substructures_task{args.task_idx}.json')

    if not os.path.exists(phase1_path):
        print(f"ERROR: Phase 1 results not found at {phase1_path}")
        print("Run phase1_run_explanations.py first")
        return

    with open(phase1_path) as f:
        phase1_data = json.load(f)

    task_name = phase1_data['task_name']

    # Rebuild data
    featurizer = Molecule3DFeaturizer()
    explanations_raw = phase1_data['explanations']

    data_list = []
    valid_raw = []
    for raw in explanations_raw:
        data = featurizer.featurize(raw['smiles'])
        if data:
            data_list.append(data)
            valid_raw.append(raw)

    explanations = [
        {
            'node_importance': torch.tensor(e['node_importance']),
            'edge_mask': torch.tensor(e['edge_mask']),
            'prediction': e['prediction'],
            'target': float(e['label']),
        }
        for e in valid_raw
    ]

    # ── Ablation Study ──────────────────────────────────────────────────
    print("\n[1/3] Running ablation study...")
    fid_eval = FidelityEvaluator(model, device=device)
    stab_eval = StabilityEvaluator(n_runs=3)
    ablation_results = {}

    for cfg in ABLATION_CONFIGS:
        cfg_name = cfg['name']
        print(f"  Running config: {cfg_name}...")

        explainer = GNNExplainer(
            model,
            epochs=cfg['epochs'],
            edge_size=cfg['edge_size'],
            edge_entropy=cfg['edge_entropy'],
        )

        # Run on subset (3 molecules for speed)
        subset_data = data_list[:3]
        abl_exps = explainer.explain_batch(
            subset_data, args.task_idx,
            device=device, verbose=False,
        )
        abl_masks = [e['edge_mask'] for e in abl_exps]

        fid = fid_eval.evaluate_dataset(subset_data, abl_masks, args.task_idx)
        stab = stab_eval.evaluate_dataset(
            explainer, subset_data, args.task_idx,
            device=device, max_mols=2,
        )

        ablation_results[cfg_name] = {
            **fid,
            'stability_mean': stab['stability_mean'],
            'stability_std': stab['stability_std'],
            'params': {
                'edge_size': cfg['edge_size'],
                'edge_entropy': cfg['edge_entropy'],
                'epochs': cfg['epochs'],
            },
        }
        print(
            f"    Fid+={fid['fidelity_plus_mean']:.3f} "
            f"Fid-={fid['fidelity_minus_mean']:.3f} "
            f"Sp={fid['sparsity_mean']:.3f}"
        )

    # ── Main Metrics Table ──────────────────────────────────────────────
    print("\n[2/3] Computing main metrics table...")
    fid_result = fid_eval.evaluate_dataset(
        data_list,
        [e['edge_mask'] for e in explanations],
        args.task_idx,
    )
    stab_result = stab_eval.evaluate_dataset(
        GNNExplainer(model, epochs=50),
        data_list, args.task_idx,
        device=device, max_mols=5,
    )

    metrics_by_task = {
        task_name: {
            **fid_result,
            'stability_mean': stab_result['stability_mean'],
            'stability_std': stab_result['stability_std'],
        }
    }

    # ── Substructures Table ─────────────────────────────────────────────
    task_substructures = {}
    if os.path.exists(phase2_path):
        with open(phase2_path) as f:
            phase2_data = json.load(f)
        top_subs = phase2_data.get('top_substructures', [])
        task_substructures[task_name] = [
            (s['name'], s['score']) for s in top_subs[:5]
        ]

    # ── Generate LaTeX Tables ───────────────────────────────────────────
    print("\n[3/3] Generating LaTeX tables...")
    gen = LatexTableGenerator(output_dir=tables_dir)

    gen.table1_explanation_metrics(metrics_by_task)
    print("  ✅ Table 1: Explanation Metrics")

    if task_substructures:
        gen.table2_top_substructures(task_substructures, top_k=5)
        print("  ✅ Table 2: Top Substructures")

    gen.table3_ablation(ablation_results)
    print("  ✅ Table 3: Ablation Study")

    # Save ablation results
    abl_path = os.path.join(args.results_dir, 'ablation_results.json')
    with open(abl_path, 'w') as f:
        json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                      for kk, vv in v.items()}
                   for k, v in ablation_results.items()}, f, indent=2)

    print(f"\n✅ Ablation saved to: {abl_path}")
    print("=" * 60)
    print("Phase 4 Complete!")
    print(f"Tables saved to: {tables_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()