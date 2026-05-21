#!/usr/bin/env python
"""
Phase 1: Generate explanations for all molecules.
Saves explanations to disk for downstream use.

Usage:
    python feature3/scripts/phase1_run_explanations.py \
        --checkpoint checkpoints/best_model.pt \
        --task_idx 0 --epochs 100 --device cpu
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


TASK_NAMES = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53',
    'ClinTox_CT', 'ClinTox_FDA', 'BBBP', 'BACE', 'HIV_active',
]

TEST_SMILES = [
    ('c1ccccc1[N+](=O)[O-]', 1, 'Nitrobenzene (toxic)'),
    ('Nc1ccccc1', 1, 'Aniline'),
    ('CC(=O)Oc1ccccc1C(=O)O', 0, 'Aspirin'),
    ('c1ccccc1', 0, 'Benzene'),
    ('CCO', 0, 'Ethanol'),
    ('CC(=O)O', 0, 'Acetic acid'),
    ('c1ccc2c(c1)ccc2', 0, 'Naphthalene'),
    ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 0, 'Caffeine'),
    ('CC(C)Cc1ccc(C(C)C(=O)O)cc1', 0, 'Ibuprofen'),
    ('c1ccc(cc1)O', 0, 'Phenol'),
    ('C1=CC=CC=C1C(=O)O', 0, 'Benzoic acid'),
    ('C(C(=O)O)N', 0, 'Glycine'),
    ('C1=CC(=CC=C1O)O', 0, 'Hydroquinone'),
    ('C1=CC=NC=C1', 0, 'Pyridine'),
    ('CC(=O)N', 0, 'Acetamide'),
    ('C1=CC(=O)C=CC1=O', 1, 'Benzoquinone (toxic)'),
    ('C(CCl)Cl', 1, '1,2-Dichloroethane (toxic)'),
    ('Nc1ccccc1O', 1, 'Aminophenol'),
    ('c1ccc(cc1)[N+](=O)[O-]', 1, 'Para-nitrobenzene'),
    ('O=C1c2ccccc2C(=O)c3ccccc13', 0, 'Anthraquinone'),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Path to F1/F2 checkpoint')
    parser.add_argument('--task_idx', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--edge_size', type=float, default=0.005)
    parser.add_argument('--edge_entropy', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_dir', type=str,
                        default='results/feature3')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = TASK_NAMES[args.task_idx] if args.task_idx < len(TASK_NAMES) else f'Task_{args.task_idx}'

    print("=" * 60)
    print("Feature 3 Phase 1: Generating Explanations")
    print(f"Task: {task_name} (idx={args.task_idx})")
    print("=" * 60)

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    base_model = MultiTaskClassifier(
        node_dim=129, edge_dim=6, hidden_dim=128,
        task_dim=64, num_tasks=17, num_layers=4,
    )
    base_model.load_state_dict(state_dict, strict=False)

    model = MaskableModelWrapper(base_model)
    model.to(device)

    explainer = GNNExplainer(
        model=model,
        epochs=args.epochs,
        lr=args.lr,
        edge_size=args.edge_size,
        edge_entropy=args.edge_entropy,
    )

    featurizer = Molecule3DFeaturizer()

    # Process molecules
    print(f"\nProcessing {len(TEST_SMILES)} molecules...")
    explanations_out = []

    for smiles, label, description in TEST_SMILES:
        print(f"  Explaining: {description}...", end=' ', flush=True)

        data = featurizer.featurize(smiles)
        if data is None:
            print("SKIP (featurize failed)")
            continue

        target = torch.tensor([float(label)])
        exp = explainer.explain(
            data, task_idx=args.task_idx,
            target=target, device=device,
        )

        exp_serializable = {
            'smiles': smiles,
            'label': label,
            'description': description,
            'prediction': float(exp['prediction']),
            'edge_mask': exp['edge_mask'].tolist(),
            'node_importance': exp['node_importance'].tolist(),
            'node_feat_mask': exp['node_feat_mask'].tolist(),
            'converged': exp['converged'],
            'epochs_run': exp['epochs_run'],
            'final_loss': float(exp['loss_curve'][-1]) if exp['loss_curve'] else 0.0,
        }
        explanations_out.append(exp_serializable)

        correct = '✓' if round(exp['prediction']) == label else '✗'
        print(f"pred={exp['prediction']:.3f} [{correct}] loss={exp_serializable['final_loss']:.4f}")

    # Save
    output = {
        'task_idx': args.task_idx,
        'task_name': task_name,
        'n_molecules': len(explanations_out),
        'hyperparams': {
            'epochs': args.epochs,
            'lr': args.lr,
            'edge_size': args.edge_size,
            'edge_entropy': args.edge_entropy,
        },
        'explanations': explanations_out,
    }

    out_path = os.path.join(args.output_dir, f'phase1_task{args.task_idx}.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Saved {len(explanations_out)} explanations to: {out_path}")

    # Quick summary
    correct = sum(
        1 for e in explanations_out
        if round(e['prediction']) == e['label']
    )
    print(f"\nSummary for {task_name}:")
    print(f"  Molecules explained: {len(explanations_out)}")
    print(f"  Correctly predicted: {correct}/{len(explanations_out)}")
    preds = [e['prediction'] for e in explanations_out]
    print(f"  Pred range: [{min(preds):.3f}, {max(preds):.3f}]")


if __name__ == '__main__':
    main()