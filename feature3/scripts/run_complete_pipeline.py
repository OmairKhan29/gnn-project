#!/usr/bin/env python
"""
Complete Feature 3 Pipeline.
Usage:
    python feature3/scripts/run_complete_pipeline.py \
        --checkpoint checkpoints/best_model.pt \
        --task_idx 0 \
        --n_molecules 50
"""

import argparse
import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.task_conditioned_egnn import MultiTaskClassifier
from data.featurizer import Molecule3DFeaturizer
from feature3.models.maskable_wrapper import MaskableModelWrapper
from feature3.explainer.gnn_explainer import GNNExplainer
from feature3.analysis.substructure_mapper import SubstructureMapper
from feature3.visualization.mol_visualizer import MoleculeVisualizer
from feature3.evaluation.metrics import evaluate_explanations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--task_idx', type=int, default=0)
    parser.add_argument('--n_molecules', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_dir', type=str, default='results/feature3')
    return parser.parse_args()


def get_test_smiles():
    """Diverse test set."""
    return [
        'c1ccccc1[N+](=O)[O-]',  # Nitrobenzene
        'Nc1ccccc1',              # Aniline
        'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
        'c1ccccc1',               # Benzene
        'CCO',                    # Ethanol
        'CC(=O)O',                # Acetic acid
        'c1ccc2c(c1)ccc2',        # Naphthalene
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CC(C)Cc1ccc(C(C)C(=O)O)cc1',    # Ibuprofen
        'c1ccc(cc1)O',            # Phenol
        'C1=CC=CC=C1C(=O)O',      # Benzoic acid
        'C(C(=O)O)N',             # Glycine
        'C1=CC(=CC=C1O)O',        # Hydroquinone
        'C1=CC=NC=C1',            # Pyridine
        'CC(=O)N',                # Acetamide
        'C1CCCCC1',               # Cyclohexane
        'C=C',                    # Ethylene
        'C#C',                    # Acetylene
        'c1ccc(cc1)[N+](=O)[O-]', # Nitrobenzene (check)
        'Nc1ccccc1O',             # Aminophenol
    ]


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("FEATURE 3: EXPLAINABLE MOLECULAR PREDICTION SYSTEM")
    print("=" * 70)
    
    # 1. Load F1 model (unchanged!)
    print(f"\n[1/5] Loading F1 model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    base_model = MultiTaskClassifier(
        node_dim=129, edge_dim=6, hidden_dim=128,
        task_dim=64, num_tasks=17, num_layers=4
    )
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.to(device)
    
    # 2. Wrap with Feature 3 wrapper (adds edge_weight support)
    print("[2/5] Wrapping model for explanation...")
    model = MaskableModelWrapper(base_model)
    
    # 3. Setup components
    featurizer = Molecule3DFeaturizer()
    explainer = GNNExplainer(model, epochs=args.epochs, lr=0.01)
    mapper = SubstructureMapper()
    visualizer = MoleculeVisualizer()
    
    # 4. Get molecules
    smiles_list = get_test_smiles()[:args.n_molecules]
    print(f"[3/5] Processing {len(smiles_list)} molecules...")
    
    data_list = []
    valid_smiles = []
    for smi in smiles_list:
        data = featurizer.featurize(smi)
        if data:
            data_list.append(data)
            valid_smiles.append(smi)
    
    # 5. Run explanations
    print(f"[4/5] Running GNNExplainer (task={args.task_idx}, epochs={args.epochs})...")
    explanations = explainer.explain_batch(data_list, args.task_idx, device)
    
    # 6. Substructure analysis
    print("[5/5] Analyzing substructures...")
    all_group_scores = {}
    
    for exp, smi in zip(explanations, valid_smiles):
        sub_data = mapper.map_to_substructures(smi, exp['node_importance'])
        for group, info in sub_data.items():
            if info['present']:
                if group not in all_group_scores:
                    all_group_scores[group] = []
                all_group_scores[group].append(info['score'])
    
    # Aggregate
    group_summary = {
        name: {'mean': np.mean(scores), 'std': np.std(scores), 'count': len(scores)}
        for name, scores in all_group_scores.items() if scores
    }
    
    # Rank
    ranked = sorted(group_summary.items(), key=lambda x: x[1]['mean'], reverse=True)
    print("\nTop 5 Important Substructures:")
    for name, stats in ranked[:5]:
        print(f"  {name:20s}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")
    
    # 7. Evaluation metrics
    print("\n[Bonus] Computing fidelity metrics...")
    metrics = evaluate_explanations(model, explainer, data_list[:10], valid_smiles[:10], 
                                   args.task_idx, device)
    
    print(f"  Fidelity+: {metrics['fidelity_plus_mean']:.3f} ± {metrics['fidelity_plus_std']:.3f}")
    print(f"  Fidelity-: {metrics['fidelity_minus_mean']:.3f} ± {metrics['fidelity_minus_std']:.3f}")
    print(f"  Sparsity:  {metrics['sparsity_mean']:.3f} ± {metrics['sparsity_std']:.3f}")
    
    # 8. Generate figures
    print("\n[Bonus] Generating figures...")
    
    # Figure 1: Explanation grid
    predictions = [exp['prediction'] for exp in explanations]
    labels = [exp['target'] for exp in explanations]
    
    fig1_path = os.path.join(args.output_dir, f'fig1_explanations_task{args.task_idx}.png')
    visualizer.plot_explanation_grid(valid_smiles[:9], explanations[:9], 
                                    predictions[:9], labels[:9],
                                    f"Task {args.task_idx}", save_path=fig1_path)
    print(f"  Saved: {fig1_path}")
    
    # Figure 2: Substructure bar chart
    if group_summary:
        fig2_path = os.path.join(args.output_dir, 'fig2_substructures.png')
        visualizer.plot_substructure_bar(
            {k: v['mean'] for k, v in group_summary.items()},
            title=f"Substructure Importance - Task {args.task_idx}",
            save_path=fig2_path
        )
        print(f"  Saved: {fig2_path}")
    
    # 9. Save results
    results = {
        'task_idx': args.task_idx,
        'n_molecules': len(valid_smiles),
        'metrics': metrics,
        'top_substructures': [{'name': name, **stats} for name, stats in ranked[:10]],
        'explanations': [
            {
                'smiles': smi,
                'prediction': exp['prediction'],
                'top_3_atoms': torch.topk(exp['node_importance'], min(3, len(exp['node_importance']))).indices.tolist()
            }
            for smi, exp in zip(valid_smiles, explanations)
        ]
    }
    
    json_path = os.path.join(args.output_dir, f'results_task{args.task_idx}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output_dir}")
    print("=" * 70)
    print("Feature 3 Pipeline Complete!")
    print("Deliverables:")
    print(f"  - JSON results: {json_path}")
    print(f"  - Explanation grid: {fig1_path}")
    print(f"  - Substructure chart: {fig2_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()