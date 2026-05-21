"""
ExplanationEvaluator: Orchestrates complete evaluation.
Runs fidelity + stability + substructure for all tasks.
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional
import torch

from feature3.evaluation.fidelity import FidelityEvaluator
from feature3.evaluation.stability import StabilityEvaluator
from feature3.analysis.substructure_mapper import SubstructureMapper


class ExplanationEvaluator:
    """
    Runs complete evaluation of GNNExplainer explanations.

    Usage:
        evaluator = ExplanationEvaluator(model, device=device)
        report = evaluator.run(
            explainer, data_list, smiles_list, task_idx=0
        )
        evaluator.save_report(report, 'results/feature3')
    """

    def __init__(
        self,
        model,
        device: torch.device = torch.device('cpu'),
        fidelity_threshold: float = 0.5,
        stability_runs: int = 3,
    ):
        self.model = model
        self.device = device
        self.fidelity_evaluator = FidelityEvaluator(
            model, fidelity_threshold, device
        )
        self.stability_evaluator = StabilityEvaluator(stability_runs)
        self.substructure_mapper = SubstructureMapper()

    def run(
        self,
        explainer,
        data_list: List,
        smiles_list: List[str],
        task_idx: int,
        task_name: Optional[str] = None,
        n_stability_mols: int = 5,
        verbose: bool = True,
    ) -> Dict:
        """
        Complete evaluation pipeline for one task.

        Args:
            explainer: GNNExplainer instance
            data_list: List of PyG Data objects
            smiles_list: Corresponding SMILES strings
            task_idx: Task to evaluate
            task_name: Human-readable task name
            n_stability_mols: Molecules to use for stability (slow)
            verbose: Print progress

        Returns:
            Complete evaluation report dict
        """
        name = task_name or f'Task_{task_idx}'
        if verbose:
            print(f"\n{'='*50}")
            print(f"Evaluating Task: {name} (idx={task_idx})")
            print(f"{'='*50}")

        # 1. Generate all explanations
        if verbose:
            print(f"[1/4] Generating explanations for {len(data_list)} molecules...")
        explanations = explainer.explain_batch(
            data_list, task_idx, device=self.device, verbose=verbose
        )

        # 2. Fidelity evaluation
        if verbose:
            print(f"[2/4] Computing fidelity metrics...")
        valid_data = []
        valid_masks = []
        for data, exp in zip(data_list, explanations):
            if not exp.get('failed', False):
                valid_data.append(data)
                valid_masks.append(exp['edge_mask'])

        fidelity = self.fidelity_evaluator.evaluate_dataset(
            valid_data, valid_masks, task_idx
        )

        # 3. Stability evaluation
        if verbose:
            print(f"[3/4] Computing stability ({n_stability_mols} molecules)...")
        stability = self.stability_evaluator.evaluate_dataset(
            explainer, data_list, task_idx,
            device=self.device, max_mols=n_stability_mols,
        )

        # 4. Substructure analysis
        if verbose:
            print(f"[4/4] Analyzing substructures...")
        valid_exps = [e for e in explanations if not e.get('failed', False)]
        valid_smi = [
            s for s, e in zip(smiles_list, explanations)
            if not e.get('failed', False)
        ]
        group_summary = self.substructure_mapper.dataset_summary(
            valid_smi, valid_exps
        )

        # Top substructures
        ranked = self.substructure_mapper.rank_substructures(
            {
                g: {'present': info['count'] > 0, 'score': info['mean']}
                for g, info in group_summary.items()
            },
            top_k=10,
        )

        # Prediction distribution
        preds = [e.get('prediction', 0.5) for e in valid_exps]
        targets = [e.get('target', 0) for e in valid_exps]

        report = {
            'task_idx': task_idx,
            'task_name': name,
            'n_molecules': len(data_list),
            'n_valid': len(valid_exps),
            'fidelity': fidelity,
            'stability': stability,
            'substructure_summary': group_summary,
            'top_substructures': ranked,
            'predictions': {
                'mean': float(np.mean(preds)) if preds else 0.0,
                'std': float(np.std(preds)) if preds else 0.0,
                'fraction_positive': float(np.mean([p > 0.5 for p in preds])) if preds else 0.0,
            },
        }

        if verbose:
            print(f"\nResults for {name}:")
            print(f"  Fidelity+  : {fidelity['fidelity_plus_mean']:.3f} ± {fidelity['fidelity_plus_std']:.3f}")
            print(f"  Fidelity−  : {fidelity['fidelity_minus_mean']:.3f} ± {fidelity['fidelity_minus_std']:.3f}")
            print(f"  Sparsity   : {fidelity['sparsity_mean']:.3f} ± {fidelity['sparsity_std']:.3f}")
            print(f"  Stability  : {stability['stability_mean']:.3f} ± {stability['stability_std']:.3f}")
            print(f"  Top groups : {ranked[:3]}")

        return report

    def save_report(self, report: Dict, output_dir: str) -> str:
        """Save evaluation report to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        task_name = report.get('task_name', 'unknown')
        path = os.path.join(output_dir, f'report_{task_name}.json')

        # Make serializable
        serializable = self._make_serializable(report)

        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f"Report saved: {path}")
        return path

    def _make_serializable(self, obj):
        """Recursively make object JSON-serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(x) for x in obj]
        elif isinstance(obj, tuple):
            return [self._make_serializable(x) for x in obj]
        elif isinstance(obj, (np.float32, np.float64, float)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, int)):
            return int(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj