"""
SubstructureMapper
==================
Maps GNNExplainer atom importance scores to named chemical substructures.

Uses RDKit SMARTS pattern matching to identify functional groups,
then aggregates atom importance within each matched substructure.
"""

from rdkit import Chem
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


# ── Functional Group SMARTS Patterns ────────────────────────────────────
FUNCTIONAL_GROUPS = {
    # Toxicity-critical groups
    'nitro':             '[N+](=O)[O-]',
    'nitroso':           '[N]=O',
    'primary_amine':     '[NX3;H2;!$(NC=O)]',
    'secondary_amine':   '[NX3;H1;!$(NC=O)]',
    'aromatic_amine':    '[c][NH2]',
    'aldehyde':          '[CX3H1](=O)[#6]',
    'epoxide':           '[C]1[O][C]1',
    'peroxide':          '[OX2][OX2]',
    'alkyl_halide':      '[CX4][F,Cl,Br,I]',
    'acyl_halide':       '[CX3](=[OX1])[F,Cl,Br,I]',

    # Drug-relevant groups
    'carboxylic_acid':   '[CX3](=O)[OX2H1]',
    'ester':             '[#6][CX3](=O)[OX2H0][#6]',
    'amide':             '[NX3][CX3](=[OX1])[#6]',
    'ketone':            '[#6][CX3](=O)[#6]',
    'alcohol':           '[OX2H]',
    'phenol':            '[c][OX2H]',
    'thiol':             '[#16X2H]',
    'sulfone':           '[#16X4](=[OX1])(=[OX1])',
    'halide':            '[F,Cl,Br,I]',

    # Ring systems
    'benzene':           'c1ccccc1',
    'pyridine':          'c1ccncc1',
    'imidazole':         'c1cnc[nH]1',
    'furan':             'c1ccoc1',
    'thiophene':         'c1ccsc1',

    # Bonds
    'double_bond':       '[CX3]=[CX3]',
    'triple_bond':       '[CX2]#[CX2]',
    'carbonyl':          '[CX3]=[OX1]',
}

# Node feature names matching F1's Molecule3DFeaturizer
# These correspond to the 129-dim node feature vector
NODE_FEATURE_NAMES = (
    # Atom type one-hot (44 dims)
    [f'atom_type_{i}' for i in range(44)] +
    # Degree (11 dims)
    [f'degree_{i}' for i in range(11)] +
    # Implicit valence (7 dims)
    [f'valence_{i}' for i in range(7)] +
    # Formal charge (5 dims)
    [f'charge_{i}' for i in range(5)] +
    # Num Hs (5 dims)
    [f'num_hs_{i}' for i in range(5)] +
    # Hybridization (5 dims)
    [f'hybrid_{i}' for i in range(5)] +
    # Aromaticity (1 dim)
    ['is_aromatic'] +
    # Ring membership (7 dims)
    [f'in_ring_{i}' for i in range(7)] +
    # Chirality (4 dims)
    [f'chirality_{i}' for i in range(4)] +
    # 3D distance features (35 dims - approximate)
    [f'dist_feat_{i}' for i in range(35)]
)


class SubstructureMapper:
    """
    Maps atom-level importance to named functional groups.

    Args:
        patterns: Custom SMARTS patterns dict {name: smarts}
                  If None, uses built-in FUNCTIONAL_GROUPS
        aggregation: How to aggregate atom importance per group
                     'mean' | 'max' | 'sum'
    """

    def __init__(
        self,
        patterns: Optional[Dict[str, str]] = None,
        aggregation: str = 'mean',
    ):
        self.patterns = patterns if patterns is not None else FUNCTIONAL_GROUPS
        self.aggregation = aggregation

        # Compile SMARTS patterns
        self.compiled = {}
        self.failed = []
        for name, smarts in self.patterns.items():
            mol = Chem.MolFromSmarts(smarts)
            if mol is not None:
                self.compiled[name] = mol
            else:
                self.failed.append(name)

        if self.failed:
            print(f"[SubstructureMapper] Warning: Invalid SMARTS for: {self.failed}")

    def map_to_substructures(
        self,
        smiles: str,
        node_importance: torch.Tensor,
    ) -> Dict[str, Dict]:
        """
        Map atom importance to functional groups.

        Args:
            smiles: SMILES string
            node_importance: [N] per-atom importance scores in [0,1]

        Returns:
            Dict of {group_name: {
                present: bool,
                atoms: List[int],
                score: float,
                frequency: int (number of occurrences)
            }}
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        importance = node_importance.numpy()
        num_atoms = mol.GetNumAtoms()

        # Pad or clip importance to match atom count
        if len(importance) < num_atoms:
            importance = np.pad(
                importance, (0, num_atoms - len(importance))
            )
        else:
            importance = importance[:num_atoms]

        results = {}

        for name, pattern in self.compiled.items():
            matches = mol.GetSubstructMatches(pattern)

            if not matches:
                results[name] = {
                    'present': False,
                    'atoms': [],
                    'score': 0.0,
                    'frequency': 0,
                }
                continue

            # Collect unique atoms across all matches
            all_atoms = set()
            for match in matches:
                all_atoms.update(match)

            valid_atoms = [a for a in all_atoms if a < num_atoms]

            if not valid_atoms:
                score = 0.0
            elif self.aggregation == 'mean':
                score = float(np.mean(importance[valid_atoms]))
            elif self.aggregation == 'max':
                score = float(np.max(importance[valid_atoms]))
            elif self.aggregation == 'sum':
                score = float(np.sum(importance[valid_atoms]))
            else:
                score = float(np.mean(importance[valid_atoms]))

            results[name] = {
                'present': True,
                'atoms': valid_atoms,
                'score': score,
                'frequency': len(matches),
            }

        return results

    def rank_substructures(
        self,
        substructure_dict: Dict[str, Dict],
        top_k: int = 10,
        present_only: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Rank functional groups by importance score.

        Returns:
            List of (name, score) tuples, sorted descending
        """
        items = []
        for name, info in substructure_dict.items():
            if present_only and not info['present']:
                continue
            items.append((name, info['score']))

        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top_k]

    def get_atom_to_groups(self, smiles: str) -> Dict[int, List[str]]:
        """
        For each atom index, list functional groups it belongs to.
        Useful for annotation in visualizations.
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {}

        atom_groups = {i: [] for i in range(mol.GetNumAtoms())}
        for name, pattern in self.compiled.items():
            for match in mol.GetSubstructMatches(pattern):
                for atom_idx in match:
                    if atom_idx in atom_groups:
                        atom_groups[atom_idx].append(name)
        return atom_groups

    def dataset_summary(
        self,
        smiles_list: List[str],
        explanations: List[Dict],
    ) -> Dict[str, Dict]:
        """
        Aggregate substructure importance across a dataset.

        Args:
            smiles_list: List of SMILES strings
            explanations: List of GNNExplainer outputs

        Returns:
            Summary dict: {group_name: {mean, std, count, frequency}}
        """
        group_scores = {name: [] for name in self.compiled}

        for smiles, exp in zip(smiles_list, explanations):
            node_imp = exp['node_importance']
            results = self.map_to_substructures(smiles, node_imp)

            for name, info in results.items():
                if info['present'] and name in group_scores:
                    group_scores[name].append(info['score'])

        summary = {}
        for name, scores in group_scores.items():
            if scores:
                summary[name] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'count': len(scores),
                    'frequency': len(scores) / max(len(smiles_list), 1),
                }
            else:
                summary[name] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'count': 0,
                    'frequency': 0.0,
                }

        return summary

    def compare_tasks(
        self,
        smiles_list: List[str],
        explanations_by_task: Dict[int, List[Dict]],
        task_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare substructure importance across tasks.

        Args:
            smiles_list: Shared molecule set
            explanations_by_task: {task_idx: [explanations]}
            task_names: Optional human-readable task names

        Returns:
            {task_name: {group_name: mean_importance}}
        """
        result = {}
        for task_idx, exps in explanations_by_task.items():
            name = (
                task_names[task_idx]
                if task_names and task_idx < len(task_names)
                else f'Task_{task_idx}'
            )
            summary = self.dataset_summary(smiles_list, exps)
            result[name] = {
                group: info['mean']
                for group, info in summary.items()
                if info['count'] > 0
            }
        return result