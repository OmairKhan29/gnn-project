"""
MoleculeVisualizer
==================
Renders molecular structures with atom/bond importance heatmaps.
Uses RDKit's MolDraw2DCairo for high-quality PNG output.
"""

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import torch
import io
import os
from typing import Dict, List, Optional, Tuple
from PIL import Image


def _importance_to_rgb(
    importance: float,
    colormap: str = 'RdYlGn',
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> Tuple[float, float, float]:
    """Map scalar importance to RGB tuple for RDKit."""
    cmap = cm.get_cmap(colormap)
    norm = np.clip((importance - vmin) / (vmax - vmin + 1e-8), 0, 1)
    rgba = cmap(norm)
    return (float(rgba[0]), float(rgba[1]), float(rgba[2]))


class MoleculeVisualizer:
    """
    Draws molecules with atom/bond importance highlights.

    Args:
        img_size: (width, height) in pixels
        colormap: matplotlib colormap for importance colors
                  'RdYlGn' = red (low) to green (high)
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (400, 400),
        colormap: str = 'RdYlGn',
    ):
        self.img_size = img_size
        self.colormap = colormap

    def draw_atom_importance(
        self,
        smiles: str,
        node_importance: torch.Tensor,
        normalize: bool = True,
        threshold: float = 0.1,
    ) -> Optional[Image.Image]:
        """
        Draw molecule with atoms colored by importance score.

        Args:
            smiles: SMILES string
            node_importance: [N] per-atom importance
            normalize: Min-max normalize before coloring
            threshold: Atoms below this are not highlighted

        Returns:
            PIL Image or None if rendering fails
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        rdDepictor.Compute2DCoords(mol)
        num_atoms = mol.GetNumAtoms()

        importance = node_importance.numpy()

        # Match length
        if len(importance) < num_atoms:
            importance = np.pad(importance, (0, num_atoms - len(importance)))
        else:
            importance = importance[:num_atoms]

        # Normalize
        if normalize and importance.max() > importance.min():
            importance = (importance - importance.min()) / (
                importance.max() - importance.min() + 1e-8
            )

        # Build highlight dicts for RDKit
        atom_colors = {}
        highlight_atoms = []

        for i in range(num_atoms):
            color = _importance_to_rgb(importance[i], self.colormap)
            atom_colors[i] = color
            if importance[i] >= threshold:
                highlight_atoms.append(i)

        bond_colors = {}
        highlight_bonds = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            avg = (importance[i] + importance[j]) / 2.0
            bond_colors[bond.GetIdx()] = _importance_to_rgb(avg, self.colormap)
            if avg >= threshold:
                highlight_bonds.append(bond.GetIdx())

        return self._render(
            mol,
            highlight_atoms,
            atom_colors,
            highlight_bonds,
            bond_colors,
        )

    def draw_edge_importance(
        self,
        smiles: str,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        normalize: bool = True,
    ) -> Optional[Image.Image]:
        """
        Draw molecule with bonds colored by edge mask importance.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        rdDepictor.Compute2DCoords(mol)

        mask = edge_mask.numpy()
        if normalize and mask.max() > mask.min():
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        src, dst = edge_index[0].numpy(), edge_index[1].numpy()
        bond_importance = {}

        for k in range(len(mask)):
            rdkit_bond = mol.GetBondBetweenAtoms(int(src[k]), int(dst[k]))
            if rdkit_bond:
                bid = rdkit_bond.GetIdx()
                bond_importance[bid] = max(
                    bond_importance.get(bid, 0.0), float(mask[k])
                )

        # Build atom importance from bonds
        atom_importance = {}
        for bid, imp in bond_importance.items():
            bond = mol.GetBondWithIdx(bid)
            for aidx in [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]:
                atom_importance[aidx] = max(
                    atom_importance.get(aidx, 0.0), imp * 0.7
                )

        atom_colors = {
            i: _importance_to_rgb(atom_importance.get(i, 0.0), self.colormap)
            for i in range(mol.GetNumAtoms())
        }
        bond_colors = {
            bid: _importance_to_rgb(imp, self.colormap)
            for bid, imp in bond_importance.items()
        }

        highlight_atoms = list(atom_importance.keys())
        highlight_bonds = list(bond_importance.keys())

        return self._render(
            mol,
            highlight_atoms,
            atom_colors,
            highlight_bonds,
            bond_colors,
        )

    def _render(
        self,
        mol,
        highlight_atoms,
        atom_colors,
        highlight_bonds,
        bond_colors,
    ) -> Optional[Image.Image]:
        """Render RDKit molecule to PIL Image."""
        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(*self.img_size)
            opts = drawer.drawOptions()
            opts.addAtomIndices = False
            opts.addStereoAnnotation = False

            rdMolDraw2D.PrepareMolForDrawing(mol)
            drawer.DrawMolecule(
                mol,
                highlightAtoms=highlight_atoms,
                highlightAtomColors=atom_colors,
                highlightBonds=highlight_bonds,
                highlightBondColors=bond_colors,
            )
            drawer.FinishDrawing()
            png_bytes = drawer.GetDrawingText()
            return Image.open(io.BytesIO(png_bytes))

        except Exception as e:
            print(f"[MoleculeVisualizer] Rendering failed: {e}")
            return None