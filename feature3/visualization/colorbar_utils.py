"""
Standalone colorbar and legend utilities for publication figures.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Tuple, Dict


def make_importance_colorbar(
    ax: plt.Axes,
    colormap: str = 'RdYlGn',
    label: str = 'Atom Importance',
    orientation: str = 'vertical',
) -> None:
    """Add a standalone importance colorbar to an axes."""
    sm = plt.cm.ScalarMappable(
        cmap=colormap,
        norm=plt.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation=orientation, fraction=0.05)
    cbar.set_label(label, fontsize=11)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['Low', '', 'Medium', '', 'High'])


def make_prediction_legend(
    ax: plt.Axes,
    task_name: str = 'Task',
) -> None:
    """Add correct/incorrect prediction legend patches."""
    correct_patch = mpatches.Patch(color='green', label='Correct prediction')
    wrong_patch = mpatches.Patch(color='red', label='Wrong prediction')
    ax.legend(handles=[correct_patch, wrong_patch], loc='upper right',
              fontsize=9)


def importance_summary_table(
    group_importances: Dict[str, float],
    top_k: int = 15,
) -> Tuple[List[str], List[float]]:
    """
    Sort functional groups by importance for table display.

    Returns (names, scores) sorted descending.
    """
    sorted_items = sorted(
        group_importances.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    names = [item[0].replace('_', ' ').title() for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    return names, scores