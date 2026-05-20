"""
Generate all 9 publication figures for Feature 2.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from feature2.visualization.figure_builder import generate_all_figures


def main():
    generate_all_figures()


if __name__ == "__main__":
    main()