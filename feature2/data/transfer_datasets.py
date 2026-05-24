"""
Transfer Learning Dataset Loader for Feature 2.
Loads SIDER and MUV from CSV files.
Reuses Feature 1 featurizer and scaffold splitter.
"""

import os
import ssl
import urllib.request
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Reuse Feature 1 components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.featurizer import Molecule3DFeaturizer
from data.moleculenet import ScaffoldSplitter


# ─────────────────────────────────────────────
# Dataset Metadata
# ─────────────────────────────────────────────

TRANSFER_DATASET_INFO = {
    "sider": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
        "filename": "sider.csv.gz",
        "smiles_col": "smiles",
        "num_tasks": 27,
        "task_type": "classification",
        "description": "Side-effect resource — 27 adverse drug reaction tasks",
        "task_names": [
            "Hepatobiliary disorders",
            "Metabolism and nutrition disorders",
            "Product issues",
            "Eye disorders",
            "Investigations",
            "Musculoskeletal and connective tissue disorders",
            "Gastrointestinal disorders",
            "Social circumstances",
            "Immune system disorders",
            "Reproductive system and breast disorders",
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
            "General disorders and administration site conditions",
            "Endocrine disorders",
            "Surgical and medical procedures",
            "Vascular disorders",
            "Blood and lymphatic system disorders",
            "Skin and subcutaneous tissue disorders",
            "Congenital, familial and genetic disorders",
            "Infections and infestations",
            "Respiratory, thoracic and mediastinal disorders",
            "Psychiatric disorders",
            "Renal and urinary disorders",
            "Pregnancy, puerperium and perinatal conditions",
            "Ear and labyrinth disorders",
            "Cardiac disorders",
            "Nervous system disorders",
            "Injury, poisoning and procedural complications",
        ],
    },
    "muv": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz",
        "filename": "muv.csv.gz",
        "smiles_col": "smiles",
        "num_tasks": 17,
        "task_type": "classification",
        "description": "Maximum Unbiased Validation — 17 bioactivity tasks",
        "task_names": [
            "MUV-466", "MUV-548", "MUV-600", "MUV-644", "MUV-652",
            "MUV-689", "MUV-692", "MUV-712", "MUV-713", "MUV-733",
            "MUV-737", "MUV-810", "MUV-832", "MUV-846", "MUV-852",
            "MUV-858", "MUV-859",
        ],
    },
}


# ─────────────────────────────────────────────
# Dataset Downloader
# ─────────────────────────────────────────────

def download_transfer_dataset(
    dataset_name: str,
    data_dir: str = "data/transfer",
    verbose: bool = True,
) -> str:
    """
    Download a transfer dataset CSV to data_dir.
    Returns path to downloaded file.
    """
    os.makedirs(data_dir, exist_ok=True)

    info = TRANSFER_DATASET_INFO[dataset_name]
    url = info["url"]
    filename = info["filename"]
    filepath = os.path.join(data_dir, filename)

    if os.path.exists(filepath):
        if verbose:
            print(f"[download] {dataset_name} already exists at {filepath}")
        return filepath

    if verbose:
        print(f"[download] Downloading {dataset_name} from {url}")

    # SSL workaround (same as Feature 1)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(url, context=ctx) as response:
            data = response.read()
        with open(filepath, "wb") as f:
            f.write(data)
        if verbose:
            print(f"[download] Saved to {filepath}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {dataset_name}.\n"
            f"URL: {url}\n"
            f"Error: {e}\n"
            f"Manual fix: Download from {url} and place at {filepath}"
        )

    return filepath


# ─────────────────────────────────────────────
# CSV Loader
# ─────────────────────────────────────────────

def load_transfer_dataset_csv(
    dataset_name: str,
    data_dir: str = "data/transfer",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load transfer dataset as DataFrame.
    Handles .csv and .csv.gz formats.
    """
    info = TRANSFER_DATASET_INFO[dataset_name]
    filename = info["filename"]
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        filepath = download_transfer_dataset(dataset_name, data_dir, verbose)

    if verbose:
        print(f"[loader] Loading {dataset_name} from {filepath}")

    df = pd.read_csv(filepath, compression="gzip" if filename.endswith(".gz") else None)

    # Standardize label encoding
    task_names = info["task_names"]
    for col in task_names:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace NaN with -1 (missing label convention from Feature 1)
    df[task_names] = df[task_names].fillna(-1)

    if verbose:
        valid_smiles = df[info["smiles_col"]].notna().sum()
        print(f"[loader] {dataset_name}: {len(df)} molecules, "
              f"{len(task_names)} tasks, {valid_smiles} valid SMILES")

    return df


# ─────────────────────────────────────────────
# Transfer PyG Dataset
# ─────────────────────────────────────────────

class TransferDataset(Dataset):
    """
    PyG-compatible Dataset for SIDER / MUV transfer evaluation.

    Each item is a PyG Data object with:
        - x: node features [N, 129]
        - edge_index: [2, E]
        - edge_attr: edge features [E, 6]
        - pos: 3D coordinates [N, 3]
        - y: labels [num_tasks] (float, -1 = missing)
        - smiles: SMILES string
        - dataset_name: source dataset
    """

    def __init__(
        self,
        dataset_name: str,
        smiles_list: List[str],
        labels: np.ndarray,          # [N_mol, num_tasks]
        task_names: List[str],
        featurizer: Optional[Molecule3DFeaturizer] = None,
        verbose: bool = True,
    ):
        self.dataset_name = dataset_name
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.featurizer = featurizer or Molecule3DFeaturizer()

        # Featurize all molecules
        self.data_list = []
        failed = 0

        for i, smiles in enumerate(smiles_list):
            try:
                graph = self.featurizer.featurize(smiles)
                if graph is None:
                    failed += 1
                    continue

                # Attach labels
                graph.y = torch.tensor(labels[i], dtype=torch.float32)
                graph.smiles = smiles
                graph.dataset_name = dataset_name
                graph.dataset_id = self._dataset_to_id(dataset_name)

                self.data_list.append(graph)

            except Exception:
                failed += 1
                continue

        if verbose:
            print(f"[TransferDataset] {dataset_name}: "
                  f"{len(self.data_list)} graphs, {failed} failed")

    def _dataset_to_id(self, name: str) -> int:
        mapping = {"sider": 0, "muv": 1}
        return mapping.get(name, -1)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]

    def get_labels(self) -> np.ndarray:
        """Return all labels as numpy array [N, num_tasks]."""
        return np.stack([d.y.numpy() for d in self.data_list])

    def get_task_names(self) -> List[str]:
        return self.task_names


# ─────────────────────────────────────────────
# Dataset Factory
# ─────────────────────────────────────────────

def create_transfer_datasets(
    dataset_name: str,
    data_dir: str = "data/transfer",
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    verbose: bool = True,
) -> Tuple[TransferDataset, TransferDataset, TransferDataset]:
    """
    Load and scaffold-split a transfer dataset.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    info = TRANSFER_DATASET_INFO[dataset_name]
    df = load_transfer_dataset_csv(dataset_name, data_dir, verbose)

    smiles_col = info["smiles_col"]
    task_names = info["task_names"]

    # Validate SMILES column
    df = df.dropna(subset=[smiles_col])
    smiles_list = df[smiles_col].tolist()
    labels = df[task_names].values.astype(np.float32)

    # Scaffold split (reuse Feature 1 splitter)
    splitter = ScaffoldSplitter()
    train_idx, val_idx, test_idx = splitter.split(
        smiles_list,
        frac_train=split_ratios[0],
        frac_valid=split_ratios[1],
        frac_test=split_ratios[2],
    )

    if verbose:
        print(f"[split] {dataset_name}: "
              f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    featurizer = Molecule3DFeaturizer()

    def build_subset(indices):
        sub_smiles = [smiles_list[i] for i in indices]
        sub_labels = labels[indices]
        return TransferDataset(
            dataset_name=dataset_name,
            smiles_list=sub_smiles,
            labels=sub_labels,
            task_names=task_names,
            featurizer=featurizer,
            verbose=verbose,
        )

    return build_subset(train_idx), build_subset(val_idx), build_subset(test_idx)


# ─────────────────────────────────────────────
# Dataset Info Utilities
# ─────────────────────────────────────────────

def get_transfer_dataset_info(dataset_name: str) -> Dict:
    """Return metadata for a transfer dataset."""
    if dataset_name not in TRANSFER_DATASET_INFO:
        raise ValueError(
            f"Unknown transfer dataset: {dataset_name}. "
            f"Available: {list(TRANSFER_DATASET_INFO.keys())}"
        )
    return TRANSFER_DATASET_INFO[dataset_name]


def get_transfer_task_names(dataset_name: str) -> List[str]:
    return TRANSFER_DATASET_INFO[dataset_name]["task_names"]


def get_num_transfer_tasks(dataset_name: str) -> int:
    return TRANSFER_DATASET_INFO[dataset_name]["num_tasks"]