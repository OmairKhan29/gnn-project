"""
Embedding Extractor for Feature 2.
Extracts graph-level embeddings from pretrained encoder.
Used for t-SNE, UMAP, cosine similarity analysis.
"""

import torch
import numpy as np
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ─────────────────────────────────────────────
# Embedding Extractor
# ─────────────────────────────────────────────

class EmbeddingExtractor:
    """
    Extracts molecular embeddings from a trained encoder.
    Supports batch extraction across multiple datasets.
    """

    def __init__(
        self,
        encoder,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.encoder = encoder
        self.device = device
        self.batch_size = batch_size
        self.encoder.eval()

    @torch.no_grad()
    def extract(
        self,
        dataset,
        task_id: Optional[int] = None,
        label: str = "unknown",
    ) -> Dict:
        """
        Extract embeddings from all molecules in a dataset.

        Returns:
            dict with keys:
                embeddings: [N, D] numpy array
                labels: [N, T] numpy array
                smiles: list of SMILES
                dataset_label: str
        """
        loader = PyGDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        all_embeddings = []
        all_labels = []
        all_smiles = []

        for batch in loader:
            batch = batch.to(self.device)

            if hasattr(self.encoder, "model_type"):
                emb = self.encoder(batch, task_id=task_id)
            else:
                emb = self.encoder(batch)

            all_embeddings.append(emb.cpu().numpy())

            labels = batch.y
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            all_labels.append(labels.cpu().numpy())

            if hasattr(batch, "smiles"):
                smiles = batch.smiles
                if isinstance(smiles, str):
                    smiles = [smiles]
                all_smiles.extend(smiles)

        embeddings = np.concatenate(all_embeddings, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0)

        return {
            "embeddings": embeddings,
            "labels": labels_arr,
            "smiles": all_smiles,
            "dataset_label": label,
            "n_molecules": len(embeddings),
            "embedding_dim": embeddings.shape[1],
        }

    def extract_multiple_datasets(
        self,
        datasets: List,
        dataset_labels: List[str],
        task_ids: Optional[List[int]] = None,
    ) -> Dict:
        """
        Extract embeddings from multiple datasets.
        Returns combined dict for visualization.
        """
        all_embeddings = []
        all_dataset_labels = []
        all_molecule_labels = []

        for i, (dataset, label) in enumerate(zip(datasets, dataset_labels)):
            task_id = task_ids[i] if task_ids else None
            result = self.extract(dataset, task_id=task_id, label=label)

            all_embeddings.append(result["embeddings"])
            all_dataset_labels.extend([label] * result["n_molecules"])
            all_molecule_labels.append(result["labels"])

        combined_embeddings = np.concatenate(all_embeddings, axis=0)

        return {
            "embeddings": combined_embeddings,
            "dataset_labels": all_dataset_labels,
            "n_total": len(combined_embeddings),
            "embedding_dim": combined_embeddings.shape[1],
            "datasets": dataset_labels,
        }


# ─────────────────────────────────────────────
# Cosine Similarity Computation
# ─────────────────────────────────────────────

def compute_inter_dataset_similarity(
    embeddings_dict: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Compute mean cosine similarity between datasets.

    Args:
        embeddings_dict: {dataset_name: [N, D] embeddings}

    Returns:
        similarity_matrix: [num_datasets, num_datasets]
    """
    dataset_names = list(embeddings_dict.keys())
    n = len(dataset_names)
    sim_matrix = np.zeros((n, n))

    for i, name_i in enumerate(dataset_names):
        emb_i = embeddings_dict[name_i]
        # Normalize
        emb_i = emb_i / (np.linalg.norm(emb_i, axis=1, keepdims=True) + 1e-8)
        mean_i = emb_i.mean(axis=0)

        for j, name_j in enumerate(dataset_names):
            emb_j = embeddings_dict[name_j]
            emb_j = emb_j / (np.linalg.norm(emb_j, axis=1, keepdims=True) + 1e-8)
            mean_j = emb_j.mean(axis=0)

            # Cosine similarity between mean embeddings
            sim = float(np.dot(mean_i, mean_j) / (
                np.linalg.norm(mean_i) * np.linalg.norm(mean_j) + 1e-8
            ))
            sim_matrix[i, j] = sim

    return sim_matrix, dataset_names