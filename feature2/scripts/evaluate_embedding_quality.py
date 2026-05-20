"""
Evaluate embedding quality after alignment training.

Computes:
    - Inter-dataset cosine similarity matrix
    - Cluster separation score (Davies-Bouldin / silhouette)
    - t-SNE / UMAP visualization
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from feature2.models.pretrained_encoder import (
    load_feature1_checkpoint, FrozenEncoder
)
from feature2.evaluation.embedding_extractor import (
    EmbeddingExtractor, compute_inter_dataset_similarity
)
from data.dataset import MoleculeDataset


def cluster_separation_score(
    embeddings: np.ndarray,
    labels: list,
) -> dict:
    """Compute silhouette and Davies-Bouldin scores."""
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from sklearn.preprocessing import LabelEncoder

    if len(set(labels)) < 2:
        return {"silhouette": 0.0, "davies_bouldin": 0.0}

    le = LabelEncoder()
    y = le.fit_transform(labels)

    try:
        sil = float(silhouette_score(embeddings, y, sample_size=min(1000, len(y))))
    except Exception:
        sil = 0.0
    try:
        db = float(davies_bouldin_score(embeddings, y))
    except Exception:
        db = 0.0

    return {"silhouette": sil, "davies_bouldin": db}


def plot_tsne(
    embeddings: np.ndarray,
    labels: list,
    save_path: str,
    title: str = "t-SNE",
):
    """t-SNE plot colored by dataset."""
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="random")
    proj = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, lbl in enumerate(unique_labels):
        idx = [j for j, l in enumerate(labels) if l == lbl]
        ax.scatter(proj[idx, 0], proj[idx, 1], c=[colors[i]],
                   label=lbl, s=15, alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved t-SNE: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="task_conditioned")
    parser.add_argument("--datasets", nargs="+",
                        default=["bace", "bbbp", "clintox", "tox21"])
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_name", type=str, default="default")
    parser.add_argument("--max_per_dataset", type=int, default=300)
    args = parser.parse_args()

    out_dir = "results/feature2/embeddings"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Embedding Quality Evaluation")
    print("=" * 60)

    # Load encoder
    full_model = load_feature1_checkpoint(
        args.checkpoint, args.model_type, args.device
    )
    encoder = FrozenEncoder(full_model, args.model_type)
    extractor = EmbeddingExtractor(encoder, args.device, batch_size=32)

    # Extract embeddings per dataset
    all_embeddings = []
    all_labels = []
    per_dataset_embs = {}

    for ds_name in args.datasets:
        try:
            ds = MoleculeDataset(
                dataset_name=ds_name,
                split="test",
                data_dir=args.data_dir,
            )
            # Subsample for speed
            n = min(args.max_per_dataset, len(ds))
            from torch.utils.data import Subset
            ds_sub = Subset(ds, list(range(n)))

            result = extractor.extract(ds_sub, label=ds_name)
            all_embeddings.append(result["embeddings"])
            all_labels.extend([ds_name] * result["n_molecules"])
            per_dataset_embs[ds_name] = result["embeddings"]
            print(f"  {ds_name}: {result['n_molecules']} embeddings extracted")
        except Exception as e:
            print(f"  {ds_name}: skipped ({e})")

    if not all_embeddings:
        print("No embeddings extracted.")
        return

    combined = np.concatenate(all_embeddings, axis=0)

    # Cluster scores
    print("\n[1/3] Cluster separation scores...")
    cluster_scores = cluster_separation_score(combined, all_labels)
    print(f"  Silhouette:    {cluster_scores['silhouette']:.4f}")
    print(f"  Davies-Bouldin: {cluster_scores['davies_bouldin']:.4f}")

    # Inter-dataset similarity
    print("\n[2/3] Inter-dataset cosine similarity...")
    sim_matrix, names = compute_inter_dataset_similarity(per_dataset_embs)
    print(f"  Matrix shape: {sim_matrix.shape}")
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            print(f"    sim({ni}, {nj}) = {sim_matrix[i, j]:.4f}")

    # t-SNE
    print("\n[3/3] Generating t-SNE plot...")
    tsne_path = os.path.join(out_dir, f"tsne_{args.output_name}.png")
    plot_tsne(combined, all_labels, tsne_path,
              title=f"Embedding t-SNE [{args.output_name}]")

    # Save metrics
    metrics_path = os.path.join(out_dir, f"metrics_{args.output_name}.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "cluster_scores": cluster_scores,
            "similarity_matrix": sim_matrix.tolist(),
            "dataset_names": names,
            "checkpoint": args.checkpoint,
            "n_total_embeddings": len(combined),
        }, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()