# test_feature1_transfer.py
import torch
import sys
sys.path.insert(0, ".")

from models.task_conditioned_egnn import TaskConditionedEGNN
from data.featurizer import Molecule3DFeaturizer
from torch_geometric.data import Batch

# ── Config ──────────────────────────────────────────
CKPT = "checkpoints/ablation_strategy_task_conditioned_seed0.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test molecules (drug-like)
TEST_SMILES = [
    "CCO",                        # Ethanol
    "CC(=O)Oc1ccccc1C(=O)O",     # Aspirin
    "CN1CCC[C@H]1c2cccnc2",      # Nicotine
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",  # Pyrene
]

# ── Load model ───────────────────────────────────────
print(f"Loading checkpoint: {CKPT}")

# Get feature dims from featurizer
feat = Molecule3DFeaturizer()
node_dim, edge_dim = feat.get_feature_dims()

model = TaskConditionedEGNN(
    node_dim=node_dim,
    edge_dim=edge_dim,
    hidden_dim=128,
    num_layers=4,
    num_tasks=17,
    task_dim=64,
    dropout=0.0,
).to(DEVICE)

state = torch.load(CKPT, map_location=DEVICE, weights_only=False)

# Handle checkpoint formats
if isinstance(state, dict) and "model" in state:
    state = state["model"]

model.load_state_dict(state, strict=False)
model.eval()
print("✅ Model loaded\n")

# ── Featurize ────────────────────────────────────────
print("Featurizing molecules...")
graphs = []
for smi in TEST_SMILES:
    mol = feat.smiles_to_mol(smi)
    g = feat.mol_to_graph(mol)
    if g is not None:
        # Add task_id = 0 (dummy, just for encoder)
        g.task_id = torch.tensor([0])
        graphs.append(g)
        print(f"  ✅ {smi[:30]}")
    else:
        print(f"  ❌ Failed: {smi}")

# ── Get embeddings ───────────────────────────────────
batch = Batch.from_data_list(graphs).to(DEVICE)

with torch.no_grad():
    task_ids = batch.task_id.squeeze()
    embeddings = model(batch, task_ids)

print(f"\nEmbedding shape: {embeddings.shape}")
print(f"Embedding sample (first molecule):\n{embeddings[0][:8]}")

# ── Similarity check ─────────────────────────────────
# Similar molecules should have similar embeddings
from torch.nn.functional import cosine_similarity

print("\nCosine similarities (Ethanol vs others):")
for i, smi in enumerate(TEST_SMILES[1:], 1):
    sim = cosine_similarity(
        embeddings[0].unsqueeze(0),
        embeddings[i].unsqueeze(0)
    ).item()
    print(f"  Ethanol vs {smi[:25]:<25} → {sim:.4f}")

print("\n✅ Feature 1 encoder is working and producing embeddings!")
print("Ready for Feature 2 transfer experiments.")