"""
Evaluate a saved checkpoint on test set and print per-task AUCs.
Usage: python eval_checkpoint.py
"""
import sys
import torch
import numpy as np

sys.path.insert(0, ".")

from data.multitask_dataset import MultiTaskDataset, get_all_task_names, get_num_tasks
from models.task_conditioned_egnn import MultiTaskClassifier
from training.multitask_trainer import MultiTaskTrainer
from torch_geometric.loader import DataLoader

CKPT = "checkpoints/ablation_strategy_task_conditioned_pcgrad_seed1.pt"
DATASETS = ["bbbp", "bace", "hiv", "clintox", "tox21"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Build model
train_ds = MultiTaskDataset([DATASETS[0]], "train")
sample = train_ds[0]
num_tasks = get_num_tasks()

model = MultiTaskClassifier(
    node_dim=sample.x.size(1),
    edge_dim=sample.edge_attr.size(1),
    hidden_dim=128,
    num_layers=4,
    num_tasks=num_tasks,
    task_dim=64,
    dropout=0.1,
)

# Load checkpoint
state = torch.load(CKPT, map_location=DEVICE, weights_only=False)
if isinstance(state, dict) and "model" in state:
    model.load_state_dict(state["model"])
else:
    model.load_state_dict(state)

model = model.to(DEVICE)
model.eval()

# Build test loaders
test_loaders = {}
for ds_name in DATASETS:
    test_ds = MultiTaskDataset([ds_name], "test")
    for task_id, indices in test_ds.task_to_samples.items():
        graphs = [test_ds.samples[i][0] for i in indices]
        if graphs:
            test_loaders[task_id] = DataLoader(graphs, batch_size=64, shuffle=False)

# Build trainer just for eval
val_loaders = {}
for ds_name in DATASETS:
    val_ds = MultiTaskDataset([ds_name], "valid")
    for task_id, indices in val_ds.task_to_samples.items():
        graphs = [val_ds.samples[i][0] for i in indices]
        if graphs:
            val_loaders[task_id] = DataLoader(graphs, batch_size=64, shuffle=False)

trainer = MultiTaskTrainer(
    model=model,
    train_loader=DataLoader(train_ds, batch_size=64),
    val_loaders=val_loaders,
    cfg={"lr": 0.001, "wd": 1e-5, "patience": 30, "epochs": 1},
    device=DEVICE,
    ckpt_path=CKPT,
)

# Eval
all_task_names = get_all_task_names()
test_aucs = {}
for task_id, loader in test_loaders.items():
    auc = trainer.eval_task(task_id, loader)
    test_aucs[all_task_names[task_id]] = auc

valid_aucs = [v for v in test_aucs.values() if v == v]
avg = np.mean(valid_aucs)

print(f"\nCheckpoint: {CKPT}")
print(f"Average Test AUC: {avg:.4f}")
print("\nPer-task AUCs:")
for task, auc in sorted(test_aucs.items()):
    print(f"  {task:30s}: {auc:.4f}")
