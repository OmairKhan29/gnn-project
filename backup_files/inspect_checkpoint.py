import torch
import sys
import os

if len(sys.argv) > 1:
    paths = [sys.argv[1]]
else:
    paths = [
        f"checkpoints/ablation_strategy_task_conditioned_pcgrad_seed{seed}_resume.pt"
        for seed in [0, 1, 2]
    ]

for path in paths:
    if not os.path.exists(path):
        if len(sys.argv) > 1:
            print(f"\nCheckpoint not found: {path}")
        continue

    state = torch.load(path, weights_only=False, map_location="cpu")

    print(f"\nCheckpoint: {path}")
    print("=" * 50)
    print(f"Keys:            {list(state.keys())}")
    print(f"Epoch:           {state.get('epoch', 'N/A')}")
    print(f"Best AUC:        {state.get('best_avg_auc', 'N/A'):.4f}")
    print(f"Patience:        {state.get('patience_counter', 'N/A')}")

    history = state.get("history", {})
    print(f"\nHistory length:  {len(history.get('train_loss', []))} epochs")

    if history.get("train_loss"):
        print(f"Last train loss: {history['train_loss'][-1]:.4f}")

    if history.get("val_auc_avg"):
        print(f"Last val AUC:    {history['val_auc_avg'][-1]:.4f}")
        print(f"Best val AUC:    {max(history['val_auc_avg']):.4f} (epoch {history['val_auc_avg'].index(max(history['val_auc_avg'])) + 1})")

    if history.get("pcgrad_stats"):
        stats = history["pcgrad_stats"]
        print(f"\nPCGrad stats:    {len(stats)} epochs logged")
        last = stats[-1]
        print(f"Last conflict ratio before: {last.get('conflict_ratio_before', 0):.2%}")
        print(f"Last conflict ratio after:  {last.get('conflict_ratio_before', 0) - last.get('conflict_reduction', 0):.2%}")
