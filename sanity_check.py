import torch
from data.multitask_dataset import MultiTaskDataset, get_num_tasks
from models.task_conditioned_egnn import MultiTaskClassifier
from training.pcgrad import PCGradOptimizer
from torch.optim import Adam
from torch_geometric.loader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds = MultiTaskDataset(['bbbp', 'bace', 'clintox'], 'train')
loader = DataLoader(ds, batch_size=64, shuffle=True)
batch = next(iter(loader)).to(device)

num_tasks = get_num_tasks()
model = MultiTaskClassifier(129, 6, 64, 2, num_tasks, 32).to(device)
opt = PCGradOptimizer(Adam(model.parameters()), log_stats=True)

losses = list(model.compute_per_task_losses(batch).values())
print(f'Num tasks in batch: {len(losses)}')
opt.zero_grad()
stats = opt.backward(losses)
opt.step()

print(f'Device: {device}')
if stats:
    print(f'Conflict ratio: {stats["conflict_ratio_before"]:.2%}')
    print(f'Num conflicts: {stats["num_conflicts_before"]}/{stats["num_pairs"]} pairs')
else:
    print('Stats: None (need >1 task in batch)')
print('PCGrad sanity check: PASSED')
