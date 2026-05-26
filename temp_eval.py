import torch, json, numpy as np 
import sys 
sys.path.insert(0, '.') 
from data.multitask_dataset import MultiTaskDataset, get_all_task_names, get_num_tasks 
from models.task_conditioned_egnn import MultiTaskClassifier 
from training.multitask_trainer import MultiTaskTrainer 
from torch_geometric.loader import DataLoader 
 
device = 'cuda' 
datasets = ['bbbp', 'bace', 'hiv', 'clintox', 'tox21'] 
all_task_names = get_all_task_names() 
 
test_loaders = {} 
for ds_name in datasets: 
    split_ds = MultiTaskDataset([ds_name], 'test') 
    for task_id, indices in split_ds.task_to_samples.items(): 
        graphs = [split_ds.samples[i][0] for i in indices] 
        if len(graphs) > 0: 
            test_loaders[task_id] = DataLoader(graphs, batch_size=64, shuffle=False) 
 
for seed in [1, 2]: 
    ckpt_path = f'checkpoints/ablation_strategy_task_conditioned_pcgrad_seed{seed}_resume.pt' 
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False) 
    sample = MultiTaskDataset(datasets, 'train')[0] 
    model = MultiTaskClassifier(node_dim=sample.x.size(1), edge_dim=sample.edge_attr.size(1), hidden_dim=128, num_layers=4, num_tasks=get_num_tasks(), task_dim=64, dropout=0.1) 
    model.load_state_dict(ckpt['model']) 
    model = model.to(device) 
    trainer = MultiTaskTrainer(model=model, train_loader=None, val_loaders={}, cfg={}, device=device, ckpt_path=ckpt_path, use_pcgrad=True) 
    test_aucs = {} 
    for task_id, loader in test_loaders.items(): 
        auc = trainer.eval_task(task_id, loader) 
        test_aucs[all_task_names[task_id]] = auc 
    valid_aucs = [v for v in test_aucs.values() if v == v] 
    avg = sum(valid_aucs) / len(valid_aucs) 
    print(f'Seed {seed} test_auc_avg: {avg}') 
    print(f'Seed {seed} per_task: {json.dumps(test_aucs, indent=2)}') 
