import torch, json, numpy as np 
ckpt1 = torch.load('checkpoints/ablation_strategy_task_conditioned_pcgrad_seed1_resume.pt', map_location='cpu', weights_only=False) 
ckpt2 = torch.load('checkpoints/ablation_strategy_task_conditioned_pcgrad_seed2_resume.pt', map_location='cpu', weights_only=False) 
print('Seed1 best_avg_auc:', ckpt1['best_avg_auc']) 
print('Seed2 best_avg_auc:', ckpt2['best_avg_auc']) 
print('Seed1 epoch:', ckpt1['epoch']) 
print('Seed2 epoch:', ckpt2['epoch']) 
print('Seed1 history keys:', list(ckpt1['history'].keys())) 
