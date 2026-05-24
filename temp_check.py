import torch, json 
ckpt = torch.load('checkpoints/ablation_strategy_task_conditioned_pcgrad_seed1_resume.pt', map_location='cpu', weights_only=False) 
print(list(ckpt.keys())) 
