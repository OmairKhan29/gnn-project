import json, numpy as np 
 
with open("results/ablations/ablation_strategy.json", "r") as f: 
    data = json.load(f) 
 
print("="*60) 
print("MULTI-TASK MODEL COMPARISON (3 seeds each)") 
print("="*60) 
 
baseline_mean = None 
for model, res in data["results"].items(): 
    seeds = res["per_seed"] 
    aucs = [s["test_auc_avg"] for s in seeds] 
    print(f"\n{model.upper()}") 
    print(f"  Seeds:    {len(aucs)}") 
    print(f"  Mean AUC: {np.mean(aucs):.4f}") 
    print(f"  Std AUC:  {np.std(aucs):.4f}") 
    if baseline_mean is None: 
        baseline_mean = np.mean(aucs) 
    else: 
        delta = np.mean(aucs) - baseline_mean 
        print(f"  vs baseline: {delta:+.4f} ({100*delta/baseline_mean:+.2f}%)") 
