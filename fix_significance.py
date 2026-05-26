# Read the file
with open('evaluation/significance.py', 'r') as f:
    content = f.read()

old = '''    for f in results_path.glob("multitask_*.json"):
        with open(f) as fp:
            data = json.load(fp)

        model = data["model"]
        seed = data["seed"]
        auc = data["test_auc_avg"]

        if model not in model_seeds:
            model_seeds[model] = {}
        model_seeds[model][seed] = auc'''

new = '''    # Try reading from ablation strategy JSON first (has all 3 seeds)
    ablation_path = Path("results/ablations/ablation_strategy.json")
    if ablation_path.exists():
        with open(ablation_path) as fp:
            abl_data = json.load(fp)
        for model, res in abl_data["results"].items():
            model_seeds[model] = {}
            for seed_idx, seed_res in enumerate(res["per_seed"]):
                model_seeds[model][seed_idx] = seed_res["test_auc_avg"]
    else:
        for f in results_path.glob("multitask_*.json"):
            with open(f) as fp:
                data = json.load(fp)
            model = data.get("model", f.stem.rsplit("_seed", 1)[0].replace("multitask_", "", 1))
            seed = data.get("seed", 0)
            auc = data["test_auc_avg"]
            if model not in model_seeds:
                model_seeds[model] = {}
            model_seeds[model][seed] = auc'''

content = content.replace(old, new)

with open('evaluation/significance.py', 'w') as f:
    f.write(content)

print('Fixed!')