import json
import os
import numpy as np

# Read the file
with open('scripts/generate_figures.py', 'r') as f:
    content = f.read()

# Fix fig1
old1 = '''    for model_tag in ["hard_sharing", "task_conditioned", "task_conditioned_pcgrad"]:
        aucs = []
        for seed in range(5):
            path = f"results/multitask_{model_tag}_seed{seed}.json"
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                aucs.append(data["test_auc_avg"])

        if aucs:
            strategies.append(model_tag)
            means.append(np.mean(aucs))
            stds.append(np.std(aucs))'''

new1 = '''    ablation_path = "results/ablations/ablation_strategy.json"
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            abl = json.load(f)
        for model_tag, res in abl["results"].items():
            aucs = [s["test_auc_avg"] for s in res["per_seed"]]
            strategies.append(model_tag)
            means.append(np.mean(aucs))
            stds.append(np.std(aucs))
    else:
        for model_tag in ["hard_sharing", "task_conditioned", "task_conditioned_pcgrad"]:
            aucs = []
            for seed in range(5):
                path = f"results/multitask_{model_tag}_seed{seed}.json"
                if os.path.exists(path):
                    with open(path) as f:
                        data = json.load(f)
                    aucs.append(data["test_auc_avg"])
            if aucs:
                strategies.append(model_tag)
                means.append(np.mean(aucs))
                stds.append(np.std(aucs))'''

# Fix fig2
old2 = '''    for model_tag in ["hard_sharing", "task_conditioned", "task_conditioned_pcgrad"]:
        # Average per-task results across seeds
        task_aucs_all = {}

        for seed in range(5):
            path = f"results/multitask_{model_tag}_seed{seed}.json"
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)

                for task, auc in data.get("test_auc_per_task", {}).items():
                    if task not in task_aucs_all:
                        task_aucs_all[task] = []
                    task_aucs_all[task].append(auc)

        if task_aucs_all:
            model_results[model_tag] = {
                t: np.mean(aucs) for t, aucs in task_aucs_all.items()
            }'''

new2 = '''    ablation_path = "results/ablations/ablation_strategy.json"
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            abl = json.load(f)
        for model_tag, res in abl["results"].items():
            task_aucs_all = {}
            for seed_res in res["per_seed"]:
                for task, auc in seed_res["test_auc_per_task"].items():
                    if task not in task_aucs_all:
                        task_aucs_all[task] = []
                    task_aucs_all[task].append(auc)
            model_results[model_tag] = {t: np.mean(aucs) for t, aucs in task_aucs_all.items()}
    else:
        for model_tag in ["hard_sharing", "task_conditioned", "task_conditioned_pcgrad"]:
            task_aucs_all = {}
            for seed in range(5):
                path = f"results/multitask_{model_tag}_seed{seed}.json"
                if os.path.exists(path):
                    with open(path) as f:
                        data = json.load(f)
                    for task, auc in data.get("test_auc_per_task", {}).items():
                        if task not in task_aucs_all:
                            task_aucs_all[task] = []
                        task_aucs_all[task].append(auc)
            if task_aucs_all:
                model_results[model_tag] = {t: np.mean(aucs) for t, aucs in task_aucs_all.items()}'''

# Fix fig3
old3 = '''    for model_tag, target in [
        ("hard_sharing", baseline_aucs),
        ("task_conditioned_pcgrad", improved_aucs),
    ]:
        task_aucs_all = {}
        for seed in range(5):
            path = f"results/multitask_{model_tag}_seed{seed}.json"
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                for task, auc in data.get("test_auc_per_task", {}).items():
                    if task not in task_aucs_all:
                        task_aucs_all[task] = []
                    task_aucs_all[task].append(auc)

        for t, aucs in task_aucs_all.items():
            target[t] = np.mean(aucs)'''

new3 = '''    ablation_path = "results/ablations/ablation_strategy.json"
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            abl = json.load(f)
        for model_tag, target in [("hard_sharing", baseline_aucs), ("task_conditioned_pcgrad", improved_aucs)]:
            if model_tag in abl["results"]:
                task_aucs_all = {}
                for seed_res in abl["results"][model_tag]["per_seed"]:
                    for task, auc in seed_res["test_auc_per_task"].items():
                        if task not in task_aucs_all:
                            task_aucs_all[task] = []
                        task_aucs_all[task].append(auc)
                for t, aucs in task_aucs_all.items():
                    target[t] = np.mean(aucs)
    else:
        for model_tag, target in [("hard_sharing", baseline_aucs), ("task_conditioned_pcgrad", improved_aucs)]:
            task_aucs_all = {}
            for seed in range(5):
                path = f"results/multitask_{model_tag}_seed{seed}.json"
                if os.path.exists(path):
                    with open(path) as f:
                        data = json.load(f)
                    for task, auc in data.get("test_auc_per_task", {}).items():
                        if task not in task_aucs_all:
                            task_aucs_all[task] = []
                        task_aucs_all[task].append(auc)
            for t, aucs in task_aucs_all.items():
                target[t] = np.mean(aucs)'''

# Apply fixes
c1 = content.replace(old1, new1)
c2 = c1.replace(old2, new2)
c3 = c2.replace(old3, new3)

if c3 == content:
    print('WARNING: No changes made - patterns not found!')
else:
    with open('scripts/generate_figures.py', 'w') as f:
        f.write(c3)
    print('Fix 2 done!')