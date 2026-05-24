"""
Ablation Experiment Runner for Feature 2 Phase 4.

Runs systematic component removal experiments:
    A1: No task conditioning (Hard Sharing baseline)
    A2: No alignment (vs each alignment strategy)
    A3: Alignment λ_weight variations (0.0, 0.1, 0.3, 0.5, 1.0)
    A4: Projection dimension variations (16, 32, 64, 128)
    A5: Temperature variations for contrastive (0.05, 0.1, 0.2, 0.5)

Each ablation runs on SIDER + MUV with multiple seeds.
"""

import os
import json
import time
import torch
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.multitask_dataset import MultiTaskDataset, get_all_task_names
from models.task_conditioned_egnn import MultiTaskClassifier, HardSharingClassifier
from feature2.alignment.alignment_trainer import AlignmentTrainer, create_alignment_module
from training.trainer import set_seed


# ─────────────────────────────────────────────
# Ablation Configurations
# ─────────────────────────────────────────────

ABLATION_CONFIGS = {
    "no_task_conditioning": {
        "description": "Replace TaskConditionedEGNN with HardSharing",
        "model_type": "hard_sharing",
        "alignment_strategy": "none",
        "task_dim": None,
    },
    "no_alignment": {
        "description": "No representation alignment (baseline)",
        "model_type": "task_conditioned",
        "alignment_strategy": "none",
    },
    "lambda_0.0": {
        "description": "Alignment weight = 0 (equivalent to no alignment)",
        "model_type": "task_conditioned",
        "alignment_strategy": "prototype",
        "alignment_weight": 0.0,
    },
    "lambda_0.1": {
        "description": "Alignment weight = 0.1 (weak regularization)",
        "model_type": "task_conditioned",
        "alignment_strategy": "prototype",
        "alignment_weight": 0.1,
    },
    "lambda_0.3": {
        "description": "Alignment weight = 0.3 (moderate)",
        "model_type": "task_conditioned",
        "alignment_strategy": "prototype",
        "alignment_weight": 0.3,
    },
    "lambda_0.5": {
        "description": "Alignment weight = 0.5 (default)",
        "model_type": "task_conditioned",
        "alignment_strategy": "prototype",
        "alignment_weight": 0.5,
    },
    "lambda_1.0": {
        "description": "Alignment weight = 1.0 (strong regularization)",
        "model_type": "task_conditioned",
        "alignment_strategy": "prototype",
        "alignment_weight": 1.0,
    },
    "proj_dim_16": {
        "description": "Contrastive projection dim = 16",
        "model_type": "task_conditioned",
        "alignment_strategy": "contrastive",
        "projection_dim": 16,
    },
    "proj_dim_32": {
        "description": "Contrastive projection dim = 32",
        "model_type": "task_conditioned",
        "alignment_strategy": "contrastive",
        "projection_dim": 32,
    },
    "proj_dim_64": {
        "description": "Contrastive projection dim = 64 (default)",
        "model_type": "task_conditioned",
        "alignment_strategy": "contrastive",
        "projection_dim": 64,
    },
    "proj_dim_128": {
        "description": "Contrastive projection dim = 128",
        "model_type": "task_conditioned",
        "alignment_strategy": "contrastive",
        "projection_dim": 128,
    },
    "temp_0.05": {
        "description": "Contrastive temperature = 0.05 (harsh)",
        "model_type": "task_conditioned",
        "alignment_strategy": "contrastive",
        "temperature": 0.05,
    },
    "temp_0.1": {
        "description": "Contrastive temperature = 0.1 (default)",
        "model_type": "task_conditioned",
        "alignment_strategy": "contrastive",
        "temperature": 0.1,
    },
    "temp_0.2": {
        "description": "Contrastive temperature = 0.2 (soft)",
        "model_type": "task_conditioned",
        "alignment_strategy": "contrastive",
        "temperature": 0.2,
    },
    "temp_0.5": {
        "description": "Contrastive temperature = 0.5 (very soft)",
        "model_type": "task_conditioned",
        "alignment_strategy": "contrastive",
        "temperature": 0.5,
    },
}


class AblationRunner:
    """
    Runs ablation experiments systematically.

    For each ablation config:
        1. Build model according to ablation settings
        2. Train on multi-task dataset (Feature 1 style)
        3. Evaluate on held-out test set
        4. Save results to JSON
    """

    def __init__(
        self,
        config: Dict,
        data_dir: str = "data/processed",
        checkpoint_dir: str = "checkpoints/feature2/ablations",
        result_dir: str = "results/feature2/ablations",
        device: str = "cpu",
        verbose: bool = True,
    ):
        self.config = config
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.device = device
        self.verbose = verbose

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

    def _build_model(self, ablation_name: str) -> Tuple[nn.Module, Optional[nn.Module]]:
        """Build model and alignment module according to ablation config."""
        ablation_cfg = ABLATION_CONFIGS.get(ablation_name, {})
        
        model_type = ablation_cfg.get("model_type", "task_conditioned")
        alignment_strategy = ablation_cfg.get("alignment_strategy", "none")
        hidden_dim = self.config.get("hidden_dim", 128)
        n_layers = self.config.get("n_layers", 4)
        task_dim = ablation_cfg.get("task_dim", self.config.get("task_dim", 64)) if ablation_cfg.get("task_dim") is not None else self.config.get("task_dim", 64)
        
        num_tasks = len(get_all_task_names())
        
        if model_type == "hard_sharing":
            task_model = HardSharingClassifier(
                node_dim=129, edge_dim=6,
                hidden_dim=hidden_dim, n_layers=n_layers,
                num_tasks=num_tasks,
            )
        else:
            task_model = MultiTaskClassifier(
                node_dim=129, edge_dim=6,
                hidden_dim=hidden_dim, n_layers=n_layers,
                num_tasks=num_tasks, task_dim=task_dim,
            )
        
        alignment_module = create_alignment_module(
            strategy=alignment_strategy,
            embedding_dim=hidden_dim,
            num_tasks=num_tasks,
            temperature=ablation_cfg.get("temperature", 0.1),
            grl_lambda=1.0,
        )
        
        if alignment_module and ablation_cfg.get("projection_dim"):
            alignment_module.projection_dim = ablation_cfg["projection_dim"]
            alignment_module.projection[-1] = nn.Linear(hidden_dim, ablation_cfg["projection_dim"])
        
        return task_model, alignment_module

    def _train_single_ablation(
        self,
        ablation_name: str,
        seed: int,
    ) -> Dict:
        """Train and evaluate single ablation config."""
        set_seed(seed)
        
        task_model, alignment_module = self._build_model(ablication_name)
        
        # Load datasets
        train_ds = MultiTaskDataset(
            split="train",
            data_dir=self.data_dir,
            datasets=["tox21", "clintox", "bbbp", "bace", "hiv"],
        )
        val_ds = MultiTaskDataset(
            split="valid",
            data_dir=self.data_dir,
            datasets=["tox21", "clintox", "bbbp", "bace", "hiv"],
        )
        test_ds = MultiTaskDataset(
            split="test",
            data_dir=self.data_dir,
            datasets=["tox21", "clintox", "bbbp", "bace", "hiv"],
        )
        
        task_names = get_all_task_names()
        
        # Training config
        ablation_cfg = ABLATION_CONFIGS.get(ablication_name, {})
        align_weight = ablation_cfg.get("alignment_weight", 0.5)
        
        train_config = {
            "lr": self.config.get("lr", 1e-3),
            "weight_decay": self.config.get("weight_decay", 1e-5),
            "batch_size": self.config.get("batch_size", 64),
            "epochs": self.config.get("epochs", 80),
            "patience": self.config.get("patience", 20),
            "grad_clip": self.config.get("grad_clip", 1.0),
            "alignment_weight": align_weight,
        }
        
        exp_name = f"abl_{ablation_name}_seed{seed}"
        
        trainer = AlignmentTrainer(
            task_model=task_model,
            alignment_module=alignment_module,
            alignment_strategy=ablation_cfg.get("alignment_strategy", "none"),
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            task_names=task_names,
            smiles_list=None,
            config=train_config,
            checkpoint_dir=self.checkpoint_dir,
            result_dir=self.result_dir,
            device=self.device,
            verbose=False,
        )
        
        results = trainer.train(experiment_name=exp_name)
        results["ablation_name"] = ablication_name
        results["seed"] = seed
        
        return results

    def run_all_ablations(
        self,
        ablation_names: List[str],
        seeds: List[int],
    ) -> Dict:
        """
        Run all ablation experiments.

        Returns:
            Nested dict: {ablation_name: {seed: results}}
        """
        all_results = {}
        total = len(ablation_names) * len(seeds)
        done = 0

        print(f"\n[Ablation Runner] Running {total} experiments...")
        print(f"  Ablations: {ablation_names}")
        print(f"  Seeds: {seeds}")
        print()

        for abl_name in ablation_names:
            all_results[abl_name] = {}
            desc = ABLATION_CONFIGS.get(abl_name, {}).get("description", "")

            for seed in seeds:
                done += 1
                print(f"  [{done}/{total}] {abl_name} [{desc[:50]}...] | seed{seed}")

                try:
                    r = self._train_single_ablation(abl_name, seed)
                    all_results[abl_name][seed] = r
                except Exception as e:
                    print(f"    FAILED: {e}")
                    all_results[abl_name][seed] = {"error": str(e)}

        # Save all results
        out_path = os.path.join(self.result_dir, "all_ablation_results.json")
        save_data = {}
        for abl_name, seed_results in all_results.items():
            seed_keys = [k for k in seed_results.keys() if isinstance(k, int)]
            if seed_keys:
                save_data[abl_name] = {
                    str(s): seed_results[s]
                    for s in sorted(seed_keys)
                }

        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2)

        print(f"\n[Ablation Runner] Results saved to {out_path}")
        return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations", nargs="+", default=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--task_dim", type=int, default=64)
    args = parser.parse_args()

    config = {
        "lr": args.lr,
        "weight_decay": 1e-5,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": 20,
        "grad_clip": 1.0,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "task_dim": args.task_dim,
    }

    runner = AblationRunner(
        config=config,
        device=args.device,
        verbose=True,
    )

    runner.run_all_ablations(args.ablations, args.seeds)


if __name__ == "__main__":
    main()