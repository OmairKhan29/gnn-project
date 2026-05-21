"""
LaTeX table generator for Feature 3 results.

Tables:
1. Explanation metrics by task (fidelity, sparsity, stability)
2. Top substructures per toxicity task
3. Ablation: GNNExplainer hyperparameters
4. Cross-dataset substructure importance
5. Prediction vs explanation quality correlation
"""

import os
from typing import Dict, List, Optional
import numpy as np

TABLE_DIR = 'results/feature3/tables/'


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def table_explanation_metrics(
    metrics_by_task: Dict[str, Dict],
    save: bool = True,
) -> str:
    """
    Table 1: Explanation quality metrics per task.

    Columns: Task | Fidelity+ | Fidelity- | Sparsity | Stability
    """
    header = (
        r"\begin{table}[h]" "\n"
        r"\centering" "\n"
        r"\caption{GNNExplainer Explanation Quality by Task}" "\n"
        r"\label{tab:explanation_metrics}" "\n"
        r"\begin{tabular}{lcccc}" "\n"
        r"\toprule" "\n"
        r"\textbf{Task} & \textbf{Fidelity+} $\uparrow$ & "
        r"\textbf{Fidelity-} $\uparrow$ & "
        r"\textbf{Sparsity} $\uparrow$ & "
        r"\textbf{Stability} $\uparrow$ \\" "\n"
        r"\midrule" "\n"
    )

    rows = []
    for task, metrics in metrics_by_task.items():
        fp = metrics.get('fidelity_plus_mean', 0)
        fp_s = metrics.get('fidelity_plus_std', 0)
        fm = metrics.get('fidelity_minus_mean', 0)
        fm_s = metrics.get('fidelity_minus_std', 0)
        sp = metrics.get('sparsity_mean', 0)
        sp_s = metrics.get('sparsity_std', 0)
        st = metrics.get('stability_mean', 0)
        st_s = metrics.get('stability_std', 0)

        row = (
            f"{task} & "
            f"${fp:.3f} \\pm {fp_s:.3f}$ & "
            f"${fm:.3f} \\pm {fm_s:.3f}$ & "
            f"${sp:.3f} \\pm {sp_s:.3f}$ & "
            f"${st:.3f} \\pm {st_s:.3f}$ \\\\"
        )
        rows.append(row)

    footer = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )

    latex = header + "\n".join(rows) + "\n" + footer

    if save:
        ensure_dir(TABLE_DIR)
        path = os.path.join(TABLE_DIR, 'table1_explanation_metrics.tex')
        with open(path, 'w') as f:
            f.write(latex)
        print(f"Saved: {path}")

    return latex


def table_top_substructures(
    task_substructures: Dict[str, List],
    top_k: int = 5,
    save: bool = True,
) -> str:
    """
    Table 2: Top-K most important substructures per task.
    """
    tasks = list(task_substructures.keys())
    n_tasks = len(tasks)

    header = (
        r"\begin{table}[h]" "\n"
        r"\centering" "\n"
        r"\caption{Top Functional Groups by Importance Score per Task}" "\n"
        r"\label{tab:top_substructures}" "\n"
        r"\begin{tabular}{l" + "l" * n_tasks + "}" "\n"
        r"\toprule" "\n"
        r"\textbf{Rank} & " +
        " & ".join(f"\\textbf{{{t}}}" for t in tasks) + r" \\" "\n"
        r"\midrule" "\n"
    )

    rows = []
    for rank in range(top_k):
        row_parts = [f"Top-{rank + 1}"]
        for task in tasks:
            subs = task_substructures[task]
            if rank < len(subs):
                name, score = subs[rank]
                row_parts.append(
                    f"{name.replace('_', ' ').title()} ({score:.2f})"
                )
            else:
                row_parts.append("—")
        rows.append(" & ".join(row_parts) + r" \\")

    footer = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )

    latex = header + "\n".join(rows) + "\n" + footer

    if save:
        ensure_dir(TABLE_DIR)
        path = os.path.join(TABLE_DIR, 'table2_top_substructures.tex')
        with open(path, 'w') as f:
            f.write(latex)
        print(f"Saved: {path}")

    return latex


def table_ablation_hyperparams(
    ablation_results: Dict[str, Dict],
    save: bool = True,
) -> str:
    """
    Table 3: GNNExplainer hyperparameter ablation.

    Rows: configurations (edge_size, entropy, epochs, lr)
    Cols: Fidelity+, Fidelity-, Sparsity
    """
    header = (
        r"\begin{table}[h]" "\n"
        r"\centering" "\n"
        r"\caption{GNNExplainer Hyperparameter Ablation}" "\n"
        r"\label{tab:ablation_hyperparams}" "\n"
        r"\begin{tabular}{llllccc}" "\n"
        r"\toprule" "\n"
        r"\textbf{Config} & \textbf{edge\_size} & \textbf{entropy} & "
        r"\textbf{epochs} & \textbf{Fidelity+} & "
        r"\textbf{Fidelity-} & \textbf{Sparsity} \\" "\n"
        r"\midrule" "\n"
    )

    rows = []
    best_score = -1
    best_config = None

    for config_name, results in ablation_results.items():
        fp = results.get('fidelity_plus_mean', 0)
        fm = results.get('fidelity_minus_mean', 0)
        sp = results.get('sparsity_mean', 0)
        score = fp + fm + sp

        if score > best_score:
            best_score = score
            best_config = config_name

        params = results.get('params', {})
        row = (
            f"{config_name} & "
            f"{params.get('edge_size', '—')} & "
            f"{params.get('edge_entropy', '—')} & "
            f"{params.get('epochs', '—')} & "
            f"${fp:.3f}$ & "
            f"${fm:.3f}$ & "
            f"${sp:.3f}$ \\\\"
        )
        if config_name == best_config:
            row = r"\rowcolor{gray!15} " + row + r" $\star$"
        rows.append(row)

    footer = (
        r"\bottomrule" "\n"
        r"\multicolumn{7}{l}{"
        r"\small $\star$ Best overall configuration}" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )

    latex = header + "\n".join(rows) + "\n" + footer

    if save:
        ensure_dir(TABLE_DIR)
        path = os.path.join(TABLE_DIR, 'table3_ablation_hyperparams.tex')
        with open(path, 'w') as f:
            f.write(latex)
        print(f"Saved: {path}")

    return latex