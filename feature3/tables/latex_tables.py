"""
LaTeX table generator for Feature 3 results.
Produces publication-ready tables.
"""

import os
import numpy as np
from typing import Dict, List, Optional


class LatexTableGenerator:
    """
    Generates LaTeX tables for Feature 3.

    Tables:
        1. Explanation metrics by task (main result)
        2. Top substructures per task
        3. Hyperparameter ablation
        4. Dataset substructure frequency
        5. Stability across tasks
    """

    def __init__(self, output_dir: str = 'results/feature3/tables'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _save(self, latex: str, filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            f.write(latex)
        print(f"  [Table] Saved: {path}")
        return path

    # ── Table 1: Main Explanation Metrics ───────────────────────────────

    def table1_explanation_metrics(
        self,
        metrics_by_task: Dict[str, Dict],
    ) -> str:
        """
        Main result table.
        Rows: tasks. Cols: Fidelity+, Fidelity-, Sparsity, Stability.
        """
        rows_latex = []
        for task, m in metrics_by_task.items():
            fp = m.get('fidelity_plus_mean', 0)
            fp_s = m.get('fidelity_plus_std', 0)
            fm = m.get('fidelity_minus_mean', 0)
            fm_s = m.get('fidelity_minus_std', 0)
            sp = m.get('sparsity_mean', 0)
            sp_s = m.get('sparsity_std', 0)
            st = m.get('stability_mean', 0)
            st_s = m.get('stability_std', 0)
            rows_latex.append(
                f"{task.replace('_', '-')} & "
                f"${fp:.3f} \\pm {fp_s:.3f}$ & "
                f"${fm:.3f} \\pm {fm_s:.3f}$ & "
                f"${sp:.3f} \\pm {sp_s:.3f}$ & "
                f"${st:.3f} \\pm {st_s:.3f}$ \\\\"
            )

        latex = (
            r"\begin{table}[htbp]" + "\n"
            r"\centering" + "\n"
            r"\caption{GNNExplainer Explanation Quality by Task. "
            r"Fidelity$^+$: important subgraph captures prediction; "
            r"Fidelity$^-$: removing important edges changes prediction; "
            r"Sparsity: fraction of unimportant edges; "
            r"Stability: cross-run consistency.}" + "\n"
            r"\label{tab:explanation_metrics}" + "\n"
            r"\begin{tabular}{lcccc}" + "\n"
            r"\toprule" + "\n"
            r"\textbf{Task} & "
            r"\textbf{Fidelity}$^+$ $\uparrow$ & "
            r"\textbf{Fidelity}$^-$ $\uparrow$ & "
            r"\textbf{Sparsity} $\uparrow$ & "
            r"\textbf{Stability} $\uparrow$ \\" + "\n"
            r"\midrule" + "\n" +
            "\n".join(rows_latex) + "\n"
            r"\bottomrule" + "\n"
            r"\end{tabular}" + "\n"
            r"\end{table}"
        )

        self._save(latex, 'table1_explanation_metrics.tex')
        return latex

    # ── Table 2: Top Substructures Per Task ─────────────────────────────

    def table2_top_substructures(
        self,
        task_substructures: Dict[str, List],
        top_k: int = 5,
    ) -> str:
        """
        Table of top functional groups per task.
        """
        tasks = list(task_substructures.keys())

        header_cols = " & ".join(
            f"\\textbf{{{t.replace('_', '-')}}}"
            for t in tasks
        )
        col_fmt = "l" + "l" * len(tasks)

        rows = []
        for rank in range(top_k):
            row_parts = [f"\\textbf{{Top-{rank+1}}}"]
            for task in tasks:
                subs = task_substructures[task]
                if rank < len(subs):
                    name, score = subs[rank]
                    row_parts.append(
                        f"{name.replace('_', ' ').title()} ({score:.2f})"
                    )
                else:
                    row_parts.append("---")
            rows.append(" & ".join(row_parts) + r" \\")

        latex = (
            r"\begin{table}[htbp]" + "\n"
            r"\centering" + "\n"
            r"\caption{Top Functional Groups by Mean Importance Score per Task.}" + "\n"
            r"\label{tab:top_substructures}" + "\n"
            r"\begin{tabular}{" + col_fmt + "}" + "\n"
            r"\toprule" + "\n"
            r"\textbf{Rank} & " + header_cols + r" \\" + "\n"
            r"\midrule" + "\n" +
            "\n".join(rows) + "\n"
            r"\bottomrule" + "\n"
            r"\end{tabular}" + "\n"
            r"\end{table}"
        )

        self._save(latex, 'table2_top_substructures.tex')
        return latex

    # ── Table 3: Ablation Study ──────────────────────────────────────────

    def table3_ablation(
        self,
        ablation_results: Dict[str, Dict],
    ) -> str:
        """
        Ablation: effect of GNNExplainer hyperparameters.
        Rows: configs. Cols: edge_size, entropy, epochs, Fidelity+, Sparsity.
        """
        rows = []
        best_score = -1
        best_cfg = None

        for cfg, result in ablation_results.items():
            params = result.get('params', {})
            fp = result.get('fidelity_plus_mean', 0)
            fm = result.get('fidelity_minus_mean', 0)
            sp = result.get('sparsity_mean', 0)
            score = fp + fm + sp

            if score > best_score:
                best_score = score
                best_cfg = cfg

            rows.append((cfg, params, fp, fm, sp, score))

        latex_rows = []
        for cfg, params, fp, fm, sp, score in rows:
            prefix = r"\rowcolor{gray!15} " if cfg == best_cfg else ""
            suffix = r" $\star$" if cfg == best_cfg else ""
            latex_rows.append(
                prefix +
                f"{cfg} & "
                f"{params.get('edge_size', '---')} & "
                f"{params.get('edge_entropy', '---')} & "
                f"{params.get('epochs', '---')} & "
                f"${fp:.3f}$ & "
                f"${fm:.3f}$ & "
                f"${sp:.3f}$" +
                suffix + r" \\"
            )

        latex = (
            r"\begin{table}[htbp]" + "\n"
            r"\centering" + "\n"
            r"\caption{GNNExplainer Hyperparameter Ablation. "
            r"$\star$ denotes best configuration.}" + "\n"
            r"\label{tab:ablation}" + "\n"
            r"\begin{tabular}{lrrrrrr}" + "\n"
            r"\toprule" + "\n"
            r"\textbf{Config} & \textbf{edge\_size} & "
            r"\textbf{entropy} & \textbf{epochs} & "
            r"\textbf{Fid$^+$} & \textbf{Fid$^-$} & "
            r"\textbf{Sparsity} \\" + "\n"
            r"\midrule" + "\n" +
            "\n".join(latex_rows) + "\n"
            r"\bottomrule" + "\n"
            r"\end{tabular}" + "\n"
            r"\end{table}"
        )

        self._save(latex, 'table3_ablation.tex')
        return latex