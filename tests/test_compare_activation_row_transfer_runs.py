from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CompareActivationRowTransferRunsScriptTest(unittest.TestCase):
    def test_compare_activation_row_transfer_runs_outputs_aligned_metrics_and_plots(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "compare_activation_row_transfer_runs.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_repo = Path(tmpdir) / "temp_repo"
            (temp_repo / ".git").mkdir(parents=True)

            datasets = [
                {"name": "Dataset A", "behavior": "sycophancy"},
                {"name": "Dataset B", "behavior": "deception"},
            ]
            fieldnames = [
                "method",
                "layer_spec",
                "layer_label",
                "source_dataset",
                "target_dataset",
                "train_split",
                "eval_split",
                "train_count",
                "auroc",
                "auprc",
                "balanced_accuracy",
                "f1",
                "accuracy",
                "count",
                "completed_at",
            ]

            def write_run(
                label: str,
                method: str,
                values: dict[tuple[str, str], float],
                *,
                layer_spec: str = "14",
            ) -> Path:
                run_dir = temp_repo / "artifacts" / "runs" / label / "run"
                (run_dir / "meta").mkdir(parents=True)
                (run_dir / "results").mkdir(parents=True)
                (run_dir / "meta" / "run_manifest.json").write_text(
                    json.dumps(
                        {
                            "model_name": "meta-llama/Llama-3.2-3B-Instruct",
                            "behavior_name": "harmful-behavior",
                            "dataset_set_name": "toy-mixed",
                            "datasets": datasets,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with (run_dir / "results" / "pairwise_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    for source_dataset in ("Dataset A", "Dataset B"):
                        for target_dataset in ("Dataset A", "Dataset B"):
                            auroc = values[(source_dataset, target_dataset)]
                            writer.writerow(
                                {
                                    "method": method,
                                    "layer_spec": layer_spec,
                                    "layer_label": "L14",
                                    "source_dataset": source_dataset,
                                    "target_dataset": target_dataset,
                                    "train_split": "train",
                                    "eval_split": "eval",
                                    "train_count": 4,
                                    "auroc": auroc,
                                    "auprc": auroc,
                                    "balanced_accuracy": auroc,
                                    "f1": auroc,
                                    "accuracy": auroc,
                                    "count": 4,
                                    "completed_at": "2026-04-14T00:00:00Z",
                                }
                            )
                return run_dir

            raw_run = write_run(
                "raw",
                "activation_logistic",
                {
                    ("Dataset A", "Dataset A"): 0.70,
                    ("Dataset A", "Dataset B"): 0.55,
                    ("Dataset B", "Dataset A"): 0.52,
                    ("Dataset B", "Dataset B"): 0.72,
                },
            )
            native_run = write_run(
                "native",
                "activation_logistic_pc_projection",
                {
                    ("Dataset A", "Dataset A"): 0.74,
                    ("Dataset A", "Dataset B"): 0.60,
                    ("Dataset B", "Dataset A"): 0.58,
                    ("Dataset B", "Dataset B"): 0.75,
                },
            )
            shared_run = write_run(
                "shared",
                "activation_logistic_pc_projection",
                {
                    ("Dataset A", "Dataset A"): 0.76,
                    ("Dataset A", "Dataset B"): 0.62,
                    ("Dataset B", "Dataset A"): 0.61,
                    ("Dataset B", "Dataset B"): 0.77,
                },
                layer_spec="14__14",
            )

            env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--run",
                    "raw",
                    str(raw_run),
                    "--run",
                    "native",
                    str(native_run),
                    "--run",
                    "shared",
                    str(shared_run),
                    "--baseline-label",
                    "raw",
                    "--artifact-root",
                    str(temp_repo / "artifacts"),
                    "--run-id",
                    "comparison-run",
                ],
                cwd=str(repo_root),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("activation-row-transfer-comparison", completed.stdout)

            run_roots = sorted((temp_repo / "artifacts" / "runs" / "activation-row-transfer-comparison").rglob("comparison-run"))
            self.assertEqual(len(run_roots), 1)
            run_root = run_roots[0]

            with (run_root / "results" / "aligned_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertIn("auroc__raw", rows[0])
            self.assertIn("auroc__native", rows[0])
            self.assertIn("delta__native__minus__raw", rows[0])
            self.assertIn("delta__shared__minus__native", rows[0])

            self.assertTrue((run_root / "results" / "aggregate_summary.csv").exists())
            self.assertTrue((run_root / "results" / "plots" / "native__minus__raw__auroc_heatmap.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "shared__minus__raw__auroc_heatmap.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "shared__minus__native__auroc_heatmap.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "aggregate_summary_comparison.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pairwise_auroc_comparison.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pairwise_offdiag_auroc_comparison.png").exists())


if __name__ == "__main__":
    unittest.main()
