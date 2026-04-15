from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class AnalyzeActivationRowTransferPCProjectionSweepScriptTest(unittest.TestCase):
    def test_analysis_script_writes_summary_plots_and_delta_heatmaps(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "analyze_activation_row_transfer_pc_projection_sweep.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_repo = Path(tmpdir) / "temp_repo"
            (temp_repo / ".git").mkdir(parents=True)

            baseline_run_dir = (
                temp_repo
                / "artifacts"
                / "runs"
                / "activation-row-transfer"
                / "meta-llama-llama-3-2-3b-instruct"
                / "deception"
                / "quantity-v2-deception-six"
                / "mean_response-layer-14"
                / "raw-run"
            )
            sweep_run_dir = (
                temp_repo
                / "artifacts"
                / "runs"
                / "activation-row-transfer-pc-projection-sweep"
                / "meta-llama-llama-3-2-3b-instruct"
                / "deception"
                / "quantity-v2-deception-six"
                / "quantity-axis-v2-cumulative-pc-sweep"
                / "mean_response-layer-14-cumulative-pcs-1-to-2"
                / "sweep-run"
            )
            for run_dir in (baseline_run_dir, sweep_run_dir):
                (run_dir / "meta").mkdir(parents=True)
                (run_dir / "results" / "plots").mkdir(parents=True)

            baseline_manifest = {
                "model_name": "meta-llama/Llama-3.2-3B-Instruct",
                "behavior_name": "deception",
                "dataset_set_name": "quantity-v2-deception-six",
                "datasets": [
                    {"name": "Deception-ConvincingGame-completion", "behavior": "deception"},
                    {"name": "Deception-HarmPressureChoice-completion", "behavior": "deception"},
                ],
            }
            (baseline_run_dir / "meta" / "run_manifest.json").write_text(
                json.dumps(baseline_manifest) + "\n",
                encoding="utf-8",
            )
            baseline_rows = [
                {
                    "method": "activation_logistic",
                    "layer_spec": "14",
                    "layer_label": "L14",
                    "source_dataset": "Deception-ConvincingGame-completion",
                    "target_dataset": "Deception-ConvincingGame-completion",
                    "auroc": 0.90,
                    "auprc": 0.90,
                    "balanced_accuracy": 0.80,
                    "f1": 0.80,
                    "accuracy": 0.80,
                    "count": 10,
                },
                {
                    "method": "activation_logistic",
                    "layer_spec": "14",
                    "layer_label": "L14",
                    "source_dataset": "Deception-ConvincingGame-completion",
                    "target_dataset": "Deception-HarmPressureChoice-completion",
                    "auroc": 0.55,
                    "auprc": 0.55,
                    "balanced_accuracy": 0.55,
                    "f1": 0.55,
                    "accuracy": 0.55,
                    "count": 10,
                },
                {
                    "method": "activation_logistic",
                    "layer_spec": "14",
                    "layer_label": "L14",
                    "source_dataset": "Deception-HarmPressureChoice-completion",
                    "target_dataset": "Deception-ConvincingGame-completion",
                    "auroc": 0.58,
                    "auprc": 0.58,
                    "balanced_accuracy": 0.58,
                    "f1": 0.58,
                    "accuracy": 0.58,
                    "count": 10,
                },
                {
                    "method": "activation_logistic",
                    "layer_spec": "14",
                    "layer_label": "L14",
                    "source_dataset": "Deception-HarmPressureChoice-completion",
                    "target_dataset": "Deception-HarmPressureChoice-completion",
                    "auroc": 0.75,
                    "auprc": 0.75,
                    "balanced_accuracy": 0.75,
                    "f1": 0.75,
                    "accuracy": 0.75,
                    "count": 10,
                },
            ]
            with (baseline_run_dir / "results" / "pairwise_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(baseline_rows[0].keys()))
                writer.writeheader()
                writer.writerows(baseline_rows)
            (baseline_run_dir / "results" / "results.json").write_text(
                json.dumps({"datasets": [row["name"] for row in baseline_manifest["datasets"]]}) + "\n",
                encoding="utf-8",
            )
            (baseline_run_dir / "results" / "plots" / "activation_logistic__14__auroc_heatmap.png").write_text(
                "baseline-heatmap",
                encoding="utf-8",
            )

            sweep_manifest = {
                "model_name": "meta-llama/Llama-3.2-3B-Instruct",
                "behavior_name": "deception",
                "dataset_set_name": "quantity-v2-deception-six",
                "axis_slug": "quantity-axis-v2-cumulative-pc-sweep",
                "datasets": [
                    {"name": "Deception-ConvincingGame-completion", "behavior": "deception"},
                    {"name": "Deception-HarmPressureChoice-completion", "behavior": "deception"},
                ],
            }
            (sweep_run_dir / "meta" / "run_manifest.json").write_text(
                json.dumps(sweep_manifest) + "\n",
                encoding="utf-8",
            )
            sweep_rows = [
                {
                    "method": "activation_logistic_pc_projection",
                    "selected_pc_count": 1,
                    "layer_spec": "14",
                    "layer_label": "14",
                    "source_dataset": "Deception-ConvincingGame-completion",
                    "target_dataset": "Deception-ConvincingGame-completion",
                    "auroc": 0.85,
                    "auprc": 0.85,
                    "balanced_accuracy": 0.75,
                    "f1": 0.75,
                    "accuracy": 0.75,
                    "count": 10,
                },
                {
                    "method": "activation_logistic_pc_projection",
                    "selected_pc_count": 1,
                    "layer_spec": "14",
                    "layer_label": "14",
                    "source_dataset": "Deception-ConvincingGame-completion",
                    "target_dataset": "Deception-HarmPressureChoice-completion",
                    "auroc": 0.65,
                    "auprc": 0.65,
                    "balanced_accuracy": 0.65,
                    "f1": 0.65,
                    "accuracy": 0.65,
                    "count": 10,
                },
                {
                    "method": "activation_logistic_pc_projection",
                    "selected_pc_count": 1,
                    "layer_spec": "14",
                    "layer_label": "14",
                    "source_dataset": "Deception-HarmPressureChoice-completion",
                    "target_dataset": "Deception-ConvincingGame-completion",
                    "auroc": 0.64,
                    "auprc": 0.64,
                    "balanced_accuracy": 0.64,
                    "f1": 0.64,
                    "accuracy": 0.64,
                    "count": 10,
                },
                {
                    "method": "activation_logistic_pc_projection",
                    "selected_pc_count": 1,
                    "layer_spec": "14",
                    "layer_label": "14",
                    "source_dataset": "Deception-HarmPressureChoice-completion",
                    "target_dataset": "Deception-HarmPressureChoice-completion",
                    "auroc": 0.70,
                    "auprc": 0.70,
                    "balanced_accuracy": 0.70,
                    "f1": 0.70,
                    "accuracy": 0.70,
                    "count": 10,
                },
                {
                    "method": "activation_logistic_pc_projection",
                    "selected_pc_count": 2,
                    "layer_spec": "14",
                    "layer_label": "14",
                    "source_dataset": "Deception-ConvincingGame-completion",
                    "target_dataset": "Deception-ConvincingGame-completion",
                    "auroc": 0.82,
                    "auprc": 0.82,
                    "balanced_accuracy": 0.72,
                    "f1": 0.72,
                    "accuracy": 0.72,
                    "count": 10,
                },
                {
                    "method": "activation_logistic_pc_projection",
                    "selected_pc_count": 2,
                    "layer_spec": "14",
                    "layer_label": "14",
                    "source_dataset": "Deception-ConvincingGame-completion",
                    "target_dataset": "Deception-HarmPressureChoice-completion",
                    "auroc": 0.69,
                    "auprc": 0.69,
                    "balanced_accuracy": 0.69,
                    "f1": 0.69,
                    "accuracy": 0.69,
                    "count": 10,
                },
                {
                    "method": "activation_logistic_pc_projection",
                    "selected_pc_count": 2,
                    "layer_spec": "14",
                    "layer_label": "14",
                    "source_dataset": "Deception-HarmPressureChoice-completion",
                    "target_dataset": "Deception-ConvincingGame-completion",
                    "auroc": 0.68,
                    "auprc": 0.68,
                    "balanced_accuracy": 0.68,
                    "f1": 0.68,
                    "accuracy": 0.68,
                    "count": 10,
                },
                {
                    "method": "activation_logistic_pc_projection",
                    "selected_pc_count": 2,
                    "layer_spec": "14",
                    "layer_label": "14",
                    "source_dataset": "Deception-HarmPressureChoice-completion",
                    "target_dataset": "Deception-HarmPressureChoice-completion",
                    "auroc": 0.68,
                    "auprc": 0.68,
                    "balanced_accuracy": 0.68,
                    "f1": 0.68,
                    "accuracy": 0.68,
                    "count": 10,
                },
            ]
            with (sweep_run_dir / "results" / "pairwise_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0].keys()))
                writer.writeheader()
                writer.writerows(sweep_rows)
            (sweep_run_dir / "results" / "results.json").write_text(
                json.dumps({"axis_slug": "quantity-axis-v2-cumulative-pc-sweep", "k_values": [1, 2]}) + "\n",
                encoding="utf-8",
            )

            env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--baseline-run-dir",
                    str(baseline_run_dir),
                    "--pc-sweep-run-dir",
                    str(sweep_run_dir),
                    "--artifact-root",
                    str(temp_repo / "artifacts"),
                    "--run-id",
                    "analysis-run",
                ],
                cwd=str(repo_root),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("completed 2 k values", completed.stdout)

            run_roots = sorted(
                (temp_repo / "artifacts" / "runs" / "activation-row-transfer-pc-projection-sweep-analysis").rglob(
                    "analysis-run"
                )
            )
            self.assertEqual(len(run_roots), 1)
            run_root = run_roots[0]

            summary_df = pd.read_csv(run_root / "results" / "summary_by_k.csv")
            self.assertEqual(summary_df["selected_pc_count"].tolist(), [1, 2])

            self.assertTrue((run_root / "results" / "plots" / "summary_vs_k.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pareto_offdiag_gain_vs_diag_drop.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pair_delta_vs_k_heatmap.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "source_target_delta_vs_k.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "k-001__delta_vs_raw_heatmap.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "k-002__delta_vs_raw_heatmap.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "baseline__auroc_heatmap.png").exists())


if __name__ == "__main__":
    unittest.main()
