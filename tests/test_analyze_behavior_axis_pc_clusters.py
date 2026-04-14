from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class AnalyzeBehaviorAxisPCClustersScriptTest(unittest.TestCase):
    def test_analyze_behavior_axis_pc_clusters_runs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "analyze_behavior_axis_pc_clusters.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_repo = Path(tmpdir) / "temp_repo"
            (temp_repo / ".git").mkdir(parents=True)
            pc_sweep_run_dir = (
                temp_repo
                / "artifacts"
                / "runs"
                / "behavior-axis-pc-sweep"
                / "meta-llama-llama-3-2-3b-instruct"
                / "harmful-behavior"
                / "toy-mixed"
                / "quantity-sycophancy-unified-native-pc-sweep"
                / "pc-sweep-run"
            )
            (pc_sweep_run_dir / "results").mkdir(parents=True)
            (pc_sweep_run_dir / "meta").mkdir(parents=True)

            datasets = [
                ("Sycophancy Dataset", "sycophancy"),
                ("OEQ Validation", "sycophancy"),
                ("Deception-ConvincingGame-completion", "deception"),
                ("Deception-Mask-completion", "deception"),
            ]
            rows: list[dict[str, object]] = []
            pc_values = {
                1: [0.85, 0.82, 0.55, 0.58],
                2: [0.78, 0.76, 0.62, 0.64],
                3: [0.52, 0.50, 0.88, 0.84],
                4: [0.58, 0.56, 0.80, 0.79],
            }
            for pc_index, values in pc_values.items():
                for (dataset_name, behavior_name), auroc in zip(datasets, values, strict=False):
                    rows.append(
                        {
                            "method": f"pc{pc_index}_zero_shot",
                            "pc_index": pc_index,
                            "pc_label": f"PC{pc_index}",
                            "layer_spec": "14",
                            "layer_label": "L14",
                            "source_dataset": "__zero_shot__",
                            "target_dataset": dataset_name,
                            "target_behavior": behavior_name,
                            "train_split": "axis",
                            "eval_split": "eval",
                            "auroc": auroc,
                            "auprc": auroc,
                            "balanced_accuracy": auroc,
                            "f1": auroc,
                            "accuracy": auroc,
                            "count": 100,
                            "completed_at": "2026-04-14T00:00:00+00:00",
                        }
                    )

            with (pc_sweep_run_dir / "results" / "pairwise_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            (pc_sweep_run_dir / "results" / "results.json").write_text(
                json.dumps(
                    {
                        "datasets": [dataset_name for dataset_name, _behavior in datasets],
                        "axis_slug": "quantity-sycophancy-unified-native-pc-sweep",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (pc_sweep_run_dir / "meta" / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "dataset_behaviors": {dataset_name: behavior_name for dataset_name, behavior_name in datasets}
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--pc-sweep-run-dir",
                    str(pc_sweep_run_dir),
                    "--run-id",
                    "clusters-run",
                    "--profile-clusters",
                    "2",
                    "--summary-clusters",
                    "2",
                ],
                cwd=str(repo_root),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("behavior-axis-pc-clusters", completed.stdout)

            cluster_roots = sorted((temp_repo / "artifacts" / "runs" / "behavior-axis-pc-clusters").rglob("clusters-run"))
            self.assertEqual(len(cluster_roots), 1)
            run_root = cluster_roots[0]

            self.assertTrue((run_root / "results" / "plots" / "pc_profile_clustered_heatmap__14.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pc_summary_clustered_heatmap__14.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pc_profile_cluster_scatter__14.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pc_summary_cluster_scatter__14.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pc_profile_embedding__14.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pc_summary_embedding__14.png").exists())
            self.assertTrue((run_root / "results" / "pc_clusters__14.csv").exists())


if __name__ == "__main__":
    unittest.main()
