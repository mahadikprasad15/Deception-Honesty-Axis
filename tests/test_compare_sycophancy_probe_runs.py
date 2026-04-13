from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from deception_honesty_axis.common import read_json


class CompareSycophancyProbeRunsTest(unittest.TestCase):
    def test_compare_sycophancy_probe_runs_writes_combined_plot(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "compare_sycophancy_probe_runs.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            artifact_root = tmp_root / "artifacts"

            run_specs = [
                (
                    "dataset-a",
                    {
                        "contrast_zero_shot": 0.61,
                        "pc1_zero_shot": 0.58,
                        "pc2_zero_shot": 0.55,
                        "pc3_zero_shot": 0.53,
                    },
                ),
                (
                    "dataset-b",
                    {
                        "contrast_zero_shot": 0.74,
                        "pc1_zero_shot": 0.71,
                        "pc2_zero_shot": 0.66,
                        "pc3_zero_shot": 0.63,
                    },
                ),
            ]
            run_dirs: list[Path] = []
            for dataset_name, method_values in run_specs:
                run_dir = tmp_root / dataset_name
                (run_dir / "results").mkdir(parents=True)
                (run_dir / "meta").mkdir(parents=True)
                with (run_dir / "meta" / "run_manifest.json").open("w", encoding="utf-8") as handle:
                    handle.write('{"target_name": "%s"}\n' % dataset_name)
                with (run_dir / "results" / "pairwise_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "method",
                            "layer_spec",
                            "layer_label",
                            "source_dataset",
                            "target_dataset",
                            "train_split",
                            "eval_split",
                            "auroc",
                            "auprc",
                            "balanced_accuracy",
                            "f1",
                            "accuracy",
                            "count",
                            "completed_at",
                        ],
                    )
                    writer.writeheader()
                    for method, auroc in method_values.items():
                        writer.writerow(
                            {
                                "method": method,
                                "layer_spec": "14",
                                "layer_label": "L14",
                                "source_dataset": "__zero_shot__",
                                "target_dataset": dataset_name,
                                "train_split": "",
                                "eval_split": "all",
                                "auroc": auroc,
                                "auprc": 0.5,
                                "balanced_accuracy": 0.5,
                                "f1": 0.5,
                                "accuracy": 0.5,
                                "count": 200,
                                "completed_at": "2026-04-13T00:00:00Z",
                            }
                        )
                run_dirs.append(run_dir)

            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--artifact-root",
                    str(artifact_root),
                    "--run-dirs",
                    *[str(path) for path in run_dirs],
                ],
                cwd=str(repo_root),
                env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("sycophancy-probe-compare", result.stdout)
            comparison_roots = sorted((artifact_root / "runs" / "sycophancy-activation-probe-comparison").iterdir())
            self.assertEqual(len(comparison_roots), 1)
            run_root = comparison_roots[0]
            summary = read_json(run_root / "results" / "summary.json")
            self.assertEqual(summary["record_count"], 8)
            self.assertEqual(summary["datasets"], ["dataset-a", "dataset-b"])
            self.assertEqual(len(summary["plots"]), 1)
            self.assertTrue((run_root / "results" / "combined_metrics.csv").exists())
            self.assertTrue((run_root / "results" / "plots" / "sycophancy_probe_comparison__auroc__14.png").exists())


if __name__ == "__main__":
    unittest.main()
