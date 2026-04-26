from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from deception_honesty_axis.common import read_json


class EvaluateActivationRowTransferSubspaceBaselinesScriptTest(unittest.TestCase):
    def test_random_orthonormal_subspace_is_orthonormal(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "evaluate_activation_row_transfer_subspace_baselines.py"
        spec = importlib.util.spec_from_file_location("subspace_baselines", script_path)
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["subspace_baselines"] = module
        spec.loader.exec_module(module)

        subspace = module.random_orthonormal_subspace(8, 3, seed=123)
        gram = subspace.components @ subspace.components.T
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-6)

    def test_subspace_baselines_run_and_resume(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "evaluate_activation_row_transfer_subspace_baselines.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_repo = Path(tmpdir) / "temp_repo"
            (temp_repo / ".git").mkdir(parents=True)
            (temp_repo / "configs").mkdir(parents=True)
            (temp_repo / "data").mkdir(parents=True)
            records_path = temp_repo / "data" / "records.jsonl"

            rows = [
                {
                    "ids": "a-pos-1",
                    "dataset_source": "a",
                    "activation": [2.0, 2.2, 0.4, -0.2],
                    "label": 1,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "a-pos-2",
                    "dataset_source": "a",
                    "activation": [1.8, 2.1, 0.5, -0.3],
                    "label": 1,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "a-neg-1",
                    "dataset_source": "a",
                    "activation": [-2.0, -2.2, -0.4, 0.2],
                    "label": 0,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "a-neg-2",
                    "dataset_source": "a",
                    "activation": [-1.8, -2.1, -0.5, 0.3],
                    "label": 0,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "b-pos-1",
                    "dataset_source": "b",
                    "activation": [2.3, 1.9, 0.7, -0.1],
                    "label": 1,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "b-pos-2",
                    "dataset_source": "b",
                    "activation": [2.1, 1.7, 0.6, -0.2],
                    "label": 1,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "b-neg-1",
                    "dataset_source": "b",
                    "activation": [-2.3, -1.9, -0.7, 0.1],
                    "label": 0,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "b-neg-2",
                    "dataset_source": "b",
                    "activation": [-2.1, -1.7, -0.6, 0.2],
                    "label": 0,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
            ]
            with records_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            config_path = temp_repo / "configs" / "datasets.json"
            config_path.write_text(
                json.dumps(
                    {
                        "artifacts": {"root": "artifacts"},
                        "experiment": {
                            "name": "activation-row-transfer",
                            "behavior": "deception",
                            "model_name": "meta-llama/Llama-3.2-3B-Instruct",
                            "dataset_set": "toy",
                        },
                        "activation_rows": {
                            "expected_pooling": "mean_response",
                            "expected_layer_number": 14,
                        },
                        "datasets": [
                            {
                                "name": "Dataset A",
                                "activation_jsonl": "data/records.jsonl",
                                "group_field": "dataset_source",
                                "group_value": "a",
                            },
                            {
                                "name": "Dataset B",
                                "activation_jsonl": "data/records.jsonl",
                                "group_field": "dataset_source",
                                "group_value": "b",
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
            command = [
                sys.executable,
                str(script_path),
                "--config",
                str(config_path),
                "--run-id",
                "subspace-run",
                "--k-values",
                "1",
                "2",
                "--random-seeds",
                "101",
                "103",
            ]
            first = subprocess.run(
                command,
                cwd=str(repo_root),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("activation-row-transfer-subspace-baselines", first.stdout)

            second = subprocess.run(
                command,
                cwd=str(repo_root),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("resuming", second.stdout)

            run_roots = sorted(
                (temp_repo / "artifacts" / "runs" / "activation-row-transfer-subspace-baselines").rglob(
                    "subspace-run"
                )
            )
            self.assertEqual(len(run_roots), 1)
            run_root = run_roots[0]
            with (run_root / "results" / "pairwise_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
                metric_rows = list(csv.DictReader(handle))
            self.assertEqual(len(metric_rows), 16)
            self.assertEqual({row["method"] for row in metric_rows}, {"dataset_pca_logistic", "random_orthonormal_logistic"})
            self.assertEqual({int(row["selected_pc_count"]) for row in metric_rows}, {1, 2})

            with (run_root / "results" / "per_seed_pairwise_metrics.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                seed_rows = list(csv.DictReader(handle))
            self.assertEqual(len(seed_rows), 16)
            self.assertTrue((run_root / "results" / "summary_by_method_k.csv").exists())
            self.assertTrue((run_root / "results" / "fit_summary.csv").exists())
            self.assertTrue(
                (run_root / "results" / "plots" / "dataset_pca_logistic__k-001__auroc_heatmap.png").exists()
            )
            self.assertTrue(
                (run_root / "results" / "plots" / "random_orthonormal_logistic__k-002__auroc_heatmap.png").exists()
            )

            results = read_json(run_root / "results" / "results.json")
            self.assertEqual(results["k_values"], [1, 2])
            self.assertEqual(results["random_seeds"], [101, 103])


if __name__ == "__main__":
    unittest.main()
