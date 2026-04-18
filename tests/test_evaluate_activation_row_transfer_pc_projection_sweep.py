from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from deception_honesty_axis.common import read_json, write_json
from deception_honesty_axis.role_axis_transfer import build_role_axis_bundle, write_role_axis_bundle


class EvaluateActivationRowTransferPCProjectionSweepScriptTest(unittest.TestCase):
    def test_projected_transfer_sweep_runs_and_resumes(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "evaluate_activation_row_transfer_pc_projection_sweep.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_repo = Path(tmpdir) / "temp_repo"
            (temp_repo / ".git").mkdir(parents=True)
            (temp_repo / "configs").mkdir(parents=True)
            (temp_repo / "data").mkdir(parents=True)
            shared_records_path = temp_repo / "data" / "shared_records.jsonl"
            split_eval_dir = temp_repo / "data" / "Deception-B-completion" / "test"
            split_eval_dir.mkdir(parents=True)
            axis_run_dir = (
                temp_repo
                / "artifacts"
                / "runs"
                / "role-axis-bundles"
                / "meta-llama-llama-3-2-3b-instruct"
                / "assistant-axis"
                / "quantity-axis-v2-cumulative-pc-sweep"
                / "mean_response"
                / "axis-run"
            )
            (axis_run_dir / "meta").mkdir(parents=True)

            rows = [
                {
                    "ids": "a-pos-1",
                    "dataset_source": "group-a",
                    "activation": [2.2, 1.8, 0.4, -0.3],
                    "label": 1,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "a-pos-2",
                    "dataset_source": "group-a",
                    "activation": [1.8, 2.2, 0.3, -0.4],
                    "label": 1,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "a-neg-1",
                    "dataset_source": "group-a",
                    "activation": [-2.2, -1.8, -0.4, 0.3],
                    "label": 0,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "a-neg-2",
                    "dataset_source": "group-a",
                    "activation": [-1.8, -2.2, -0.3, 0.4],
                    "label": 0,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
            ]
            with shared_records_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            manifest_rows = [
                {"id": "b-pos-1", "label": 1, "user_end_idx": 1, "completion_end_idx": 3},
                {"id": "b-pos-2", "label": 1, "user_end_idx": 1, "completion_end_idx": 3},
                {"id": "b-neg-1", "label": 0, "user_end_idx": 1, "completion_end_idx": 3},
                {"id": "b-neg-2", "label": 0, "user_end_idx": 1, "completion_end_idx": 3},
            ]
            (split_eval_dir / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in manifest_rows),
                encoding="utf-8",
            )
            save_file(
                {
                    "b-pos-1": torch.tensor([[[0.0, 0.0, 0.0, 0.0], [2.5, 2.2, 0.5, -0.5], [2.7, 2.4, 0.4, -0.4]]]),
                    "b-pos-2": torch.tensor([[[0.0, 0.0, 0.0, 0.0], [2.1, 2.3, 0.6, -0.6], [2.2, 2.5, 0.5, -0.5]]]),
                    "b-neg-1": torch.tensor([[[0.0, 0.0, 0.0, 0.0], [-2.5, -2.2, -0.5, 0.5], [-2.7, -2.4, -0.4, 0.4]]]),
                    "b-neg-2": torch.tensor([[[0.0, 0.0, 0.0, 0.0], [-2.1, -2.3, -0.6, 0.6], [-2.2, -2.5, -0.5, 0.5]]]),
                },
                str(split_eval_dir / "shard_0000.safetensors"),
            )

            bundle = build_role_axis_bundle(
                role_vectors={
                    "default": torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
                    "honest_a": torch.tensor([[1.0, 1.0, 0.2, 0.1]], dtype=torch.float32),
                    "honest_b": torch.tensor([[0.8, 1.2, 0.1, 0.2]], dtype=torch.float32),
                    "honest_c": torch.tensor([[1.1, 0.9, 0.3, 0.0]], dtype=torch.float32),
                    "harm_a": torch.tensor([[-1.0, -1.0, -0.2, -0.1]], dtype=torch.float32),
                    "harm_b": torch.tensor([[-0.8, -1.2, -0.1, -0.2]], dtype=torch.float32),
                    "harm_c": torch.tensor([[-1.1, -0.9, -0.3, 0.0]], dtype=torch.float32),
                },
                honest_roles=["honest_a", "honest_b", "honest_c"],
                deceptive_roles=["harm_a", "harm_b", "harm_c"],
                anchor_role="default",
                layer_specs=["14"],
                layer_numbers=[14],
                pc_count=3,
            )
            write_role_axis_bundle(axis_run_dir, bundle)
            write_json(
                axis_run_dir / "meta" / "run_manifest.json",
                {
                    "bundle_name": "quantity-axis-v2-cumulative-pc-sweep",
                    "expected_pooling": "mean_response",
                },
            )

            config_path = temp_repo / "configs" / "datasets.json"
            config_path.write_text(
                json.dumps(
                    {
                        "artifacts": {"root": "artifacts"},
                        "experiment": {
                            "name": "activation-row-transfer",
                            "behavior": "harmful-behavior",
                            "model_name": "meta-llama/Llama-3.2-3B-Instruct",
                            "dataset_set": "toy-mixed",
                        },
                        "activation_rows": {
                            "expected_pooling": "mean_response",
                            "expected_layer_number": 14,
                        },
                        "datasets": [
                            {
                                "name": "Dataset A",
                                "behavior": "sycophancy",
                                "activation_jsonl": "data/shared_records.jsonl",
                                "group_field": "dataset_source",
                                "group_value": "group-a",
                            },
                            {
                                "name": "Deception-B",
                                "behavior": "deception",
                                "source_kind": "completion_split",
                                "split_dir": "data/Deception-B-completion/test",
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
            first = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--config",
                    str(config_path),
                    "--axis-bundle-run-dir",
                    str(axis_run_dir),
                    "--run-id",
                    "projection-sweep-run",
                    "--max-pcs",
                    "2",
                ],
                cwd=str(repo_root),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("activation-row-transfer-pc-projection-sweep", first.stdout)

            second = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--config",
                    str(config_path),
                    "--axis-bundle-run-dir",
                    str(axis_run_dir),
                    "--run-id",
                    "projection-sweep-run",
                    "--max-pcs",
                    "2",
                ],
                cwd=str(repo_root),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("resuming", second.stdout)

            run_roots = sorted(
                (temp_repo / "artifacts" / "runs" / "activation-row-transfer-pc-projection-sweep").rglob(
                    "projection-sweep-run"
                )
            )
            self.assertEqual(len(run_roots), 1)
            run_root = run_roots[0]
            self.assertIn("quantity-axis-v2-cumulative-pc-sweep", str(run_root))

            with (run_root / "results" / "pairwise_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
                metric_rows = list(csv.DictReader(handle))
            self.assertEqual(len(metric_rows), 8)
            self.assertEqual({int(row["selected_pc_count"]) for row in metric_rows}, {1, 2})

            self.assertTrue((run_root / "results" / "full_projected_splits" / "dataset-a" / "train.npz").exists())
            self.assertTrue((run_root / "results" / "full_projected_splits" / "deception-b" / "eval.npz").exists())
            self.assertTrue((run_root / "results" / "summary_by_k.csv").exists())
            self.assertTrue((run_root / "results" / "fit_summary.csv").exists())
            self.assertTrue((run_root / "results" / "plots" / "k-001__auroc_heatmap.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "k-002__auroc_heatmap.png").exists())

            results = read_json(run_root / "results" / "results.json")
            self.assertEqual(results["k_values"], [1, 2])
            self.assertEqual(results["max_selected_pc_count"], 2)
            self.assertEqual(results["projected_feature_dim"], 2)

            projected_payload = np.load(run_root / "results" / "full_projected_splits" / "dataset-a" / "train.npz")
            self.assertEqual(projected_payload["features"].shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
