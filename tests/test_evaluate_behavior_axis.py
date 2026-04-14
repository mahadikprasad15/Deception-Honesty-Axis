from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from deception_honesty_axis.common import write_json
from deception_honesty_axis.role_axis_transfer import build_role_axis_bundle, write_role_axis_bundle


class EvaluateBehaviorAxisScriptTest(unittest.TestCase):
    def test_evaluate_behavior_axis_supports_mixed_dataset_sources(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "evaluate_behavior_axis.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_repo = Path(tmpdir) / "temp_repo"
            (temp_repo / ".git").mkdir(parents=True)
            (temp_repo / "configs").mkdir(parents=True)
            (temp_repo / "data").mkdir(parents=True)
            shared_records_path = temp_repo / "data" / "shared_records.jsonl"
            split_eval_dir = temp_repo / "data" / "Deception-B-completion" / "test"
            split_eval_dir.mkdir(parents=True)
            axis_run_dir = temp_repo / "artifacts" / "runs" / "unified-role-axis-bundles" / "meta-llama-llama-3-2-3b-instruct" / "harmful-behavior" / "quantity-sycophancy-unified" / "native-questions-by-behavior" / "axis-run"
            (axis_run_dir / "meta").mkdir(parents=True)

            rows = [
                {
                    "ids": "a-pos-1",
                    "activation": [2.2, 1.8],
                    "label": 1,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "a-pos-2",
                    "activation": [1.8, 2.2],
                    "label": 1,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "a-neg-1",
                    "activation": [-2.2, -1.8],
                    "label": 0,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "a-neg-2",
                    "activation": [-1.8, -2.2],
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
                    "b-pos-1": torch.tensor([[[0.0, 0.0], [2.5, 2.2], [2.7, 2.4]]]),
                    "b-pos-2": torch.tensor([[[0.0, 0.0], [2.1, 2.3], [2.2, 2.5]]]),
                    "b-neg-1": torch.tensor([[[0.0, 0.0], [-2.5, -2.2], [-2.7, -2.4]]]),
                    "b-neg-2": torch.tensor([[[0.0, 0.0], [-2.1, -2.3], [-2.2, -2.5]]]),
                },
                str(split_eval_dir / "shard_0000.safetensors"),
            )

            bundle = build_role_axis_bundle(
                role_vectors={
                    "default": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
                    "honest_a": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
                    "honest_b": torch.tensor([[0.8, 1.2]], dtype=torch.float32),
                    "harm_a": torch.tensor([[-1.0, -1.0]], dtype=torch.float32),
                    "harm_b": torch.tensor([[-0.8, -1.2]], dtype=torch.float32),
                },
                honest_roles=["honest_a", "honest_b"],
                deceptive_roles=["harm_a", "harm_b"],
                anchor_role="default",
                layer_specs=["14"],
                layer_numbers=[14],
            )
            write_role_axis_bundle(axis_run_dir, bundle)
            write_json(
                axis_run_dir / "meta" / "run_manifest.json",
                {
                    "axis_name": "quantity-sycophancy-unified",
                    "variant_name": "native-questions-by-behavior",
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
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--config",
                    str(config_path),
                    "--axis-bundle-run-dir",
                    str(axis_run_dir),
                    "--run-id",
                    "axis-eval-run",
                ],
                cwd=str(repo_root),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("behavior-axis-eval", completed.stdout)

            run_roots = sorted((temp_repo / "artifacts" / "runs" / "behavior-axis-evaluation").rglob("axis-eval-run"))
            self.assertEqual(len(run_roots), 1)
            run_root = run_roots[0]

            with (run_root / "results" / "pairwise_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
                metric_rows = list(csv.DictReader(handle))
            self.assertEqual(len(metric_rows), 6)
            self.assertEqual({row["source_dataset"] for row in metric_rows}, {"__zero_shot__"})
            self.assertEqual({row["target_dataset"] for row in metric_rows}, {"Dataset A", "Deception-B"})
            self.assertTrue((run_root / "results" / "plots" / "zero_shot_grouped_bars__auroc__14.png").exists())


if __name__ == "__main__":
    unittest.main()
