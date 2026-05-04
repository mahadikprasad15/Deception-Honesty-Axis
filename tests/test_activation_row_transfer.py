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

from deception_honesty_axis.activation_row_transfer import load_activation_row_dataset
from deception_honesty_axis.activation_row_transfer_config import load_activation_row_transfer_config
from deception_honesty_axis.common import read_json


class ActivationRowTransferConfigTest(unittest.TestCase):
    def test_load_activation_row_transfer_config_supports_nested_train_eval_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            (repo_root / ".git").mkdir(parents=True)
            (repo_root / "configs").mkdir(parents=True)
            config_path = repo_root / "configs" / "transfer.json"
            config_path.write_text(
                json.dumps(
                    {
                        "artifacts": {"root": "artifacts"},
                        "experiment": {
                            "name": "activation-row-transfer",
                            "behavior": "deception",
                            "model_name": "meta-llama/Llama-3.2-3B-Instruct",
                            "dataset_set": "toy-datasets",
                        },
                        "activation_rows": {
                            "expected_pooling": "mean_response",
                            "expected_layer_number": 14,
                        },
                        "datasets": [
                            {
                                "name": "Dataset A",
                                "train": {"activation_jsonl": "data/a_train.jsonl"},
                                "eval": {"activation_jsonl": "data/a_eval.jsonl"},
                            }
                        ],
                        "evaluation": {"methods": ["activation_logistic"]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            config = load_activation_row_transfer_config(config_path)

            self.assertEqual(config.behavior_name, "deception")
            self.assertEqual(config.expected_layer_number, 14)
            self.assertEqual(len(config.datasets), 1)
            dataset = config.datasets[0]
            self.assertFalse(dataset.uses_shared_source)
            self.assertEqual(dataset.train_source.activation_jsonl.resolve(), (repo_root / "data" / "a_train.jsonl").resolve())
            self.assertEqual(dataset.eval_source.activation_jsonl.resolve(), (repo_root / "data" / "a_eval.jsonl").resolve())

    def test_load_activation_row_transfer_config_supports_completion_split_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            (repo_root / ".git").mkdir(parents=True)
            (repo_root / "configs").mkdir(parents=True)
            config_path = repo_root / "configs" / "transfer.json"
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
                                "name": "Dataset B",
                                "behavior": "deception",
                                "train": {
                                    "source_kind": "completion_split",
                                    "split_dir": "data/Deception-B-completion/train",
                                },
                                "eval": {
                                    "source_kind": "completion_split",
                                    "split_dir": "data/Deception-B-completion/test",
                                },
                            }
                        ],
                        "evaluation": {"methods": ["activation_logistic"]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            config = load_activation_row_transfer_config(config_path)

            dataset = config.datasets[0]
            self.assertEqual(dataset.behavior, "deception")
            self.assertEqual(dataset.train_source.source_kind, "completion_split")
            self.assertEqual(
                dataset.train_source.split_dir.resolve(),
                (repo_root / "data" / "Deception-B-completion" / "train").resolve(),
            )
            self.assertEqual(dataset.eval_source.split, "test")

    def test_activation_row_sources_support_deterministic_stratified_row_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            (repo_root / ".git").mkdir(parents=True)
            (repo_root / "configs").mkdir(parents=True)
            (repo_root / "data").mkdir(parents=True)
            records_path = repo_root / "data" / "records.jsonl"
            config_path = repo_root / "configs" / "transfer.json"

            rows = []
            for label, center in ((0, -1.0), (1, 1.0)):
                for index in range(10):
                    rows.append(
                        {
                            "ids": f"{label}-{index}",
                            "dataset_source": "toy",
                            "activation": [center, float(index)],
                            "label": label,
                            "activation_pooling": "mean_response",
                            "activation_layer_number": 14,
                        }
                    )
            with records_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            split_payload = {
                "source_kind": "activation_rows",
                "activation_jsonl": "data/records.jsonl",
                "row_split_strategy": "stratified",
                "row_split_train_fraction": 0.8,
                "row_split_seed": 7,
            }
            config_path.write_text(
                json.dumps(
                    {
                        "artifacts": {"root": "artifacts"},
                        "experiment": {
                            "name": "activation-row-transfer",
                            "behavior": "sycophancy",
                            "model_name": "meta-llama/Llama-3.2-3B-Instruct",
                            "dataset_set": "toy",
                        },
                        "activation_rows": {
                            "expected_pooling": "mean_response",
                            "expected_layer_number": 14,
                        },
                        "datasets": [
                            {
                                "name": "Toy",
                                "train": {**split_payload, "row_split": "train"},
                                "eval": {**split_payload, "row_split": "eval"},
                            }
                        ],
                        "evaluation": {"methods": ["activation_logistic"]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            config = load_activation_row_transfer_config(config_path)
            dataset = load_activation_row_dataset(
                config.datasets[0],
                expected_pooling=config.expected_pooling,
                expected_layer_number=config.expected_layer_number,
            )

            self.assertEqual(dataset.train.summary()["label_counts"], {0: 8, 1: 8})
            self.assertEqual(dataset.eval.summary()["label_counts"], {0: 2, 1: 2})
            self.assertEqual(set(dataset.train.sample_ids).isdisjoint(dataset.eval.sample_ids), True)
            self.assertEqual(dataset.train.source_manifest["row_split_seed"], 7)
            self.assertEqual(dataset.eval.source_manifest["row_split_selected_count"], 4)


class EvaluateActivationRowTransferScriptTest(unittest.TestCase):
    def test_evaluate_activation_row_transfer_runs_and_resumes(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "evaluate_activation_row_transfer.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_repo = Path(tmpdir) / "temp_repo"
            (temp_repo / ".git").mkdir(parents=True)
            (temp_repo / "configs").mkdir(parents=True)
            (temp_repo / "data").mkdir(parents=True)
            shared_records_path = temp_repo / "data" / "shared_records.jsonl"
            config_path = temp_repo / "configs" / "transfer.json"

            rows = [
                {
                    "ids": "a-pos-1",
                    "dataset_source": "group-a",
                    "activation": [2.0, 1.8],
                    "label": 1,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "a-pos-2",
                    "dataset_source": "group-a",
                    "activation": [1.7, 2.2],
                    "label": 1,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "a-neg-1",
                    "dataset_source": "group-a",
                    "activation": [-2.1, -1.9],
                    "label": 0,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "a-neg-2",
                    "dataset_source": "group-a",
                    "activation": [-1.8, -2.2],
                    "label": 0,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "b-pos-1",
                    "dataset_source": "group-b",
                    "activation": [3.0, 2.8],
                    "label": 1,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "b-pos-2",
                    "dataset_source": "group-b",
                    "activation": [2.7, 3.1],
                    "label": 1,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "b-neg-1",
                    "dataset_source": "group-b",
                    "activation": [-2.9, -3.0],
                    "label": 0,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "b-neg-2",
                    "dataset_source": "group-b",
                    "activation": [-3.1, -2.7],
                    "label": 0,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
            ]
            with shared_records_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            config_path.write_text(
                json.dumps(
                    {
                        "artifacts": {"root": "artifacts"},
                        "experiment": {
                            "name": "activation-row-transfer",
                            "behavior": "sycophancy",
                            "model_name": "meta-llama/Llama-3.2-3B-Instruct",
                            "dataset_set": "toy-shared",
                        },
                        "activation_rows": {
                            "expected_pooling": "mean_response",
                            "expected_layer_number": 14,
                        },
                        "datasets": [
                            {
                                "name": "Dataset A",
                                "activation_jsonl": "data/shared_records.jsonl",
                                "group_field": "dataset_source",
                                "group_value": "group-a",
                            },
                            {
                                "name": "Dataset B",
                                "activation_jsonl": "data/shared_records.jsonl",
                                "group_field": "dataset_source",
                                "group_value": "group-b",
                            },
                        ],
                        "evaluation": {
                            "methods": ["activation_logistic"],
                            "random_seed": 42,
                            "logistic_max_iter": 200,
                        },
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
                    "--run-id",
                    "fixed-run",
                ],
                cwd=str(repo_root),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("activation-row-transfer", first.stdout)

            run_roots = sorted((temp_repo / "artifacts" / "runs" / "activation-row-transfer").rglob("fixed-run"))
            self.assertEqual(len(run_roots), 1)
            run_root = run_roots[0]

            with (run_root / "results" / "pairwise_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
                metric_rows = list(csv.DictReader(handle))
            self.assertEqual(len(metric_rows), 4)
            self.assertEqual({row["source_dataset"] for row in metric_rows}, {"Dataset A", "Dataset B"})
            self.assertEqual({row["target_dataset"] for row in metric_rows}, {"Dataset A", "Dataset B"})
            self.assertTrue(all(row["method"] == "activation_logistic" for row in metric_rows))

            results_summary = read_json(run_root / "results" / "results.json")
            self.assertEqual(results_summary["record_count"], 4)
            self.assertEqual(results_summary["datasets"], ["Dataset A", "Dataset B"])
            self.assertEqual(len(results_summary["heatmaps"]), 1)
            self.assertTrue((run_root / "results" / "summary_by_method.csv").exists())
            self.assertTrue((run_root / "results" / "fit_summary.csv").exists())
            self.assertTrue((run_root / "results" / "per_example_scores.jsonl").exists())
            self.assertTrue((run_root / "results" / "plots" / "activation_logistic__14__auroc_heatmap.png").exists())

            second = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--config",
                    str(config_path),
                    "--run-id",
                    "fixed-run",
                ],
                cwd=str(repo_root),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("resuming with 4/4", second.stdout)
            with (run_root / "results" / "pairwise_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
                metric_rows_after_resume = list(csv.DictReader(handle))
            self.assertEqual(len(metric_rows_after_resume), 4)

    def test_evaluate_activation_row_transfer_supports_mixed_sources(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "evaluate_activation_row_transfer.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_repo = Path(tmpdir) / "temp_repo"
            (temp_repo / ".git").mkdir(parents=True)
            (temp_repo / "configs").mkdir(parents=True)
            (temp_repo / "data").mkdir(parents=True)
            shared_records_path = temp_repo / "data" / "shared_records.jsonl"
            split_train_dir = temp_repo / "data" / "Deception-B-completion" / "train"
            split_eval_dir = temp_repo / "data" / "Deception-B-completion" / "test"
            split_train_dir.mkdir(parents=True)
            split_eval_dir.mkdir(parents=True)
            config_path = temp_repo / "configs" / "transfer.json"

            rows = [
                {
                    "ids": "a-pos-1",
                    "activation": [2.0, 1.8],
                    "label": 1,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "a-pos-2",
                    "activation": [1.7, 2.2],
                    "label": 1,
                    "activation_pooling": "mean_response",
                    "activation_layer_number": 14,
                },
                {
                    "ids": "a-neg-1",
                    "activation": [-2.1, -1.9],
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

            def write_completion_split(split_dir: Path, prefix: str, positive_shift: float) -> None:
                manifest_rows = [
                    {
                        "id": f"{prefix}-pos-1",
                        "label": 1,
                        "user_end_idx": 1,
                        "completion_end_idx": 3,
                    },
                    {
                        "id": f"{prefix}-pos-2",
                        "label": 1,
                        "user_end_idx": 1,
                        "completion_end_idx": 3,
                    },
                    {
                        "id": f"{prefix}-neg-1",
                        "label": 0,
                        "user_end_idx": 1,
                        "completion_end_idx": 3,
                    },
                    {
                        "id": f"{prefix}-neg-2",
                        "label": 0,
                        "user_end_idx": 1,
                        "completion_end_idx": 3,
                    },
                ]
                (split_dir / "manifest.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in manifest_rows),
                    encoding="utf-8",
                )
                shard_payload = {
                    f"{prefix}-pos-1": torch.tensor([[[0.0, 0.0], [2.0 + positive_shift, 2.1], [2.2 + positive_shift, 2.3]]]),
                    f"{prefix}-pos-2": torch.tensor([[[0.0, 0.0], [1.7 + positive_shift, 1.8], [1.9 + positive_shift, 2.0]]]),
                    f"{prefix}-neg-1": torch.tensor([[[0.0, 0.0], [-2.0 - positive_shift, -1.9], [-2.2 - positive_shift, -2.1]]]),
                    f"{prefix}-neg-2": torch.tensor([[[0.0, 0.0], [-1.7 - positive_shift, -1.8], [-1.9 - positive_shift, -2.0]]]),
                }
                save_file(shard_payload, str(split_dir / "shard_0000.safetensors"))

            write_completion_split(split_train_dir, "train", 0.0)
            write_completion_split(split_eval_dir, "test", 0.2)

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
                                "train": {
                                    "source_kind": "completion_split",
                                    "split_dir": "data/Deception-B-completion/train",
                                },
                                "eval": {
                                    "source_kind": "completion_split",
                                    "split_dir": "data/Deception-B-completion/test",
                                },
                            },
                        ],
                        "evaluation": {
                            "methods": ["activation_logistic"],
                            "random_seed": 42,
                            "logistic_max_iter": 200,
                        },
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
                    "--run-id",
                    "mixed-run",
                ],
                cwd=str(repo_root),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("activation-row-transfer", completed.stdout)

            run_roots = sorted((temp_repo / "artifacts" / "runs" / "activation-row-transfer").rglob("mixed-run"))
            self.assertEqual(len(run_roots), 1)
            run_root = run_roots[0]

            with (run_root / "results" / "pairwise_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
                metric_rows = list(csv.DictReader(handle))
            self.assertEqual(len(metric_rows), 4)
            self.assertEqual({row["source_dataset"] for row in metric_rows}, {"Dataset A", "Deception-B"})
            self.assertEqual({row["target_dataset"] for row in metric_rows}, {"Dataset A", "Deception-B"})
            self.assertTrue((run_root / "results" / "plots" / "activation_logistic__14__auroc_heatmap.png").exists())


if __name__ == "__main__":
    unittest.main()
