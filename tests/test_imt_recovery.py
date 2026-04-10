from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from deception_honesty_axis.imt_config import imt_run_root, load_imt_config
from deception_honesty_axis.imt_plotting import save_axis_variant_heatmap, save_four_axis_grouped_bars
from deception_honesty_axis.imt_recovery import (
    IMT_AXIS_ORDER,
    aggregate_template_scores,
    fit_imt_axes,
    parse_imt_score_output,
)


class IMTRecoveryTest(unittest.TestCase):
    def test_parse_imt_score_output_reads_json_and_regex(self) -> None:
        parsed = parse_imt_score_output(
            json.dumps(
                {
                    "q1": 2,
                    "q2": 1.5,
                    "q3": 0,
                    "q4": 0.5,
                    "primary_dimension": "Q1",
                    "confidence": 0.8,
                    "short_rationale": "clear falsification",
                }
            )
        )
        self.assertEqual(parsed["parse_mode"], "json")
        self.assertAlmostEqual(parsed["q2"], 1.5)

        regex_parsed = parse_imt_score_output(
            'q1: 1, q2: 2, q3: 0, q4: 3, primary_dimension: "Q4", confidence: 0.75'
        )
        self.assertEqual(regex_parsed["parse_mode"], "regex")
        self.assertEqual(regex_parsed["q4"], 3.0)
        self.assertEqual(regex_parsed["primary_dimension"], "Q4")

    def test_aggregate_template_scores_requires_complete_template_set(self) -> None:
        rows = [
            {"item_id": "a", "template_name": "rubric_first", "parsed_ok": True, "q1": 1, "q2": 0, "q3": 0, "q4": 0, "confidence": 0.9, "primary_dimension": "Q1"},
            {"item_id": "a", "template_name": "responsiveness_emphasis", "parsed_ok": True, "q1": 2, "q2": 0, "q3": 0, "q4": 0, "confidence": 0.8, "primary_dimension": "Q1"},
            {"item_id": "a", "template_name": "listener_misled_mechanism", "parsed_ok": True, "q1": 1, "q2": 1, "q3": 0, "q4": 0, "confidence": 0.7, "primary_dimension": "mixed"},
            {"item_id": "b", "template_name": "rubric_first", "parsed_ok": True, "q1": 0, "q2": 0, "q3": 1, "q4": 0, "confidence": 0.5, "primary_dimension": "Q3"},
        ]
        aggregated = aggregate_template_scores(
            rows,
            template_names=["rubric_first", "responsiveness_emphasis", "listener_misled_mechanism"],
        )
        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0]["item_id"], "a")
        self.assertAlmostEqual(aggregated[0]["q1"], 4 / 3)
        self.assertEqual(aggregated[0]["primary_dimension"], "Q1")

    def test_fit_imt_axes_returns_four_axes_and_orthogonal_copy(self) -> None:
        rng = np.random.default_rng(0)
        features = rng.normal(size=(24, 16)).astype(np.float32)
        targets = np.stack(
            [
                features[:, 0] - features[:, 1],
                features[:, 2] + 0.25 * features[:, 3],
                features[:, 4] - 0.5 * features[:, 5],
                features[:, 6] + features[:, 7],
            ],
            axis=1,
        ).astype(np.float32)
        payload = fit_imt_axes(features, targets, pca_components=8, ridge_alpha=1.0)
        self.assertEqual(payload["raw_axes"].shape, (16, 4))
        self.assertEqual(payload["orth_axes"].shape, (16, 4))
        orth = np.asarray(payload["orth_axis_cosines"], dtype=np.float32)
        np.testing.assert_allclose(np.diag(orth), np.ones(4), atol=1e-4)

    def test_load_imt_config_and_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            (repo_root / ".git").mkdir()
            (repo_root / "configs" / "experiments").mkdir(parents=True)
            experiment_payload = {
                "experiment_name": "quality-axis-v1",
                "model": {"id": "meta-llama/Llama-3.2-3B-Instruct", "generation": {}, "dtype": "float16"},
                "data": {"question_file": "data/questions/q.jsonl", "role_manifest_file": "data/manifests/r.json"},
                "subset": {"instruction_selection": "first_n", "instruction_count": 1, "question_selection": "first_n", "question_count": 1},
                "artifacts": {"root": "artifacts", "dataset_name": "assistant-axis", "corpus_name": "quality-axis-v1"},
                "saving": {"save_every": 1},
                "analysis": {"filter_name": "all-responses", "pooling": "mean_response", "anchor_role": "default"},
                "hf": {"dataset_repo_id": "owner/repo"},
            }
            for name in ("quality", "quantity", "relation", "manner"):
                (repo_root / "configs" / "experiments" / f"{name}.json").write_text(json.dumps(experiment_payload), encoding="utf-8")

            (repo_root / "configs" / "imt").mkdir(parents=True)
            config_path = repo_root / "configs" / "imt" / "imt.json"
            config_path.write_text(
                json.dumps(
                    {
                        "artifacts": {"root": "artifacts"},
                        "source_bank": {
                            "name": "four-axis-v1",
                            "sources": [
                                {"axis_name": "quality", "experiment_config": "configs/experiments/quality.json", "analysis_run_id": "r1"},
                                {"axis_name": "quantity", "experiment_config": "configs/experiments/quantity.json", "analysis_run_id": "r1"},
                                {"axis_name": "relation", "experiment_config": "configs/experiments/relation.json", "analysis_run_id": "r1"},
                                {"axis_name": "manner", "experiment_config": "configs/experiments/manner.json", "analysis_run_id": "r1"}
                            ]
                        },
                        "scoring": {
                            "model": {"id": "meta-llama/Llama-3.2-3B-Instruct"},
                            "templates": ["rubric_first", "responsiveness_emphasis", "listener_misled_mechanism"]
                        },
                        "fit": {"granularity": "instance", "layer": 14, "pca_components": 8, "ridge_alpha": 1.0},
                        "target": {
                            "activations_root": "/tmp/activations",
                            "datasets": ["Deception-AILiar-completion"],
                            "eval_split": "test",
                            "pooling": "completion_mean"
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = load_imt_config(config_path)
            run_root = imt_run_root(config, "run-1")
            self.assertEqual(config.bank_slug, "four-axis-v1")
            self.assertTrue(run_root.exists())

    def test_plot_helpers_write_expected_files(self) -> None:
        rows = []
        for axis_name in IMT_AXIS_ORDER:
            for variant in ("Raw", "Orth"):
                for dataset in (
                    "Deception-AILiar-completion",
                    "Deception-ConvincingGame-completion",
                    "Deception-HarmPressureChoice-completion",
                    "Deception-InstructedDeception-completion",
                    "Deception-Mask-completion",
                    "Deception-Roleplaying-completion",
                ):
                    rows.append(
                        {
                            "axis_name": axis_name,
                            "variant": variant,
                            "target_dataset": dataset,
                            "value": 0.6,
                        }
                    )
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            grouped = save_four_axis_grouped_bars(
                root / "grouped.png",
                rows,
                variant_order=["Raw", "Orth"],
                dataset_order=[row["target_dataset"] for row in rows[:6]],
                title="Recovered IMT Zero-Shot AUROC by Dataset",
            )
            heatmap = save_axis_variant_heatmap(
                root / "heatmap.png",
                rows,
                variant_order=["Raw", "Orth"],
                dataset_order=[row["target_dataset"] for row in rows[:6]],
                title="Recovered IMT Zero-Shot AUROC Heatmap",
            )
            self.assertTrue(Path(grouped).exists())
            self.assertTrue(Path(heatmap).exists())


if __name__ == "__main__":
    unittest.main()
