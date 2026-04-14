from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from deception_honesty_axis.activation_row_targets import (
    ensure_pooling_compatibility,
    ensure_single_layer_compatibility,
    grouped_indices,
    load_activation_rows,
    validate_and_stack_activation_rows,
)


class ActivationRowTargetsTest(unittest.TestCase):
    def test_validate_and_stack_activation_rows_reads_row_metadata(self) -> None:
        rows = [
            {
                "ids": "sample-1",
                "dataset_source": "factual",
                "activation": [1.0, 2.0],
                "label": 1,
                "activation_pooling": "mean_response",
                "activation_layer_number": 14,
            },
            {
                "ids": "sample-2",
                "dataset_source": "evaluative",
                "activation": [3.0, 4.0],
                "label": 0,
                "activation_pooling": "mean_response",
                "activation_layer_number": 14,
            },
        ]

        stacked = validate_and_stack_activation_rows(rows, group_field="dataset_source")

        self.assertEqual(stacked.features.shape, (2, 2))
        self.assertEqual(stacked.activation_pooling, "mean_response")
        self.assertEqual(stacked.activation_layer_number, 14)
        self.assertEqual(stacked.group_values, ["factual", "evaluative"])

    def test_validate_and_stack_activation_rows_rejects_manifest_conflict(self) -> None:
        rows = [
            {
                "ids": "sample-1",
                "dataset_source": "factual",
                "activation": [1.0, 2.0],
                "label": 1,
                "activation_pooling": "last_token",
                "activation_layer_number": 14,
            }
        ]

        with self.assertRaises(ValueError):
            validate_and_stack_activation_rows(
                rows,
                group_field="dataset_source",
                source_manifest={"activation_pooling": "mean_response", "activation_layer_number": 14},
            )

    def test_grouped_indices_adds_all_bucket(self) -> None:
        groups = grouped_indices(["factual", "evaluative", "factual"])

        self.assertEqual(groups["all"].tolist(), [0, 1, 2])
        self.assertEqual(groups["factual"].tolist(), [0, 2])

    def test_load_activation_rows_jsonl_reads_adjacent_run_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = Path(tmp_dir) / "run"
            (run_root / "meta").mkdir(parents=True)
            (run_root / "results").mkdir(parents=True)
            (run_root / "meta" / "run_manifest.json").write_text(
                json.dumps({"activation_pooling": "mean_response", "layer": 14}) + "\n",
                encoding="utf-8",
            )
            records_path = run_root / "results" / "records.jsonl"
            records_path.write_text(
                json.dumps(
                    {
                        "ids": "sample-1",
                        "dataset_source": "factual",
                        "activation": [1.0, 2.0],
                        "label": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows, manifest = load_activation_rows(activation_jsonl=records_path)

            self.assertEqual(len(rows), 1)
            self.assertEqual(manifest["activation_pooling"], "mean_response")
            self.assertEqual(manifest["activation_layer_number"], 14)

    def test_compatibility_checks_raise_on_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            ensure_pooling_compatibility(
                activation_pooling="last_token",
                expected_pooling="mean_response",
                context="test",
            )
        with self.assertRaises(ValueError):
            ensure_single_layer_compatibility(
                activation_layer_number=14,
                expected_layer_numbers=[13, 14],
                context="test",
            )


if __name__ == "__main__":
    unittest.main()
