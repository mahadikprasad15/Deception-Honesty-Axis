from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from deception_honesty_axis.role_axis_transfer import (
    ZERO_SHOT_SOURCE,
    build_role_axis_bundle,
    evaluate_zero_shot,
    load_completion_mean_split,
    resolve_layer_specs,
    save_transfer_lineplots,
    scores_for_layer,
    write_role_axis_bundle,
)


class RoleAxisTransferTest(unittest.TestCase):
    def test_resolve_layer_specs_supports_numeric_and_last4_mean(self) -> None:
        specs = resolve_layer_specs(28, ["7", "14", "21", "28", "last4_mean"])
        self.assertEqual([spec.key for spec in specs], ["7", "14", "21", "28", "last4_mean"])
        self.assertEqual(specs[0].layer_indices, (6,))
        self.assertEqual(specs[0].layer_numbers, (7,))
        self.assertEqual(specs[3].layer_indices, (27,))
        self.assertEqual(specs[3].layer_numbers, (28,))
        self.assertEqual(specs[4].layer_indices, (24, 25, 26, 27))
        self.assertEqual(specs[4].layer_numbers, (25, 26, 27, 28))

    def test_resolve_layer_specs_preserves_original_layer_number_for_selected_layers(self) -> None:
        specs = resolve_layer_specs(1, ["14"], [14])
        self.assertEqual(specs[0].key, "14")
        self.assertEqual(specs[0].label, "L14")
        self.assertEqual(specs[0].layer_indices, (0,))
        self.assertEqual(specs[0].layer_numbers, (14,))
        self.assertEqual(specs[0].indices_for_tensor(1), (0,))
        self.assertEqual(specs[0].indices_for_tensor(28), (13,))

    def test_build_role_axis_bundle_aligns_pc1_to_honest_direction(self) -> None:
        role_vectors = {
            "default": torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            "judge": torch.tensor([[1.0, 0.0], [1.5, 0.0], [2.0, 0.0], [2.5, 0.0]]),
            "validator": torch.tensor([[0.9, 0.0], [1.4, 0.0], [1.9, 0.0], [2.4, 0.0]]),
            "spy": torch.tensor([[-1.0, 0.0], [-1.5, 0.0], [-2.0, 0.0], [-2.5, 0.0]]),
            "rogue": torch.tensor([[-0.8, 0.0], [-1.3, 0.0], [-1.8, 0.0], [-2.3, 0.0]]),
        }
        bundle = build_role_axis_bundle(
            role_vectors=role_vectors,
            honest_roles=["judge", "validator"],
            deceptive_roles=["spy", "rogue"],
            anchor_role="default",
            layer_specs=["4"],
        )
        layer_bundle = bundle["layers"]["4"]
        pc1 = layer_bundle["pc_components"][0]
        contrast_axis = layer_bundle["contrast_axis"]
        cosine = float(torch.dot(pc1, contrast_axis) / (torch.linalg.norm(pc1) * torch.linalg.norm(contrast_axis)))
        self.assertGreater(cosine, 0.0)

    def test_load_completion_mean_split_and_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            split_dir = Path(tmp_dir) / "Deception-AILiar-completion" / "test"
            split_dir.mkdir(parents=True)
            manifest_path = split_dir / "manifest.jsonl"
            rows = [
                {
                    "id": "sample-1",
                    "label": 1,
                    "user_end_idx": 2,
                    "completion_end_idx": 4,
                }
            ]
            manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            tensor = torch.tensor(
                [
                    [[0.0, 0.0], [0.0, 0.0], [2.0, 4.0], [4.0, 6.0]],
                    [[1.0, 1.0], [1.0, 1.0], [3.0, 5.0], [5.0, 7.0]],
                    [[2.0, 2.0], [2.0, 2.0], [4.0, 6.0], [6.0, 8.0]],
                    [[3.0, 3.0], [3.0, 3.0], [5.0, 7.0], [7.0, 9.0]],
                ],
                dtype=torch.float32,
            )
            save_file({"sample-1": tensor}, str(split_dir / "shard_0000.safetensors"))

            specs = resolve_layer_specs(4, ["4", "last4_mean"])
            cached = load_completion_mean_split(split_dir, specs)
            np.testing.assert_allclose(cached.features_by_spec["4"][0], np.array([6.0, 8.0], dtype=np.float32))
            np.testing.assert_allclose(cached.features_by_spec["last4_mean"][0], np.array([4.5, 6.5], dtype=np.float32))

            bundle = build_role_axis_bundle(
                role_vectors={
                    "default": torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
                    "judge": torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
                    "validator": torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
                    "spy": torch.tensor([[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]),
                    "rogue": torch.tensor([[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]),
                },
                honest_roles=["judge", "validator"],
                deceptive_roles=["spy", "rogue"],
                anchor_role="default",
                layer_specs=["4"],
            )
            score_payload = scores_for_layer(cached.features_by_spec["4"], bundle["layers"]["4"])
            self.assertEqual(score_payload["contrast"].shape, (1,))
            self.assertEqual(score_payload["pc_scores"].shape[0], 1)

    def test_load_completion_mean_split_scales_manifest_bounds_for_resampled_fullprompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            split_dir = Path(tmp_dir) / "Deception-AILiar-full" / "test"
            split_dir.mkdir(parents=True)
            manifest_path = split_dir / "manifest.jsonl"
            rows = [
                {
                    "id": "sample-1",
                    "label": 1,
                    "generation_length": 320,
                    "user_end_idx": 160,
                    "completion_end_idx": 320,
                }
            ]
            manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            values = torch.arange(1 * 64 * 2, dtype=torch.float32).reshape(1, 64, 2)
            save_file({"sample-1": values}, str(split_dir / "shard_0000.safetensors"))

            specs = resolve_layer_specs(1, ["1"])
            cached = load_completion_mean_split(split_dir, specs)
            expected = values[0, 32:64, :].mean(dim=0).numpy().astype(np.float32)
            np.testing.assert_allclose(cached.features_by_spec["1"][0], expected)

    def test_load_completion_mean_split_uses_full_tensor_for_completion_cache_when_manifest_bounds_are_incompatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            split_dir = Path(tmp_dir) / "Deception-AILiar-completion" / "test"
            split_dir.mkdir(parents=True)
            manifest_path = split_dir / "manifest.jsonl"
            rows = [
                {
                    "id": "sample-1",
                    "label": 1,
                    "generation_length": 227,
                    "user_end_idx": 72,
                    "completion_end_idx": 299,
                }
            ]
            manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            values = torch.arange(1 * 64 * 2, dtype=torch.float32).reshape(1, 64, 2)
            save_file({"sample-1": values}, str(split_dir / "shard_0000.safetensors"))

            specs = resolve_layer_specs(1, ["1"])
            cached = load_completion_mean_split(split_dir, specs)
            expected = values[0].mean(dim=0).numpy().astype(np.float32)
            np.testing.assert_allclose(cached.features_by_spec["1"][0], expected)

    def test_load_completion_mean_split_uses_full_tensor_when_completion_bounds_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            split_dir = Path(tmp_dir) / "Deception-ClaimsDefinitional-completion" / "test"
            split_dir.mkdir(parents=True)
            manifest_path = split_dir / "manifest.jsonl"
            rows = [
                {
                    "id": "sample-1",
                    "label": 1,
                    "generation_length": 64,
                    "user_end_idx": None,
                    "completion_end_idx": None,
                }
            ]
            manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            values = torch.arange(1 * 64 * 2, dtype=torch.float32).reshape(1, 64, 2)
            save_file({"sample-1": values}, str(split_dir / "shard_0000.safetensors"))

            specs = resolve_layer_specs(1, ["1"])
            cached = load_completion_mean_split(split_dir, specs)
            expected = values[0].mean(dim=0).numpy().astype(np.float32)
            np.testing.assert_allclose(cached.features_by_spec["1"][0], expected)

    def test_zero_shot_predictions_treat_negative_honest_score_as_deceptive(self) -> None:
        metrics, deceptive_scores, probabilities = evaluate_zero_shot(
            np.array([1, 0], dtype=np.int64),
            np.array([-2.0, 2.0], dtype=np.float32),
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertGreater(deceptive_scores[0], deceptive_scores[1])
        self.assertGreater(probabilities[0], 0.5)

    def test_write_role_axis_bundle_outputs_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle = build_role_axis_bundle(
                role_vectors={
                    "default": torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
                    "judge": torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
                    "validator": torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
                    "spy": torch.tensor([[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]),
                    "rogue": torch.tensor([[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]),
                },
                honest_roles=["judge", "validator"],
                deceptive_roles=["spy", "rogue"],
                anchor_role="default",
                layer_specs=["4"],
            )
            outputs = write_role_axis_bundle(Path(tmp_dir), bundle)
            self.assertTrue(Path(outputs["axis_bundle"]).exists())
            self.assertTrue(Path(outputs["bundle_json"]).exists())
            self.assertTrue(Path(outputs["layer_summary"]).exists())
            with Path(outputs["layer_summary"]).open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["layer_spec"], "4")

    def test_build_role_axis_bundle_accepts_selected_layer_metadata(self) -> None:
        bundle = build_role_axis_bundle(
            role_vectors={
                "default": torch.tensor([[0.0, 0.0]]),
                "judge": torch.tensor([[1.0, 0.0]]),
                "validator": torch.tensor([[1.0, 0.0]]),
                "spy": torch.tensor([[-1.0, 0.0]]),
                "rogue": torch.tensor([[-1.0, 0.0]]),
            },
            honest_roles=["judge", "validator"],
            deceptive_roles=["spy", "rogue"],
            anchor_role="default",
            layer_specs=["14"],
            layer_numbers=[14],
        )
        self.assertEqual(bundle["layer_count"], 1)
        self.assertEqual(bundle["activation_layer_numbers"], [14])
        self.assertEqual(bundle["resolved_layer_specs"][0]["layer_indices"], [0])
        self.assertEqual(bundle["resolved_layer_specs"][0]["layer_numbers"], [14])
        self.assertIn("14", bundle["layers"])

    def test_save_transfer_lineplots_writes_expected_figure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metric_rows = [
                {
                    "method": "contrast_zero_shot",
                    "layer_spec": "7",
                    "source_dataset": ZERO_SHOT_SOURCE,
                    "target_dataset": "Deception-AILiar-completion",
                    "auroc": "0.60",
                },
                {
                    "method": "contrast_zero_shot",
                    "layer_spec": "14",
                    "source_dataset": ZERO_SHOT_SOURCE,
                    "target_dataset": "Deception-AILiar-completion",
                    "auroc": "0.70",
                },
                {
                    "method": "pc1_zero_shot",
                    "layer_spec": "7",
                    "source_dataset": ZERO_SHOT_SOURCE,
                    "target_dataset": "Deception-AILiar-completion",
                    "auroc": "0.55",
                },
                {
                    "method": "pc1_zero_shot",
                    "layer_spec": "14",
                    "source_dataset": ZERO_SHOT_SOURCE,
                    "target_dataset": "Deception-AILiar-completion",
                    "auroc": "0.65",
                },
                {
                    "method": "pc123_linear",
                    "layer_spec": "7",
                    "source_dataset": "Deception-ConvincingGame-completion",
                    "target_dataset": "Deception-AILiar-completion",
                    "auroc": "0.62",
                },
                {
                    "method": "pc123_linear",
                    "layer_spec": "14",
                    "source_dataset": "Deception-ConvincingGame-completion",
                    "target_dataset": "Deception-AILiar-completion",
                    "auroc": "0.74",
                },
            ]
            outputs = save_transfer_lineplots(
                Path(tmp_dir),
                metric_rows,
                ["7", "14"],
                ["Deception-AILiar-completion", "Deception-ConvincingGame-completion"],
            )
            self.assertEqual(len(outputs), 1)
            self.assertTrue(Path(outputs[0]).exists())


if __name__ == "__main__":
    unittest.main()
