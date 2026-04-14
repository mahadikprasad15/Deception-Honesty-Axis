from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from deception_honesty_axis.unified_role_axis import (
    build_unified_role_axis_bundle_payload,
    load_unified_role_axis_config,
)


class UnifiedRoleAxisTest(unittest.TestCase):
    def test_build_unified_role_axis_native_mode_merges_sources_and_aligns_requested_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            (repo_root / ".git").mkdir(parents=True)
            (repo_root / "configs").mkdir(parents=True)

            quantity_run_dir = repo_root / "artifacts" / "runs" / "role-vectors" / "quantity" / "fixed"
            sycophancy_run_dir = repo_root / "artifacts" / "runs" / "role-vectors" / "sycophancy" / "fixed"
            (quantity_run_dir / "results").mkdir(parents=True)
            (sycophancy_run_dir / "results").mkdir(parents=True)

            torch.save(
                {
                    "default": torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float32),
                    "ombudsman": torch.tensor([[0.5, 0.5], [3.0, 0.0]], dtype=torch.float32),
                    "handler": torch.tensor([[-0.5, -0.5], [-3.0, 0.0]], dtype=torch.float32),
                },
                quantity_run_dir / "results" / "role_vectors.pt",
            )
            (quantity_run_dir / "results" / "role_vectors_meta.json").write_text(
                json.dumps(
                    {
                        "activation_layer_numbers": [13, 14],
                        "pooling": "mean_response",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            torch.save(
                {
                    "default": torch.tensor([[4.0, 4.0]], dtype=torch.float32),
                    "critic": torch.tensor([[2.0, 3.0]], dtype=torch.float32),
                    "yes_man": torch.tensor([[-2.0, -3.0]], dtype=torch.float32),
                },
                sycophancy_run_dir / "results" / "role_vectors.pt",
            )
            (sycophancy_run_dir / "results" / "role_vectors_meta.json").write_text(
                json.dumps(
                    {
                        "activation_layer_numbers": [14],
                        "pooling": "mean_response",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            config_path = repo_root / "configs" / "unified.json"
            config_path.write_text(
                json.dumps(
                    {
                        "artifacts": {"root": "artifacts"},
                        "experiment": {
                            "name": "unified-role-axis-bundles",
                            "behavior": "harmful-behavior",
                            "model_name": "meta-llama/Llama-3.2-3B-Instruct",
                            "axis_name": "quantity-sycophancy-unified",
                            "variant": "native-questions-by-behavior",
                        },
                        "axis": {
                            "question_mode": "native_questions_by_behavior",
                            "expected_pooling": "mean_response",
                            "layer_specs": ["14"],
                            "native_sources": [
                                {
                                    "name": "quantity_axis_v2",
                                    "behavior": "deception",
                                    "vectors_run_dir": str(quantity_run_dir),
                                    "safe_roles": ["ombudsman"],
                                    "harmful_roles": ["handler"],
                                    "anchor_role": "default",
                                },
                                {
                                    "name": "sycophancy_pilot_v1",
                                    "behavior": "sycophancy",
                                    "vectors_run_dir": str(sycophancy_run_dir),
                                    "safe_roles": ["critic"],
                                    "harmful_roles": ["yes_man"],
                                    "anchor_role": "default",
                                },
                            ],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            config = load_unified_role_axis_config(config_path)
            payload = build_unified_role_axis_bundle_payload(config)

            self.assertEqual(payload["activation_layer_numbers"], [14])
            self.assertEqual(sorted(payload["safe_roles"]), ["critic", "ombudsman"])
            self.assertEqual(sorted(payload["harmful_roles"]), ["handler", "yes_man"])
            self.assertTrue(torch.allclose(payload["role_vectors"]["default"], torch.tensor([[3.0, 3.0]])))
            self.assertEqual(payload["bundle"]["activation_layer_numbers"], [14])
            self.assertEqual(payload["bundle"]["pc_count"], 3)
            self.assertIn("14", payload["bundle"]["layers"])

    def test_shared_question_mode_requires_shared_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            (repo_root / ".git").mkdir(parents=True)
            (repo_root / "configs").mkdir(parents=True)
            config_path = repo_root / "configs" / "unified.json"
            config_path.write_text(
                json.dumps(
                    {
                        "artifacts": {"root": "artifacts"},
                        "experiment": {
                            "behavior": "harmful-behavior",
                            "model_name": "meta-llama/Llama-3.2-3B-Instruct",
                            "axis_name": "quantity-sycophancy-unified",
                        },
                        "axis": {
                            "question_mode": "shared_questions_balanced",
                            "layer_specs": ["14"],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_unified_role_axis_config(config_path).shared_source


if __name__ == "__main__":
    unittest.main()
