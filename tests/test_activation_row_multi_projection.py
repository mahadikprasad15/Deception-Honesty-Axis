from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from deception_honesty_axis.activation_row_projection_transfer import (
    AxisProjectionSpec,
    project_loaded_dataset_multi,
    project_multi_axis_features,
)
from deception_honesty_axis.activation_row_transfer import LoadedActivationRowDataset, LoadedActivationRowSplit


class ActivationRowMultiProjectionTest(unittest.TestCase):
    def _split(self, name: str, features: np.ndarray, labels: np.ndarray) -> LoadedActivationRowSplit:
        return LoadedActivationRowSplit(
            split_name=name,
            features=features.astype(np.float32),
            labels=labels.astype(np.int64),
            sample_ids=[f"{name}-{index}" for index in range(features.shape[0])],
            activation_pooling="mean_response",
            activation_layer_number=14,
            source_manifest={"split": name},
            group_field="dataset_source",
            group_value=None,
            feature_dim=int(features.shape[1]),
            source_kind="completion_split",
        )

    def test_project_multi_axis_features_concatenates_axis_projections(self) -> None:
        features = np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            dtype=np.float32,
        )
        axis_specs = [
            AxisProjectionSpec(
                axis_bundle_run_dir=Path("/tmp/axis_a"),
                axis_slug="axis-a",
                layer_spec="14",
                layer_label="L14",
                layer_bundle={
                    "contrast_axis": torch.zeros(4),
                    "pca_mean": torch.zeros(4),
                    "pc_components": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
                },
                selected_pc_count=2,
                available_pc_count=2,
            ),
            AxisProjectionSpec(
                axis_bundle_run_dir=Path("/tmp/axis_b"),
                axis_slug="axis-b",
                layer_spec="14",
                layer_label="L14",
                layer_bundle={
                    "contrast_axis": torch.zeros(4),
                    "pca_mean": torch.zeros(4),
                    "pc_components": torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
                },
                selected_pc_count=2,
                available_pc_count=2,
            ),
        ]

        projected, axis_manifests, feature_labels = project_multi_axis_features(features, axis_specs)

        self.assertEqual(projected.shape, (2, 4))
        np.testing.assert_allclose(projected[0], np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        self.assertEqual(len(axis_manifests), 2)
        self.assertEqual(feature_labels, ["axis-a:PC1", "axis-a:PC2", "axis-b:PC1", "axis-b:PC2"])

    def test_project_loaded_dataset_multi_persists_combined_projection_metadata(self) -> None:
        dataset = LoadedActivationRowDataset(
            name="demo-dataset",
            train=self._split(
                "train",
                np.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
                np.asarray([0, 1], dtype=np.int64),
            ),
            eval=self._split(
                "eval",
                np.asarray([[6.0, 7.0, 8.0, 9.0], [7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
                np.asarray([0, 1], dtype=np.int64),
            ),
        )
        axis_specs = [
            AxisProjectionSpec(
                axis_bundle_run_dir=Path("/tmp/axis_a"),
                axis_slug="axis-a",
                layer_spec="14",
                layer_label="L14",
                layer_bundle={
                    "contrast_axis": torch.zeros(4),
                    "pca_mean": torch.zeros(4),
                    "pc_components": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                },
                selected_pc_count=1,
                available_pc_count=1,
            ),
            AxisProjectionSpec(
                axis_bundle_run_dir=Path("/tmp/axis_b"),
                axis_slug="axis-b",
                layer_spec="14",
                layer_label="L14",
                layer_bundle={
                    "contrast_axis": torch.zeros(4),
                    "pca_mean": torch.zeros(4),
                    "pc_components": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
                },
                selected_pc_count=1,
                available_pc_count=1,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            projected = project_loaded_dataset_multi(
                dataset,
                axis_projection_specs=axis_specs,
                projected_root=Path(tmp_dir),
            )

            self.assertEqual(projected.train.features.shape[1], 2)
            self.assertEqual(projected.eval.features.shape[1], 2)
            self.assertEqual(projected.train.projection_manifest["selected_feature_labels"], ["axis-a:PC1", "axis-b:PC1"])
            self.assertEqual(len(projected.train.projection_manifest["axes"]), 2)


if __name__ == "__main__":
    unittest.main()
