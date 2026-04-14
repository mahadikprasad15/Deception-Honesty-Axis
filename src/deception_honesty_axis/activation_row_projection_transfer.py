from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from deception_honesty_axis.activation_row_transfer import (
    LoadedActivationRowDataset,
    LoadedActivationRowSplit,
)
from deception_honesty_axis.common import ensure_dir, read_json, slugify, write_json
from deception_honesty_axis.role_axis_transfer import scores_for_layer


PROJECTED_TRANSFER_METHOD = "activation_logistic_pc_projection"


@dataclass(frozen=True)
class ProjectedActivationRowSplit:
    split_name: str
    features: np.ndarray
    labels: np.ndarray
    sample_ids: list[str]
    activation_pooling: str | None
    activation_layer_number: int | None
    source_manifest: dict[str, Any]
    group_field: str
    group_value: str | None
    feature_dim: int
    source_kind: str
    projection_manifest: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        unique, counts = np.unique(self.labels, return_counts=True)
        return {
            "split_name": self.split_name,
            "record_count": int(self.labels.shape[0]),
            "feature_dim": int(self.feature_dim),
            "label_counts": {int(label): int(count) for label, count in zip(unique.tolist(), counts.tolist(), strict=False)},
            "activation_pooling": self.activation_pooling,
            "activation_layer_number": self.activation_layer_number,
            "group_field": self.group_field,
            "group_value": self.group_value,
            "source_kind": self.source_kind,
            "source_manifest": self.source_manifest,
            "projection_manifest": self.projection_manifest,
        }


@dataclass(frozen=True)
class ProjectedActivationRowDataset:
    name: str
    train: ProjectedActivationRowSplit
    eval: ProjectedActivationRowSplit

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "train": self.train.summary(),
            "eval": self.eval.summary(),
        }


def resolve_axis_layer_bundle(
    axis_bundle: dict[str, Any],
    *,
    expected_layer_number: int | None,
) -> tuple[str, str, dict[str, Any]]:
    layer_specs = list(axis_bundle["layers"].keys())
    if expected_layer_number is not None:
        matches: list[str] = []
        for layer_spec in layer_specs:
            layer_numbers = [int(value) for value in axis_bundle["layers"][layer_spec].get("layer_numbers", [])]
            if len(layer_numbers) == 1 and int(layer_numbers[0]) == int(expected_layer_number):
                matches.append(layer_spec)
        if not matches:
            raise ValueError(
                f"No axis bundle layers match expected layer {expected_layer_number}; "
                f"bundle_layers={layer_specs!r}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Axis bundle matched multiple layer specs for expected layer {expected_layer_number}: {matches!r}"
            )
        layer_spec = matches[0]
    else:
        if len(layer_specs) != 1:
            raise ValueError(
                "Projected transfer requires exactly one resolved axis bundle layer when "
                f"expected_layer_number is omitted; found {layer_specs!r}"
            )
        layer_spec = layer_specs[0]

    layer_bundle = axis_bundle["layers"][layer_spec]
    layer_numbers = [int(value) for value in layer_bundle.get("layer_numbers", [])]
    if len(layer_numbers) == 1:
        layer_label = f"L{layer_numbers[0]}"
    else:
        layer_label = str(layer_spec)
    return str(layer_spec), layer_label, layer_bundle


def project_pc_features(
    features: np.ndarray,
    layer_bundle: dict[str, Any],
    *,
    max_pcs: int | None,
) -> tuple[np.ndarray, list[int]]:
    pc_scores = scores_for_layer(features, layer_bundle)["pc_scores"].astype(np.float32, copy=False)
    total_pc_count = int(pc_scores.shape[1])
    if total_pc_count <= 0:
        raise ValueError("Axis bundle does not contain any PC components")

    selected_pc_count = total_pc_count if max_pcs is None else min(int(max_pcs), total_pc_count)
    if selected_pc_count <= 0:
        raise ValueError(f"Projected transfer requires at least one PC, got max_pcs={max_pcs!r}")

    selected = pc_scores[:, :selected_pc_count].astype(np.float32, copy=False)
    return selected, list(range(1, selected_pc_count + 1))


def projected_split_artifact_paths(
    projected_root: Path,
    dataset_name: str,
    split_name: str,
) -> tuple[Path, Path]:
    dataset_dir = ensure_dir(projected_root / slugify(dataset_name))
    return dataset_dir / f"{slugify(split_name)}.npz", dataset_dir / f"{slugify(split_name)}.json"


def save_projected_split(
    array_path: Path,
    manifest_path: Path,
    split: ProjectedActivationRowSplit,
) -> dict[str, str]:
    ensure_dir(array_path.parent)
    np.savez_compressed(
        array_path,
        features=split.features.astype(np.float32, copy=False),
        labels=split.labels.astype(np.int64, copy=False),
        sample_ids=np.asarray(split.sample_ids, dtype=np.str_),
    )
    write_json(
        manifest_path,
        {
            "split_name": split.split_name,
            "activation_pooling": split.activation_pooling,
            "activation_layer_number": split.activation_layer_number,
            "group_field": split.group_field,
            "group_value": split.group_value,
            "feature_dim": split.feature_dim,
            "source_kind": split.source_kind,
            "source_manifest": split.source_manifest,
            "projection_manifest": split.projection_manifest,
        },
    )
    return {
        "array_path": str(array_path),
        "manifest_path": str(manifest_path),
    }


def load_projected_split(
    array_path: Path,
    manifest_path: Path,
) -> ProjectedActivationRowSplit:
    payload = np.load(array_path, allow_pickle=False)
    manifest = read_json(manifest_path)
    return ProjectedActivationRowSplit(
        split_name=str(manifest["split_name"]),
        features=np.asarray(payload["features"], dtype=np.float32),
        labels=np.asarray(payload["labels"], dtype=np.int64),
        sample_ids=[str(value) for value in payload["sample_ids"].tolist()],
        activation_pooling=manifest.get("activation_pooling"),
        activation_layer_number=(
            None if manifest.get("activation_layer_number") is None else int(manifest["activation_layer_number"])
        ),
        source_manifest=dict(manifest.get("source_manifest") or {}),
        group_field=str(manifest.get("group_field") or "dataset_source"),
        group_value=manifest.get("group_value"),
        feature_dim=int(manifest["feature_dim"]),
        source_kind=str(manifest.get("source_kind") or "unknown"),
        projection_manifest=dict(manifest.get("projection_manifest") or {}),
    )


def _project_loaded_split(
    split: LoadedActivationRowSplit,
    *,
    layer_bundle: dict[str, Any],
    layer_spec: str,
    layer_label: str,
    max_pcs: int | None,
    axis_bundle_run_dir: Path,
) -> ProjectedActivationRowSplit:
    projected_features, selected_pc_indices = project_pc_features(
        split.features,
        layer_bundle,
        max_pcs=max_pcs,
    )
    projection_manifest = {
        "axis_bundle_run_dir": str(axis_bundle_run_dir),
        "layer_spec": str(layer_spec),
        "layer_label": str(layer_label),
        "selected_pc_indices": selected_pc_indices,
        "selected_pc_labels": [f"PC{index}" for index in selected_pc_indices],
        "selected_pc_count": len(selected_pc_indices),
        "raw_feature_dim": int(split.feature_dim),
        "projected_feature_dim": int(projected_features.shape[1]),
    }
    return ProjectedActivationRowSplit(
        split_name=split.split_name,
        features=projected_features,
        labels=split.labels.copy(),
        sample_ids=list(split.sample_ids),
        activation_pooling=split.activation_pooling,
        activation_layer_number=split.activation_layer_number,
        source_manifest=dict(split.source_manifest),
        group_field=split.group_field,
        group_value=split.group_value,
        feature_dim=int(projected_features.shape[1]),
        source_kind=split.source_kind,
        projection_manifest=projection_manifest,
    )


def project_loaded_dataset(
    dataset: LoadedActivationRowDataset,
    *,
    layer_bundle: dict[str, Any],
    layer_spec: str,
    layer_label: str,
    max_pcs: int | None,
    axis_bundle_run_dir: Path,
    projected_root: Path,
) -> ProjectedActivationRowDataset:
    train_array_path, train_manifest_path = projected_split_artifact_paths(projected_root, dataset.name, "train")
    eval_array_path, eval_manifest_path = projected_split_artifact_paths(projected_root, dataset.name, "eval")

    if train_array_path.exists() and train_manifest_path.exists():
        projected_train = load_projected_split(train_array_path, train_manifest_path)
    else:
        projected_train = _project_loaded_split(
            dataset.train,
            layer_bundle=layer_bundle,
            layer_spec=layer_spec,
            layer_label=layer_label,
            max_pcs=max_pcs,
            axis_bundle_run_dir=axis_bundle_run_dir,
        )
        save_projected_split(train_array_path, train_manifest_path, projected_train)

    if eval_array_path.exists() and eval_manifest_path.exists():
        projected_eval = load_projected_split(eval_array_path, eval_manifest_path)
    else:
        projected_eval = _project_loaded_split(
            dataset.eval,
            layer_bundle=layer_bundle,
            layer_spec=layer_spec,
            layer_label=layer_label,
            max_pcs=max_pcs,
            axis_bundle_run_dir=axis_bundle_run_dir,
        )
        save_projected_split(eval_array_path, eval_manifest_path, projected_eval)

    return ProjectedActivationRowDataset(
        name=dataset.name,
        train=projected_train,
        eval=projected_eval,
    )


def ensure_projected_dataset_compatibility(datasets: list[ProjectedActivationRowDataset]) -> dict[str, Any]:
    if not datasets:
        raise ValueError("At least one projected dataset is required")

    feature_dims = {
        split.feature_dim
        for dataset in datasets
        for split in (dataset.train, dataset.eval)
    }
    if len(feature_dims) != 1:
        raise ValueError(f"Inconsistent projected feature dimensions across datasets: {sorted(feature_dims)!r}")

    for dataset in datasets:
        source_labels = {int(value) for value in dataset.train.labels.tolist()}
        if source_labels != {0, 1}:
            raise ValueError(
                f"Projected training split for dataset {dataset.name!r} must contain both labels 0 and 1; "
                f"found {sorted(source_labels)!r}"
            )

    first_projection = datasets[0].train.projection_manifest
    return {
        "projected_feature_dim": int(next(iter(feature_dims))),
        "selected_pc_count": int(first_projection.get("selected_pc_count") or next(iter(feature_dims))),
        "selected_pc_indices": list(first_projection.get("selected_pc_indices") or []),
    }
