from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from deception_honesty_axis.activation_row_targets import (
    ensure_pooling_compatibility,
    ensure_single_layer_compatibility,
    load_activation_rows,
    validate_and_stack_activation_rows,
)
from deception_honesty_axis.activation_row_transfer_config import (
    ActivationRowDatasetConfig,
    ActivationRowSourceConfig,
)
from deception_honesty_axis.role_axis_transfer import (
    load_completion_mean_split,
    resolve_layer_specs,
)
from deception_honesty_axis.sycophancy_activations import normalize_activation_pooling


ACTIVATION_TRANSFER_METHODS = {"activation_logistic"}


@dataclass(frozen=True)
class LoadedActivationRowSplit:
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

    def summary(self) -> dict[str, Any]:
        return {
            "split_name": self.split_name,
            "record_count": int(self.labels.shape[0]),
            "feature_dim": int(self.feature_dim),
            "label_counts": dict(sorted(Counter(int(value) for value in self.labels.tolist()).items())),
            "activation_pooling": self.activation_pooling,
            "activation_layer_number": self.activation_layer_number,
            "group_field": self.group_field,
            "group_value": self.group_value,
            "source_kind": self.source_kind,
            "source_manifest": self.source_manifest,
        }


@dataclass(frozen=True)
class LoadedActivationRowDataset:
    name: str
    train: LoadedActivationRowSplit
    eval: LoadedActivationRowSplit

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "train": self.train.summary(),
            "eval": self.eval.summary(),
        }


def _select_rows(rows: list[dict[str, Any]], source: ActivationRowSourceConfig, dataset_name: str) -> list[dict[str, Any]]:
    if source.group_value is None:
        return rows
    filtered = [row for row in rows if str(row.get(source.group_field, "")) == source.group_value]
    if not filtered:
        raise ValueError(
            f"Dataset {dataset_name!r} requested group "
            f"{source.group_field}={source.group_value!r}, but no rows matched"
        )
    return filtered


def _stratified_row_split(
    rows: list[dict[str, Any]],
    source: ActivationRowSourceConfig,
    dataset_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if source.row_split_strategy is None:
        return rows, {}
    if source.row_split is None:
        raise ValueError(
            f"Dataset {dataset_name!r} uses row_split_strategy={source.row_split_strategy!r} "
            "but does not specify row_split"
        )

    by_label: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        label = int(row.get("label", -1))
        if label not in {0, 1}:
            raise ValueError(f"Expected binary label 0/1 for row split in {dataset_name!r}, got {label}")
        by_label.setdefault(label, []).append(index)
    if set(by_label) != {0, 1}:
        raise ValueError(f"Row split for dataset {dataset_name!r} requires both labels 0 and 1")

    rng = np.random.default_rng(source.row_split_seed)
    selected_indices: list[int] = []
    split_counts: dict[int, dict[str, int]] = {}
    for label, indices in sorted(by_label.items()):
        shuffled = np.asarray(indices, dtype=np.int64)
        rng.shuffle(shuffled)
        train_count = int(round(len(shuffled) * source.row_split_train_fraction))
        train_count = max(1, min(len(shuffled) - 1, train_count))
        train_indices = shuffled[:train_count]
        eval_indices = shuffled[train_count:]
        chosen = train_indices if source.row_split == "train" else eval_indices
        selected_indices.extend(int(index) for index in chosen.tolist())
        split_counts[int(label)] = {
            "total": int(len(shuffled)),
            "train": int(train_indices.shape[0]),
            "eval": int(eval_indices.shape[0]),
        }

    selected = sorted(selected_indices)
    manifest = {
        "row_split_strategy": source.row_split_strategy,
        "row_split": source.row_split,
        "row_split_train_fraction": source.row_split_train_fraction,
        "row_split_seed": source.row_split_seed,
        "row_split_label_counts": split_counts,
        "row_split_total_count": int(len(rows)),
        "row_split_selected_count": int(len(selected)),
    }
    return [rows[index] for index in selected], manifest


def _resolve_completion_split_layer_spec(split_dir: Path, expected_layer_number: int | None):  # noqa: ANN202
    from safetensors.torch import load_file

    shard_paths = sorted(split_dir.glob("shard_*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.safetensors found in {split_dir}")
    first_shard = load_file(str(shard_paths[0]))
    if not first_shard:
        raise ValueError(f"No tensors found in {shard_paths[0]}")
    first_tensor = next(iter(first_shard.values()))
    layer_count = int(first_tensor.shape[0])
    if layer_count == 1 and expected_layer_number is not None:
        from deception_honesty_axis.role_axis_transfer import ResolvedLayerSpec

        return ResolvedLayerSpec(
            key=str(expected_layer_number),
            label=f"L{expected_layer_number}",
            layer_indices=(0,),
            layer_numbers=(int(expected_layer_number),),
        )
    if expected_layer_number is None:
        if layer_count != 1:
            raise ValueError(
                "completion_split sources with multi-layer tensors require expected_layer_number in the config"
            )
        expected_layer_number = 1
    return resolve_layer_specs(layer_count, [str(expected_layer_number)])[0]


def load_activation_row_split(
    dataset_name: str,
    split_name: str,
    source: ActivationRowSourceConfig,
    *,
    expected_pooling: str | None,
    expected_layer_number: int | None,
) -> LoadedActivationRowSplit:
    if source.source_kind == "completion_split":
        if source.split_dir is None:
            raise ValueError(f"Dataset {dataset_name!r} split {split_name!r} requires split_dir for completion_split")
        resolved_spec = _resolve_completion_split_layer_spec(source.split_dir, expected_layer_number)
        cached = load_completion_mean_split(source.split_dir, [resolved_spec])
        activation_pooling = normalize_activation_pooling("completion_mean")
        ensure_pooling_compatibility(
            activation_pooling=activation_pooling,
            expected_pooling=expected_pooling,
            context=f"{dataset_name} {split_name} completion split",
        )
        if expected_layer_number is not None:
            ensure_single_layer_compatibility(
                activation_layer_number=int(resolved_spec.layer_numbers[0]),
                expected_layer_numbers=[expected_layer_number],
                context=f"{dataset_name} {split_name} completion split",
            )
        return LoadedActivationRowSplit(
            split_name=split_name,
            features=cached.features_by_spec[resolved_spec.key],
            labels=cached.labels,
            sample_ids=cached.sample_ids,
            activation_pooling=activation_pooling,
            activation_layer_number=int(resolved_spec.layer_numbers[0]),
            source_manifest={
                "source_kind": "completion_split",
                "split_dir": str(source.split_dir),
                "requested_split_name": split_name,
                "resolved_layer_spec": resolved_spec.to_manifest(),
                "dataset": cached.dataset,
                "cached_split": cached.split,
            },
            group_field=source.group_field,
            group_value=source.group_value,
            feature_dim=int(cached.features_by_spec[resolved_spec.key].shape[1]),
            source_kind="completion_split",
        )

    rows, source_manifest = load_activation_rows(**source.resolved_load_kwargs())
    selected_rows = _select_rows(rows, source, dataset_name)
    selected_rows, row_split_manifest = _stratified_row_split(selected_rows, source, dataset_name)
    stacked = validate_and_stack_activation_rows(
        selected_rows,
        group_field=source.group_field,
        source_manifest=source_manifest,
    )
    ensure_pooling_compatibility(
        activation_pooling=stacked.activation_pooling,
        expected_pooling=expected_pooling,
        context=f"{dataset_name} {split_name} activation rows",
    )
    if expected_layer_number is not None:
        ensure_single_layer_compatibility(
            activation_layer_number=stacked.activation_layer_number,
            expected_layer_numbers=[expected_layer_number],
            context=f"{dataset_name} {split_name} activation rows",
        )
    return LoadedActivationRowSplit(
        split_name=split_name,
        features=stacked.features,
        labels=stacked.labels,
        sample_ids=stacked.sample_ids,
        activation_pooling=stacked.activation_pooling,
        activation_layer_number=stacked.activation_layer_number,
        source_manifest={
            **source_manifest,
            "source_kind": "activation_rows",
            "requested_split_name": split_name,
            "selection_group_field": source.group_field,
            "selection_group_value": source.group_value,
            **row_split_manifest,
        },
        group_field=source.group_field,
        group_value=source.group_value,
        feature_dim=int(stacked.features.shape[1]),
        source_kind="activation_rows",
    )


def load_activation_row_dataset(
    dataset_config: ActivationRowDatasetConfig,
    *,
    expected_pooling: str | None,
    expected_layer_number: int | None,
) -> LoadedActivationRowDataset:
    train_split = load_activation_row_split(
        dataset_config.name,
        "train",
        dataset_config.train_source,
        expected_pooling=expected_pooling,
        expected_layer_number=expected_layer_number,
    )
    eval_split = load_activation_row_split(
        dataset_config.name,
        "eval",
        dataset_config.eval_source,
        expected_pooling=expected_pooling,
        expected_layer_number=expected_layer_number,
    )
    return LoadedActivationRowDataset(
        name=dataset_config.name,
        train=train_split,
        eval=eval_split,
    )


def ensure_loaded_dataset_compatibility(datasets: list[LoadedActivationRowDataset]) -> dict[str, Any]:
    if not datasets:
        raise ValueError("At least one loaded activation-row dataset is required")

    feature_dims = {
        split.feature_dim
        for dataset in datasets
        for split in (dataset.train, dataset.eval)
    }
    if len(feature_dims) != 1:
        raise ValueError(f"Inconsistent activation feature dimensions across datasets: {sorted(feature_dims)!r}")

    poolings = {
        split.activation_pooling
        for dataset in datasets
        for split in (dataset.train, dataset.eval)
        if split.activation_pooling is not None
    }
    if len(poolings) > 1:
        raise ValueError(f"Inconsistent activation pooling across datasets: {sorted(poolings)!r}")

    layer_numbers = {
        int(split.activation_layer_number)
        for dataset in datasets
        for split in (dataset.train, dataset.eval)
        if split.activation_layer_number is not None
    }
    if len(layer_numbers) > 1:
        raise ValueError(f"Inconsistent activation layers across datasets: {sorted(layer_numbers)!r}")

    for dataset in datasets:
        source_labels = {int(value) for value in dataset.train.labels.tolist()}
        if source_labels != {0, 1}:
            raise ValueError(
                f"Training split for dataset {dataset.name!r} must contain both labels 0 and 1; "
                f"found {sorted(source_labels)!r}"
            )

    return {
        "feature_dim": int(next(iter(feature_dims))),
        "activation_pooling": next(iter(poolings)) if poolings else None,
        "activation_layer_number": next(iter(layer_numbers)) if layer_numbers else None,
    }
