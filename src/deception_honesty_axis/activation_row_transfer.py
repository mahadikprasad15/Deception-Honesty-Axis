from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
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


def load_activation_row_split(
    dataset_name: str,
    split_name: str,
    source: ActivationRowSourceConfig,
    *,
    expected_pooling: str | None,
    expected_layer_number: int | None,
) -> LoadedActivationRowSplit:
    rows, source_manifest = load_activation_rows(**source.resolved_load_kwargs())
    selected_rows = _select_rows(rows, source, dataset_name)
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
            "requested_split_name": split_name,
            "selection_group_field": source.group_field,
            "selection_group_value": source.group_value,
        },
        group_field=source.group_field,
        group_value=source.group_value,
        feature_dim=int(stacked.features.shape[1]),
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
