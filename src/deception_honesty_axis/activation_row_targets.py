from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np

from deception_honesty_axis.common import load_jsonl, read_json
from deception_honesty_axis.sycophancy_activations import normalize_activation_pooling


@dataclass(frozen=True)
class StackedActivationRows:
    features: np.ndarray
    labels: np.ndarray
    sample_ids: list[str]
    group_values: list[str]
    activation_pooling: str | None
    activation_layer_number: int | None


def _coerce_single_layer_number(value: Any) -> int | None:
    if value in (None, "", []):
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected exactly one activation layer number, got {value!r}")
        return int(value[0])
    return int(value)


def _read_local_run_metadata(path: Path) -> dict[str, Any]:
    run_root = None
    if path.is_dir() and path.parent.name == "results":
        run_root = path.parent.parent
    elif path.is_file() and path.parent.name == "results":
        run_root = path.parent.parent
    if run_root is None:
        return {}

    manifest_path = run_root / "meta" / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = read_json(manifest_path)
    metadata: dict[str, Any] = {"run_manifest_path": str(manifest_path.resolve())}
    pooling = normalize_activation_pooling(
        manifest.get("activation_pooling")
        or manifest.get("pooling")
        or manifest.get("activation_position")
    )
    if pooling is not None:
        metadata["activation_pooling"] = pooling
    layer_number = _coerce_single_layer_number(
        manifest.get("activation_layer_number")
        or manifest.get("layer")
        or manifest.get("activation_layer_numbers")
    )
    if layer_number is not None:
        metadata["activation_layer_number"] = layer_number
    return metadata


def load_activation_rows(
    *,
    activation_dataset_repo_id: str | None = None,
    activation_dataset_dir: Path | None = None,
    activation_jsonl: Path | None = None,
    split: str = "train",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = [
        bool(activation_dataset_repo_id),
        bool(activation_dataset_dir),
        bool(activation_jsonl),
    ]
    if sum(sources) != 1:
        raise ValueError(
            "Specify exactly one of activation_dataset_repo_id, activation_dataset_dir, or activation_jsonl"
        )

    if activation_jsonl:
        path = activation_jsonl.resolve()
        return load_jsonl(path), {
            "source_type": "jsonl",
            "path": str(path),
            **_read_local_run_metadata(path),
        }

    from datasets import load_dataset, load_from_disk

    if activation_dataset_dir:
        path = activation_dataset_dir.resolve()
        dataset = load_from_disk(str(path))
        return [dict(row) for row in dataset], {
            "source_type": "local_dataset",
            "path": str(path),
            **_read_local_run_metadata(path),
        }

    assert activation_dataset_repo_id is not None
    dataset = load_dataset(activation_dataset_repo_id, split=split, token=os.environ.get("HF_TOKEN"))
    return [dict(row) for row in dataset], {
        "source_type": "hf_dataset",
        "repo_id": activation_dataset_repo_id,
        "split": split,
    }


def _constant_row_value(rows: list[dict[str, Any]], key: str) -> Any:
    values = {row.get(key) for row in rows if row.get(key) not in (None, "", [])}
    if not values:
        return None
    if len(values) > 1:
        raise ValueError(f"Inconsistent {key} values in activation rows: {sorted(values)!r}")
    return next(iter(values))


def validate_and_stack_activation_rows(
    rows: list[dict[str, Any]],
    *,
    group_field: str,
    source_manifest: dict[str, Any] | None = None,
) -> StackedActivationRows:
    if not rows:
        raise ValueError("No activation rows loaded")

    features: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    group_values: list[str] = []
    for row in rows:
        activation = np.asarray(row.get("activation"), dtype=np.float32)
        if activation.ndim != 1:
            raise ValueError(
                f"Expected 1D activation for sample {row.get('ids')}, got shape {activation.shape}"
            )
        label = int(row.get("label", -1))
        if label not in {0, 1}:
            raise ValueError(f"Expected binary label 0/1 for sample {row.get('ids')}, got {label}")
        features.append(activation)
        labels.append(label)
        sample_ids.append(str(row.get("ids", "")))
        group_values.append(str(row.get(group_field, "unknown")))

    manifest = source_manifest or {}
    row_pooling = normalize_activation_pooling(_constant_row_value(rows, "activation_pooling"))
    manifest_pooling = normalize_activation_pooling(manifest.get("activation_pooling"))
    if row_pooling is not None and manifest_pooling is not None and row_pooling != manifest_pooling:
        raise ValueError(
            f"Activation pooling mismatch between rows ({row_pooling}) and manifest ({manifest_pooling})"
        )

    row_layer = _coerce_single_layer_number(_constant_row_value(rows, "activation_layer_number"))
    manifest_layer = _coerce_single_layer_number(manifest.get("activation_layer_number"))
    if row_layer is not None and manifest_layer is not None and row_layer != manifest_layer:
        raise ValueError(
            f"Activation layer mismatch between rows ({row_layer}) and manifest ({manifest_layer})"
        )

    return StackedActivationRows(
        features=np.stack(features).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        sample_ids=sample_ids,
        group_values=group_values,
        activation_pooling=row_pooling if row_pooling is not None else manifest_pooling,
        activation_layer_number=row_layer if row_layer is not None else manifest_layer,
    )


def grouped_indices(group_values: list[str]) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {"all": list(range(len(group_values)))}
    for index, group_value in enumerate(group_values):
        groups.setdefault(group_value, []).append(index)
    return {
        name: np.asarray(indices, dtype=np.int64)
        for name, indices in groups.items()
        if indices
    }


def ensure_pooling_compatibility(
    *,
    activation_pooling: str | None,
    expected_pooling: str | None,
    context: str,
) -> None:
    normalized_expected = normalize_activation_pooling(expected_pooling)
    if activation_pooling is None or normalized_expected is None:
        return
    if activation_pooling != normalized_expected:
        raise ValueError(
            f"{context} uses activation pooling {activation_pooling!r}, but the axis expects {normalized_expected!r}"
        )


def ensure_single_layer_compatibility(
    *,
    activation_layer_number: int | None,
    expected_layer_numbers: list[int] | tuple[int, ...],
    context: str,
) -> None:
    if activation_layer_number is None:
        return
    if len(expected_layer_numbers) != 1 or int(expected_layer_numbers[0]) != activation_layer_number:
        raise ValueError(
            f"{context} stores activation layer {activation_layer_number}, "
            f"which is incompatible with layer spec {list(expected_layer_numbers)!r}"
        )
