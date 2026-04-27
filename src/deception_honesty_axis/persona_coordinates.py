from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from deception_honesty_axis.activation_row_transfer import LoadedActivationRowDataset, LoadedActivationRowSplit
from deception_honesty_axis.common import ensure_dir, read_json, slugify, write_json


PERSONA_COORDINATE_METHOD = "persona_coordinate_logistic"


@dataclass(frozen=True)
class PersonaCoordinateSplit:
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
    coordinate_manifest: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        unique, counts = np.unique(self.labels, return_counts=True)
        return {
            "split_name": self.split_name,
            "record_count": int(self.labels.shape[0]),
            "feature_dim": int(self.feature_dim),
            "label_counts": {
                int(label): int(count)
                for label, count in zip(unique.tolist(), counts.tolist(), strict=False)
            },
            "activation_pooling": self.activation_pooling,
            "activation_layer_number": self.activation_layer_number,
            "group_field": self.group_field,
            "group_value": self.group_value,
            "source_kind": self.source_kind,
            "source_manifest": self.source_manifest,
            "coordinate_manifest": self.coordinate_manifest,
        }


@dataclass(frozen=True)
class PersonaCoordinateDataset:
    name: str
    train: PersonaCoordinateSplit
    eval: PersonaCoordinateSplit

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "train": self.train.summary(),
            "eval": self.eval.summary(),
        }


@dataclass(frozen=True)
class PersonaRoleMatrix:
    role_names: list[str]
    role_sides: dict[str, str]
    role_matrix: np.ndarray
    activation_layer_number: int | None
    source_run_dir: Path
    source_manifest: dict[str, Any]

    @property
    def feature_dim(self) -> int:
        return int(self.role_matrix.shape[0])


def _load_torch_role_vectors(run_dir: Path) -> dict[str, Any]:
    import torch

    path = run_dir / "results" / "role_vectors.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing unified role vectors at {path}")
    role_vectors = torch.load(path, map_location="cpu")
    if not role_vectors:
        raise ValueError(f"No role vectors were loaded from {path}")
    return role_vectors


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return value


def _read_optional_json_value(path: Path) -> Any:
    if not path.exists():
        return None
    return read_json(path)


def _selected_layer_index(
    *,
    available_layer_numbers: list[int],
    expected_layer_number: int | None,
    layer_count: int,
) -> tuple[int, int | None]:
    if expected_layer_number is None:
        if layer_count != 1:
            raise ValueError(
                "Persona coordinate role vectors contain multiple layers; configure "
                "activation_rows.expected_layer_number to select one."
            )
        resolved_layer_number = available_layer_numbers[0] if available_layer_numbers else None
        return 0, resolved_layer_number

    if available_layer_numbers:
        if int(expected_layer_number) not in available_layer_numbers:
            raise ValueError(
                f"Requested layer {expected_layer_number} is unavailable in role vectors; "
                f"available layers={available_layer_numbers!r}"
            )
        return available_layer_numbers.index(int(expected_layer_number)), int(expected_layer_number)

    if layer_count == 1:
        return 0, int(expected_layer_number)
    if 1 <= int(expected_layer_number) <= layer_count:
        return int(expected_layer_number) - 1, int(expected_layer_number)
    raise ValueError(
        f"Cannot resolve requested layer {expected_layer_number}; role vector layer_count={layer_count}"
    )


def load_persona_role_matrix(
    unified_axis_run_dir: Path,
    *,
    expected_layer_number: int | None,
    include_anchor: bool = False,
    role_order: list[str] | None = None,
) -> PersonaRoleMatrix:
    import torch

    run_dir = unified_axis_run_dir.resolve()
    role_vectors = _load_torch_role_vectors(run_dir)
    meta = _read_optional_json(run_dir / "results" / "role_vectors_meta.json")
    results = _read_optional_json(run_dir / "results" / "results.json")
    manifest = _read_optional_json(run_dir / "meta" / "run_manifest.json")
    source_manifests_value = _read_optional_json_value(run_dir / "inputs" / "source_manifests.json")
    source_manifests = source_manifests_value if isinstance(source_manifests_value, list) else []

    first_tensor = next(iter(role_vectors.values()))
    layer_count = int(first_tensor.shape[0])
    raw_layer_numbers = meta.get("activation_layer_numbers") or results.get("activation_layer_numbers") or []
    available_layer_numbers = [int(value) for value in raw_layer_numbers]
    layer_index, resolved_layer_number = _selected_layer_index(
        available_layer_numbers=available_layer_numbers,
        expected_layer_number=expected_layer_number,
        layer_count=layer_count,
    )

    safe_roles = [str(value) for value in results.get("safe_roles", [])]
    harmful_roles = [str(value) for value in results.get("harmful_roles", [])]
    if not safe_roles or not harmful_roles:
        for source in source_manifests:
            safe_roles.extend(str(value) for value in source.get("safe_roles", []))
            harmful_roles.extend(str(value) for value in source.get("harmful_roles", []))

    side_by_role = {role_name: "safe" for role_name in safe_roles}
    side_by_role.update({role_name: "harmful" for role_name in harmful_roles})
    anchor_role = str(results.get("anchor_role") or manifest.get("anchor_role") or "default")
    if include_anchor and anchor_role in role_vectors:
        side_by_role.setdefault(anchor_role, "anchor")

    if role_order is None:
        ordered_roles = [role for role in safe_roles + harmful_roles if role in role_vectors]
        if include_anchor and anchor_role in role_vectors and anchor_role not in ordered_roles:
            ordered_roles.insert(0, anchor_role)
        if not ordered_roles:
            ordered_roles = sorted(
                role for role in role_vectors
                if include_anchor or role != anchor_role
            )
    else:
        ordered_roles = [str(role) for role in role_order]

    missing = [role for role in ordered_roles if role not in role_vectors]
    if missing:
        raise KeyError(f"Requested persona coordinate roles are missing from role vectors: {missing!r}")

    vectors = []
    for role_name in ordered_roles:
        vector = role_vectors[role_name][layer_index]
        if not isinstance(vector, torch.Tensor):
            vector = torch.as_tensor(vector)
        vectors.append(vector.detach().cpu().to(torch.float32).numpy())
    role_matrix = np.stack(vectors, axis=1).astype(np.float32, copy=False)
    return PersonaRoleMatrix(
        role_names=ordered_roles,
        role_sides={role: side_by_role.get(role, "unknown") for role in ordered_roles},
        role_matrix=role_matrix,
        activation_layer_number=resolved_layer_number,
        source_run_dir=run_dir,
        source_manifest={
            "unified_axis_run_dir": str(run_dir),
            "role_vectors_path": str(run_dir / "results" / "role_vectors.pt"),
            "role_vectors_meta_path": str(run_dir / "results" / "role_vectors_meta.json"),
            "run_manifest_path": str(run_dir / "meta" / "run_manifest.json"),
            "axis_name": manifest.get("axis_name"),
            "variant_name": manifest.get("variant_name"),
            "question_mode": manifest.get("question_mode"),
            "expected_pooling": manifest.get("expected_pooling"),
            "source_manifests": source_manifests,
        },
    )


def persona_coordinate_split_artifact_paths(
    coordinate_root: Path,
    dataset_name: str,
    split_name: str,
) -> tuple[Path, Path]:
    dataset_dir = ensure_dir(coordinate_root / slugify(dataset_name))
    return dataset_dir / f"{slugify(split_name)}.npz", dataset_dir / f"{slugify(split_name)}.json"


def save_persona_coordinate_split(
    array_path: Path,
    manifest_path: Path,
    split: PersonaCoordinateSplit,
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
            "coordinate_manifest": split.coordinate_manifest,
        },
    )
    return {"array_path": str(array_path), "manifest_path": str(manifest_path)}


def load_persona_coordinate_split(
    array_path: Path,
    manifest_path: Path,
) -> PersonaCoordinateSplit:
    payload = np.load(array_path, allow_pickle=False)
    manifest = read_json(manifest_path)
    return PersonaCoordinateSplit(
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
        coordinate_manifest=dict(manifest.get("coordinate_manifest") or {}),
    )


def _coordinate_loaded_split(
    split: LoadedActivationRowSplit,
    *,
    role_matrix: PersonaRoleMatrix,
) -> PersonaCoordinateSplit:
    if int(split.features.shape[1]) != int(role_matrix.role_matrix.shape[0]):
        raise ValueError(
            f"Activation feature dim {split.features.shape[1]} does not match persona role vector "
            f"dim {role_matrix.role_matrix.shape[0]}"
        )
    coordinates = (split.features @ role_matrix.role_matrix).astype(np.float32, copy=False)
    coordinate_manifest = {
        "unified_axis_run_dir": str(role_matrix.source_run_dir),
        "role_names": role_matrix.role_names,
        "role_sides": role_matrix.role_sides,
        "selected_role_count": len(role_matrix.role_names),
        "selected_layer_number": role_matrix.activation_layer_number,
        "raw_feature_dim": int(split.feature_dim),
        "coordinate_feature_dim": int(coordinates.shape[1]),
        "feature_labels": [f"role:{role_name}" for role_name in role_matrix.role_names],
    }
    return PersonaCoordinateSplit(
        split_name=split.split_name,
        features=coordinates,
        labels=split.labels.copy(),
        sample_ids=list(split.sample_ids),
        activation_pooling=split.activation_pooling,
        activation_layer_number=split.activation_layer_number,
        source_manifest=dict(split.source_manifest),
        group_field=split.group_field,
        group_value=split.group_value,
        feature_dim=int(coordinates.shape[1]),
        source_kind=split.source_kind,
        coordinate_manifest=coordinate_manifest,
    )


def coordinate_loaded_dataset(
    dataset: LoadedActivationRowDataset,
    *,
    role_matrix: PersonaRoleMatrix,
    coordinate_root: Path,
) -> PersonaCoordinateDataset:
    train_array_path, train_manifest_path = persona_coordinate_split_artifact_paths(
        coordinate_root,
        dataset.name,
        "train",
    )
    eval_array_path, eval_manifest_path = persona_coordinate_split_artifact_paths(
        coordinate_root,
        dataset.name,
        "eval",
    )

    if train_array_path.exists() and train_manifest_path.exists():
        coordinate_train = load_persona_coordinate_split(train_array_path, train_manifest_path)
    else:
        coordinate_train = _coordinate_loaded_split(dataset.train, role_matrix=role_matrix)
        save_persona_coordinate_split(train_array_path, train_manifest_path, coordinate_train)

    if eval_array_path.exists() and eval_manifest_path.exists():
        coordinate_eval = load_persona_coordinate_split(eval_array_path, eval_manifest_path)
    else:
        coordinate_eval = _coordinate_loaded_split(dataset.eval, role_matrix=role_matrix)
        save_persona_coordinate_split(eval_array_path, eval_manifest_path, coordinate_eval)

    return PersonaCoordinateDataset(
        name=dataset.name,
        train=coordinate_train,
        eval=coordinate_eval,
    )


def ensure_persona_coordinate_dataset_compatibility(
    datasets: list[PersonaCoordinateDataset],
) -> dict[str, Any]:
    if not datasets:
        raise ValueError("At least one persona-coordinate dataset is required")

    feature_dims = {
        split.feature_dim
        for dataset in datasets
        for split in (dataset.train, dataset.eval)
    }
    if len(feature_dims) != 1:
        raise ValueError(f"Inconsistent persona-coordinate feature dimensions: {sorted(feature_dims)!r}")

    for dataset in datasets:
        source_labels = {int(value) for value in dataset.train.labels.tolist()}
        if source_labels != {0, 1}:
            raise ValueError(
                f"Persona-coordinate training split for dataset {dataset.name!r} must contain both labels 0 and 1; "
                f"found {sorted(source_labels)!r}"
            )

    first_manifest = datasets[0].train.coordinate_manifest
    return {
        "coordinate_feature_dim": int(next(iter(feature_dims))),
        "role_names": list(first_manifest.get("role_names") or []),
        "role_sides": dict(first_manifest.get("role_sides") or {}),
    }
