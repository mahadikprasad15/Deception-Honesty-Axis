from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deception_honesty_axis.common import ensure_dir
from deception_honesty_axis.config import find_repo_root, slugify
from deception_honesty_axis.role_axis_transfer import build_role_axis_bundle


SUPPORTED_UNIFIED_QUESTION_MODES = {
    "native_questions_by_behavior",
    "shared_questions_balanced",
}


@dataclass(frozen=True)
class UnifiedRoleSourceConfig:
    raw: dict[str, Any]
    repo_root: Path

    @property
    def name(self) -> str:
        value = self.raw.get("name")
        if value in (None, ""):
            raise ValueError("Each unified role source requires a non-empty name")
        return str(value)

    @property
    def behavior(self) -> str | None:
        value = self.raw.get("behavior")
        if value in (None, ""):
            return None
        return str(value)

    @property
    def vectors_run_dir(self) -> Path:
        value = self.raw.get("vectors_run_dir")
        if value in (None, ""):
            raise ValueError(f"Unified role source {self.name!r} requires vectors_run_dir")
        path = Path(str(value))
        return path if path.is_absolute() else (self.repo_root / path)

    @property
    def safe_roles(self) -> list[str]:
        values = self.raw.get("safe_roles")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Unified role source {self.name!r} requires a non-empty safe_roles list")
        return [str(value) for value in values]

    @property
    def harmful_roles(self) -> list[str]:
        values = self.raw.get("harmful_roles")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Unified role source {self.name!r} requires a non-empty harmful_roles list")
        return [str(value) for value in values]

    @property
    def anchor_role(self) -> str:
        return str(self.raw.get("anchor_role", "default"))

    def manifest_row(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "behavior": self.behavior,
            "vectors_run_dir": str(self.vectors_run_dir),
            "safe_roles": self.safe_roles,
            "harmful_roles": self.harmful_roles,
            "anchor_role": self.anchor_role,
        }


@dataclass(frozen=True)
class UnifiedRoleAxisConfig:
    raw: dict[str, Any]
    path: Path
    repo_root: Path

    @property
    def artifact_root(self) -> Path:
        value = self.raw.get("artifacts", {}).get("root", "artifacts")
        path = Path(str(value))
        resolved = path if path.is_absolute() else (self.repo_root / path)
        ensure_dir(resolved)
        return resolved

    @property
    def experiment_name(self) -> str:
        return str(self.raw.get("experiment", {}).get("name", "unified-role-axis-bundles"))

    @property
    def behavior_name(self) -> str:
        value = self.raw.get("experiment", {}).get("behavior")
        if value in (None, ""):
            raise ValueError("Unified role-axis config requires experiment.behavior")
        return str(value)

    @property
    def model_name(self) -> str:
        value = self.raw.get("experiment", {}).get("model_name")
        if value in (None, ""):
            raise ValueError("Unified role-axis config requires experiment.model_name")
        return str(value)

    @property
    def axis_name(self) -> str:
        value = self.raw.get("experiment", {}).get("axis_name")
        if value in (None, ""):
            raise ValueError("Unified role-axis config requires experiment.axis_name")
        return str(value)

    @property
    def question_mode(self) -> str:
        value = self.raw.get("axis", {}).get("question_mode")
        if value in (None, ""):
            raise ValueError("Unified role-axis config requires axis.question_mode")
        resolved = str(value)
        if resolved not in SUPPORTED_UNIFIED_QUESTION_MODES:
            raise ValueError(
                f"Unsupported question_mode {resolved!r}; expected one of {sorted(SUPPORTED_UNIFIED_QUESTION_MODES)!r}"
            )
        return resolved

    @property
    def variant_name(self) -> str:
        return str(self.raw.get("experiment", {}).get("variant", self.question_mode))

    @property
    def layer_specs(self) -> list[str]:
        values = self.raw.get("axis", {}).get("layer_specs")
        if not isinstance(values, list) or not values:
            raise ValueError("Unified role-axis config requires a non-empty axis.layer_specs list")
        return [str(value) for value in values]

    @property
    def expected_pooling(self) -> str | None:
        value = self.raw.get("axis", {}).get("expected_pooling")
        if value in (None, ""):
            return None
        return str(value)

    @property
    def pc_count(self) -> int:
        return int(self.raw.get("axis", {}).get("pc_count", 3))

    @property
    def question_sources(self) -> list[dict[str, Any]]:
        values = self.raw.get("axis", {}).get("question_sources", [])
        if not isinstance(values, list):
            raise ValueError("axis.question_sources must be a list when provided")
        return [dict(value) for value in values]

    @property
    def native_sources(self) -> list[UnifiedRoleSourceConfig]:
        values = self.raw.get("axis", {}).get("native_sources", [])
        if self.question_mode == "native_questions_by_behavior":
            if not isinstance(values, list) or not values:
                raise ValueError(
                    "Unified role-axis config with question_mode=native_questions_by_behavior "
                    "requires a non-empty axis.native_sources list"
                )
        return [UnifiedRoleSourceConfig(dict(value), self.repo_root) for value in values]

    @property
    def shared_source(self) -> UnifiedRoleSourceConfig | None:
        value = self.raw.get("axis", {}).get("shared_source")
        if value in (None, ""):
            if self.question_mode == "shared_questions_balanced":
                raise ValueError(
                    "Unified role-axis config with question_mode=shared_questions_balanced "
                    "requires axis.shared_source"
                )
            return None
        if not isinstance(value, dict):
            raise ValueError("axis.shared_source must be an object when provided")
        return UnifiedRoleSourceConfig(dict(value), self.repo_root)


def load_unified_role_axis_config(path: str | Path) -> UnifiedRoleAxisConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    repo_root = find_repo_root(config_path.parent)
    return UnifiedRoleAxisConfig(raw=raw, path=config_path, repo_root=repo_root)


def _load_role_vector_run(vectors_run_dir: Path) -> tuple[dict[str, Any], list[int]]:
    import torch

    role_vectors = torch.load(vectors_run_dir / "results" / "role_vectors.pt", map_location="cpu")
    if not role_vectors:
        raise ValueError(f"No role vectors found in {vectors_run_dir}")
    meta_path = vectors_run_dir / "results" / "role_vectors_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            raw_meta = json.load(handle)
        raw_layer_numbers = raw_meta.get("activation_layer_numbers", [])
        if raw_layer_numbers:
            return role_vectors, [int(value) for value in raw_layer_numbers]
    layer_count = int(next(iter(role_vectors.values())).shape[0])
    return role_vectors, list(range(1, layer_count + 1))


def _requested_numeric_layer_numbers(layer_specs: list[str]) -> list[int] | None:
    layer_numbers: list[int] = []
    for layer_spec in layer_specs:
        token = str(layer_spec).strip().lower()
        if not token.isdigit():
            return None
        layer_numbers.append(int(token))
    seen: set[int] = set()
    deduped: list[int] = []
    for layer_number in layer_numbers:
        if layer_number in seen:
            continue
        seen.add(layer_number)
        deduped.append(layer_number)
    return deduped


def _select_role_vector_layers(
    role_vectors: dict[str, Any],
    available_layer_numbers: list[int],
    requested_layer_numbers: list[int],
) -> dict[str, Any]:
    index_by_layer_number = {
        int(layer_number): layer_index
        for layer_index, layer_number in enumerate(available_layer_numbers)
    }
    missing = [layer_number for layer_number in requested_layer_numbers if layer_number not in index_by_layer_number]
    if missing:
        raise ValueError(
            f"Requested layer numbers {missing!r} are unavailable; available layers={available_layer_numbers!r}"
        )

    selected_indices = [index_by_layer_number[layer_number] for layer_number in requested_layer_numbers]
    return {
        role_name: tensor[selected_indices].to(dtype=tensor.dtype)
        for role_name, tensor in role_vectors.items()
    }


def _prepare_role_vectors_for_source(
    source: UnifiedRoleSourceConfig,
    layer_specs: list[str],
) -> tuple[dict[str, Any], list[int], dict[str, Any]]:
    role_vectors, available_layer_numbers = _load_role_vector_run(source.vectors_run_dir)
    requested_layer_numbers = _requested_numeric_layer_numbers(layer_specs)
    if requested_layer_numbers is None:
        selected_vectors = role_vectors
        selected_layer_numbers = available_layer_numbers
    else:
        selected_vectors = _select_role_vector_layers(role_vectors, available_layer_numbers, requested_layer_numbers)
        selected_layer_numbers = requested_layer_numbers

    required_roles = {source.anchor_role, *source.safe_roles, *source.harmful_roles}
    missing_roles = sorted(role_name for role_name in required_roles if role_name not in selected_vectors)
    if missing_roles:
        raise KeyError(
            f"Source {source.name!r} is missing required roles {missing_roles!r} in {source.vectors_run_dir}"
        )
    return selected_vectors, selected_layer_numbers, {
        **source.manifest_row(),
        "available_layer_numbers": available_layer_numbers,
        "selected_layer_numbers": selected_layer_numbers,
    }


def build_unified_role_axis_inputs(config: UnifiedRoleAxisConfig) -> dict[str, Any]:
    import torch

    if config.question_mode == "shared_questions_balanced":
        source = config.shared_source
        assert source is not None
        selected_vectors, selected_layer_numbers, source_manifest = _prepare_role_vectors_for_source(
            source,
            config.layer_specs,
        )
        return {
            "mode": config.question_mode,
            "role_vectors": selected_vectors,
            "safe_roles": source.safe_roles,
            "harmful_roles": source.harmful_roles,
            "anchor_role": source.anchor_role,
            "activation_layer_numbers": selected_layer_numbers,
            "source_manifests": [source_manifest],
        }

    merged_vectors: dict[str, Any] = {}
    anchor_vectors: list[Any] = []
    safe_roles: list[str] = []
    harmful_roles: list[str] = []
    source_manifests: list[dict[str, Any]] = []
    anchor_role: str | None = None
    selected_layer_numbers: list[int] | None = None

    for source in config.native_sources:
        selected_vectors, current_layer_numbers, source_manifest = _prepare_role_vectors_for_source(
            source,
            config.layer_specs,
        )
        if anchor_role is None:
            anchor_role = source.anchor_role
        elif anchor_role != source.anchor_role:
            raise ValueError(
                "All native unified sources must share the same anchor role; "
                f"found {anchor_role!r} and {source.anchor_role!r}"
            )

        if selected_layer_numbers is None:
            selected_layer_numbers = current_layer_numbers
        elif selected_layer_numbers != current_layer_numbers:
            raise ValueError(
                "All native unified sources must resolve to the same selected layer numbers; "
                f"found {selected_layer_numbers!r} and {current_layer_numbers!r}"
            )

        for role_name in source.safe_roles:
            if role_name in merged_vectors:
                raise ValueError(f"Duplicate safe role {role_name!r} across native unified sources")
            merged_vectors[role_name] = selected_vectors[role_name]
            safe_roles.append(role_name)
        for role_name in source.harmful_roles:
            if role_name in merged_vectors:
                raise ValueError(f"Duplicate harmful role {role_name!r} across native unified sources")
            merged_vectors[role_name] = selected_vectors[role_name]
            harmful_roles.append(role_name)
        anchor_vectors.append(selected_vectors[source.anchor_role].to(torch.float32))
        source_manifests.append(source_manifest)

    if anchor_role is None or selected_layer_numbers is None or not anchor_vectors:
        raise ValueError("No native unified sources were resolved")

    merged_vectors[anchor_role] = torch.stack(anchor_vectors, dim=0).mean(dim=0)
    return {
        "mode": config.question_mode,
        "role_vectors": merged_vectors,
        "safe_roles": safe_roles,
        "harmful_roles": harmful_roles,
        "anchor_role": anchor_role,
        "activation_layer_numbers": selected_layer_numbers,
        "source_manifests": source_manifests,
    }


def build_unified_role_axis_bundle_payload(config: UnifiedRoleAxisConfig) -> dict[str, Any]:
    inputs = build_unified_role_axis_inputs(config)
    bundle = build_role_axis_bundle(
        role_vectors=inputs["role_vectors"],
        honest_roles=inputs["safe_roles"],
        deceptive_roles=inputs["harmful_roles"],
        anchor_role=inputs["anchor_role"],
        layer_specs=config.layer_specs,
        layer_numbers=inputs["activation_layer_numbers"],
        pc_count=config.pc_count,
    )
    return {
        **inputs,
        "bundle": bundle,
    }


def unified_axis_run_root(config: UnifiedRoleAxisConfig, run_id: str) -> Path:
    root = (
        config.artifact_root
        / "runs"
        / slugify(config.experiment_name)
        / slugify(config.model_name)
        / slugify(config.behavior_name)
        / slugify(config.axis_name)
        / slugify(config.variant_name)
        / run_id
    )
    ensure_dir(root)
    return root
