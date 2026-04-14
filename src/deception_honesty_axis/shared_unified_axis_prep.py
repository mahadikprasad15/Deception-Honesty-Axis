from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deception_honesty_axis.common import ensure_dir, make_run_id, write_json, write_jsonl
from deception_honesty_axis.config import analysis_run_root, load_config, slugify
from deception_honesty_axis.work_units import load_questions, load_role_manifest, select_first_n


def _role_side(row: dict[str, Any]) -> str | None:
    for key in ("role_side", "side"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def role_families_from_manifest(manifest: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    anchor_role = str(manifest["anchor_role"])
    safe_roles: list[str] = []
    harmful_roles: list[str] = []
    for role in manifest.get("roles", []):
        role_name = str(role["name"])
        if role_name == anchor_role or bool(role.get("anchor_only", False)):
            continue
        side = (_role_side(role) or "").strip().lower()
        if side in {"honest", "non_sycophantic", "safe"}:
            safe_roles.append(role_name)
        elif side in {"deceptive", "sycophantic", "harmful"}:
            harmful_roles.append(role_name)
        else:
            raise ValueError(f"Unsupported role side for {role_name!r}: {side!r}")
    return anchor_role, safe_roles, harmful_roles


def merge_role_manifests(
    *,
    left_manifest: dict[str, Any],
    right_manifest: dict[str, Any],
    merged_name: str,
) -> tuple[dict[str, Any], list[str], list[str]]:
    left_anchor, left_safe, left_harmful = role_families_from_manifest(left_manifest)
    right_anchor, right_safe, right_harmful = role_families_from_manifest(right_manifest)
    if left_anchor != right_anchor:
        raise ValueError(f"Anchor role mismatch: {left_anchor!r} vs {right_anchor!r}")

    merged_roles: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    for source_manifest, behavior in ((left_manifest, "deception"), (right_manifest, "sycophancy")):
        for role in source_manifest.get("roles", []):
            role_name = str(role["name"])
            if role_name in seen_roles:
                if role_name == left_anchor:
                    continue
                raise ValueError(f"Duplicate role name across manifests: {role_name!r}")
            merged_role = dict(role)
            merged_role.setdefault("behavior", behavior)
            merged_roles.append(merged_role)
            seen_roles.add(role_name)

    merged_manifest = {
        "name": merged_name,
        "anchor_role": left_anchor,
        "pca_roles": left_harmful + left_safe + right_harmful + right_safe,
        "roles": merged_roles,
    }
    return merged_manifest, left_safe + right_safe, left_harmful + right_harmful


def build_shared_question_rows(
    *,
    left_question_file: Path,
    right_question_file: Path,
    questions_per_source: int,
) -> list[dict[str, Any]]:
    if questions_per_source < 1:
        raise ValueError(f"questions_per_source must be positive, got {questions_per_source}")

    left_rows = select_first_n(load_questions(left_question_file), questions_per_source)
    right_rows = select_first_n(load_questions(right_question_file), questions_per_source)
    if len(left_rows) < questions_per_source or len(right_rows) < questions_per_source:
        raise ValueError(
            "Not enough questions available to build a balanced shared set; "
            f"requested={questions_per_source}, got left={len(left_rows)}, right={len(right_rows)}"
        )

    merged: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()
    for question_origin, rows in (("deception", left_rows), ("sycophancy", right_rows)):
        for row in rows:
            if row.question_id in seen_question_ids:
                raise ValueError(f"Duplicate question_id across shared sources: {row.question_id!r}")
            seen_question_ids.add(row.question_id)
            merged.append(
                {
                    **row.metadata,
                    "question_id": row.question_id,
                    "question": row.question,
                    "question_origin": question_origin,
                }
            )
    return merged


@dataclass(frozen=True)
class PreparedSharedUnifiedAxis:
    run_id: str
    prep_root: Path
    question_file: Path
    role_manifest_file: Path
    experiment_config_file: Path
    bundle_config_file: Path
    role_vectors_run_dir: Path
    safe_roles: list[str]
    harmful_roles: list[str]


def prepare_shared_unified_axis_experiment(
    *,
    quantity_experiment_config_path: Path,
    sycophancy_experiment_config_path: Path,
    artifact_root: Path,
    questions_per_source: int | None,
    pc_count: int,
    layer_number: int,
    run_id: str | None,
) -> PreparedSharedUnifiedAxis:
    quantity_config = load_config(quantity_experiment_config_path.resolve())
    sycophancy_config = load_config(sycophancy_experiment_config_path.resolve())
    repo_root = quantity_config.repo_root.resolve()
    resolved_artifact_root = artifact_root.resolve()
    if quantity_config.model_id != sycophancy_config.model_id:
        raise ValueError(
            f"Model mismatch between source experiments: {quantity_config.model_id!r} vs {sycophancy_config.model_id!r}"
        )

    quantity_manifest = load_role_manifest(quantity_config.role_manifest_path)
    sycophancy_manifest = load_role_manifest(sycophancy_config.role_manifest_path)
    merged_manifest, safe_roles, harmful_roles = merge_role_manifests(
        left_manifest=quantity_manifest,
        right_manifest=sycophancy_manifest,
        merged_name="quantity-sycophancy-shared",
    )

    resolved_questions_per_source = questions_per_source or min(
        int(quantity_config.subset["question_count"]),
        int(sycophancy_config.subset["question_count"]),
    )
    merged_questions = build_shared_question_rows(
        left_question_file=quantity_config.question_file_path,
        right_question_file=sycophancy_config.question_file_path,
        questions_per_source=resolved_questions_per_source,
    )

    resolved_run_id = run_id or make_run_id()
    prep_root = ensure_dir(
        resolved_artifact_root
        / "prepared"
        / "unified-role-axis"
        / "quantity-sycophancy-shared"
        / resolved_run_id
    ).resolve()
    inputs_root = ensure_dir(prep_root / "inputs")
    question_file = prep_root / "quantity_sycophancy_shared_questions.jsonl"
    role_manifest_file = prep_root / "quantity_sycophancy_shared_roles.json"
    experiment_config_file = prep_root / "quantity_sycophancy_shared_experiment.json"
    bundle_config_file = prep_root / "quantity_sycophancy_shared_bundle.json"

    write_jsonl(question_file, merged_questions)
    write_json(role_manifest_file, merged_manifest)

    instruction_count = min(
        int(quantity_config.subset["instruction_count"]),
        int(sycophancy_config.subset["instruction_count"]),
    )
    experiment_config = {
        "experiment_name": "quantity-sycophancy-shared",
        "model": dict(quantity_config.raw["model"]),
        "data": {
            "question_file": str(question_file.relative_to(quantity_config.repo_root)),
            "role_manifest_file": str(role_manifest_file.relative_to(repo_root)),
        },
        "subset": {
            "instruction_selection": "first_n",
            "instruction_count": instruction_count,
            "question_selection": "first_n",
            "question_count": len(merged_questions),
        },
        "artifacts": {
            "root": str(resolved_artifact_root.relative_to(repo_root)),
            "dataset_name": "assistant-axis",
            "analysis_variant": "center-only",
            "corpus_name": "quantity-sycophancy-shared",
        },
        "saving": dict(quantity_config.raw["saving"]),
        "analysis": {
            "filter_name": "all-responses",
            "pooling": "mean_response",
            "anchor_role": "default",
            "activation_layers": [int(layer_number)],
        },
        "hf": dict(quantity_config.raw["hf"]),
    }
    write_json(experiment_config_file, experiment_config)

    prepared_experiment_config = load_config(experiment_config_file)
    role_vectors_run_dir = analysis_run_root(
        prepared_experiment_config,
        "role-vectors",
        prepared_experiment_config.analysis["filter_name"],
        resolved_run_id,
    )
    bundle_config = {
        "artifacts": {
            "root": str(resolved_artifact_root.relative_to(repo_root)),
        },
        "experiment": {
            "name": "unified-role-axis-bundles",
            "behavior": "harmful-behavior",
            "model_name": quantity_config.model_id,
            "axis_name": "quantity-sycophancy-unified",
            "variant": "shared-questions-balanced",
        },
        "axis": {
            "question_mode": "shared_questions_balanced",
            "expected_pooling": "mean_response",
            "pc_count": int(pc_count),
            "layer_specs": [str(layer_number)],
            "question_sources": [
                {
                    "origin": "deception",
                    "question_file": str(quantity_config.question_file_path.relative_to(repo_root)),
                    "selected_count": resolved_questions_per_source,
                },
                {
                    "origin": "sycophancy",
                    "question_file": str(sycophancy_config.question_file_path.relative_to(repo_root)),
                    "selected_count": resolved_questions_per_source,
                },
            ],
            "shared_source": {
                "name": "quantity_sycophancy_shared",
                "behavior": "harmful-behavior",
                "vectors_run_dir": str(role_vectors_run_dir.relative_to(repo_root)),
                "safe_roles": safe_roles,
                "harmful_roles": harmful_roles,
                "anchor_role": "default",
            },
        },
    }
    write_json(bundle_config_file, bundle_config)
    write_json(
        inputs_root / "results.json",
        {
            "run_id": resolved_run_id,
            "questions_per_source": resolved_questions_per_source,
            "question_count": len(merged_questions),
            "question_file": str(question_file),
            "role_manifest_file": str(role_manifest_file),
            "experiment_config_file": str(experiment_config_file),
            "bundle_config_file": str(bundle_config_file),
            "role_vectors_run_dir": str(role_vectors_run_dir),
            "safe_roles": safe_roles,
            "harmful_roles": harmful_roles,
        },
    )

    return PreparedSharedUnifiedAxis(
        run_id=resolved_run_id,
        prep_root=prep_root,
        question_file=question_file,
        role_manifest_file=role_manifest_file,
        experiment_config_file=experiment_config_file,
        bundle_config_file=bundle_config_file,
        role_vectors_run_dir=role_vectors_run_dir,
        safe_roles=safe_roles,
        harmful_roles=harmful_roles,
    )
