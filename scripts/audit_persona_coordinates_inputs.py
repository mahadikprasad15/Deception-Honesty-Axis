#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from deception_honesty_axis.common import find_repo_root, load_jsonl, read_json, write_json


DEFAULT_UNIFIED_CONFIGS = [
    Path("configs/probes/unified_role_axis_quantity_sycophancy_shared.example.json"),
    Path("configs/probes/unified_role_axis_quantity_sycophancy_native.example.json"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit inputs for Persona Co-ordinates probing: unified-axis configs, role intersections, "
            "question banks, local artifacts, and optional HF artifact availability."
        )
    )
    parser.add_argument(
        "--unified-config",
        dest="unified_configs",
        type=Path,
        action="append",
        default=None,
        help="Unified role-axis config to audit. Repeatable. Defaults to quantity+sycophancy examples.",
    )
    parser.add_argument(
        "--quantity-role-manifest",
        type=Path,
        default=Path("data/manifests/role_sets/quantity_axis_v2.json"),
    )
    parser.add_argument(
        "--sycophancy-role-manifest",
        type=Path,
        default=Path("data/manifests/role_sets/sycophancy_pilot_v1.json"),
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Optional HF dataset repo id to inspect for stored unified-axis artifacts.",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON path for the audit report.")
    return parser.parse_args()


def resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def role_names_by_side(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    roles = [dict(role) for role in manifest.get("roles", [])]
    by_side: dict[str, list[str]] = {}
    for role in roles:
        if role.get("anchor_only"):
            side = "anchor"
        else:
            side = str(role.get("role_side") or role.get("side") or "unknown")
        by_side.setdefault(side, []).append(str(role["name"]))
    return {
        "path": str(manifest_path),
        "name": manifest.get("name"),
        "anchor_role": manifest.get("anchor_role"),
        "role_count": len(roles),
        "pca_roles": list(manifest.get("pca_roles", [])),
        "roles_by_side": by_side,
    }


def question_bank_summary(repo_root: Path, question_file: str | Path) -> dict[str, Any]:
    path = resolve(repo_root, Path(question_file))
    rows = load_jsonl(path)
    question_texts = [str(row.get("question", "")) for row in rows]
    harmful_markers = [
        "harm",
        "danger",
        "risk",
        "compliance",
        "incident",
        "mistake",
        "wrong",
        "jealous",
        "suspicion",
        "confront",
        "ignore",
    ]
    marker_counts = {
        marker: sum(1 for question in question_texts if marker in question.lower())
        for marker in harmful_markers
    }
    return {
        "path": str(path),
        "count": len(rows),
        "question_ids": [str(row.get("question_id", "")) for row in rows],
        "axis_ids": sorted({str(row.get("axis_id")) for row in rows if row.get("axis_id") not in (None, "")}),
        "scenario_families": sorted(
            {str(row.get("scenario_family")) for row in rows if row.get("scenario_family") not in (None, "")}
        ),
        "target_datasets": sorted(
            {
                str(dataset_name)
                for row in rows
                for dataset_name in (row.get("target_datasets") or [])
            }
        ),
        "marker_counts": {key: value for key, value in marker_counts.items() if value},
        "first_questions": question_texts[:3],
    }


def classify_question_bank(summary: dict[str, Any]) -> str:
    axis_ids = set(summary.get("axis_ids") or [])
    target_datasets = set(summary.get("target_datasets") or [])
    if "quantity" in axis_ids or any(name.startswith("Deception-") for name in target_datasets):
        return "deception/omission-oriented scenarios, not narrowly harmful-only"
    if summary.get("marker_counts"):
        return "general advice and self-justification prompts with some risk/feedback markers"
    return "general prompts"


def local_artifact_state(repo_root: Path, run_dir: str | Path) -> dict[str, Any]:
    path = resolve(repo_root, Path(run_dir))
    files = {
        "role_vectors": path / "results" / "role_vectors.pt",
        "role_vectors_meta": path / "results" / "role_vectors_meta.json",
        "axis_bundle": path / "results" / "axis_bundle.pt",
        "axis_bundle_json": path / "results" / "axis_bundle.json",
        "run_manifest": path / "meta" / "run_manifest.json",
    }
    return {
        "run_dir": str(path),
        "exists": path.exists(),
        "files": {name: file_path.exists() for name, file_path in files.items()},
    }


def hf_artifact_state(repo_id: str, candidates: list[str]) -> dict[str, Any]:
    from huggingface_hub import HfApi

    try:
        files = set(
            HfApi().list_repo_files(
                repo_id,
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN"),
            )
        )
    except Exception as exc:  # pragma: no cover - depends on external auth/network state.
        return {
            "repo_id": repo_id,
            "status": "unavailable",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    matched: dict[str, list[str]] = {}
    for candidate in candidates:
        normalized = str(candidate).lstrip("/")
        tail = normalized
        if normalized.startswith("artifacts/"):
            tail = normalized[len("artifacts/") :]
        matches = sorted(path for path in files if tail in path)
        matched[candidate] = matches[:25]
    unified_matches = sorted(
        path
        for path in files
        if "unified-role-axis-bundles" in path
        and (path.endswith("/results/role_vectors.pt") or path.endswith("/results/axis_bundle.pt"))
    )
    return {
        "repo_id": repo_id,
        "status": "available",
        "file_count": len(files),
        "candidate_matches": matched,
        "unified_role_axis_artifacts": unified_matches,
    }


def config_audit(repo_root: Path, config_path: Path) -> dict[str, Any]:
    path = resolve(repo_root, config_path)
    config = read_json(path)
    axis = dict(config.get("axis") or {})
    question_summaries = []
    for source in axis.get("question_sources") or []:
        if source.get("question_file"):
            summary = question_bank_summary(repo_root, source["question_file"])
            question_summaries.append(
                {
                    "origin": source.get("origin"),
                    **summary,
                    "content_classification": classify_question_bank(summary),
                }
            )

    source_entries: list[dict[str, Any]] = []
    if axis.get("shared_source"):
        source_entries.append({"source_kind": "shared_source", **dict(axis["shared_source"])})
    for source in axis.get("native_sources") or []:
        source_entries.append({"source_kind": "native_source", **dict(source)})

    return {
        "config_path": str(path),
        "experiment": config.get("experiment", {}),
        "question_mode": axis.get("question_mode"),
        "expected_pooling": axis.get("expected_pooling"),
        "pc_count": axis.get("pc_count"),
        "layer_specs": axis.get("layer_specs"),
        "question_sources": question_summaries,
        "sources": [
            {
                "source_kind": source.get("source_kind"),
                "name": source.get("name"),
                "behavior": source.get("behavior"),
                "vectors_run_dir": source.get("vectors_run_dir"),
                "safe_roles": source.get("safe_roles", []),
                "harmful_roles": source.get("harmful_roles", []),
                "anchor_role": source.get("anchor_role", "default"),
                "safe_role_count": len(source.get("safe_roles", [])),
                "harmful_role_count": len(source.get("harmful_roles", [])),
                "local_artifacts": local_artifact_state(repo_root, source["vectors_run_dir"])
                if source.get("vectors_run_dir")
                else None,
            }
            for source in source_entries
        ],
    }


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(Path.cwd())
    unified_configs = args.unified_configs or DEFAULT_UNIFIED_CONFIGS
    quantity_roles = role_names_by_side(resolve(repo_root, args.quantity_role_manifest))
    sycophancy_roles = role_names_by_side(resolve(repo_root, args.sycophancy_role_manifest))
    quantity_pca_roles = set(quantity_roles["pca_roles"])
    sycophancy_pca_roles = set(sycophancy_roles["pca_roles"])
    config_reports = [config_audit(repo_root, path) for path in unified_configs]

    hf_report = None
    if args.repo_id:
        candidates = [
            str(source.get("vectors_run_dir"))
            for report in config_reports
            for source in report.get("sources", [])
            if source.get("vectors_run_dir")
        ]
        hf_report = hf_artifact_state(str(args.repo_id), candidates)

    write_json(
        resolve(repo_root, args.output),
        {
            "repo_root": str(repo_root),
            "quantity_axis_v2": quantity_roles,
            "sycophancy_pilot_v1": sycophancy_roles,
            "role_intersections": {
                "quantity_pca_x_sycophancy_pca": sorted(quantity_pca_roles & sycophancy_pca_roles),
                "quantity_pca_minus_sycophancy_pca": sorted(quantity_pca_roles - sycophancy_pca_roles),
                "sycophancy_pca_minus_quantity_pca": sorted(sycophancy_pca_roles - quantity_pca_roles),
            },
            "unified_axis_configs": config_reports,
            "hf": hf_report,
        },
    )
    print(f"[persona-coordinates-audit] wrote audit to {resolve(repo_root, args.output)}")


if __name__ == "__main__":
    main()
