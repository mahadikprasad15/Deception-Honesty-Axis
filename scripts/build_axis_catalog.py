#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from deception_honesty_axis.common import ensure_dir, find_repo_root, load_jsonl, make_run_id, read_json, slugify, utc_now_iso, write_json
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Catalog selected role axes from fixed-track search runs into canonical analysis artifacts."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional repo root. Defaults to auto-detection from this script path.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Optional fixed-track runs root. Defaults to <repo>/artifacts/runs/fixed-track-search.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional analysis output root. Defaults to <repo>/artifacts/analysis/axis-catalog.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Optional variant filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional catalog run id. Defaults to a fresh UTC token.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing catalog run if checkpoints are present.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild axis rows even if they were already checkpointed.",
    )
    return parser.parse_args()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed_axis_ids": [], "updated_at": None}
    return read_json(path)


def save_progress(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = utc_now_iso()
    write_json(path, payload)


def catalog_run_root(output_root: Path, run_id: str) -> Path:
    return output_root / run_id


def read_questions(search_run_dir: Path) -> list[dict[str, Any]]:
    selected_questions_path = search_run_dir / "selected" / "questions.jsonl"
    if selected_questions_path.exists():
        rows = load_jsonl(selected_questions_path)
        if rows:
            return rows
    final_questions_path = search_run_dir / "results" / "final_question_texts.json"
    if final_questions_path.exists():
        payload = read_json(final_questions_path)
        questions = payload.get("questions", [])
        return [dict(item) for item in questions if isinstance(item, dict)]
    return []


def read_role_entries(search_run_dir: Path, selection: dict[str, Any]) -> list[dict[str, Any]]:
    selected_manifest_path = search_run_dir / "selected" / "role_manifest.json"
    roles_payload = read_json(selected_manifest_path) if selected_manifest_path.exists() else None
    role_entries: list[dict[str, Any]] = []
    role_side_by_name = {
        "default": "anchor",
        **{str(name): "deceptive" for name in selection.get("deceptive_roles", [])},
        **{str(name): "honest" for name in selection.get("honest_roles", [])},
    }
    if roles_payload and isinstance(roles_payload.get("roles"), list):
        for order, entry in enumerate(roles_payload["roles"]):
            role_name = str(entry["name"])
            role_entries.append(
                {
                    "role_name": role_name,
                    "role_side": role_side_by_name.get(role_name, "unknown"),
                    "role_order": order,
                    "source": entry.get("source"),
                    "anchor_only": bool(entry.get("anchor_only", False)),
                }
            )
        return role_entries

    ordered_roles = [
        ("default", "anchor"),
        *[(str(name), "deceptive") for name in selection.get("deceptive_roles", [])],
        *[(str(name), "honest") for name in selection.get("honest_roles", [])],
    ]
    for order, (role_name, role_side) in enumerate(ordered_roles):
        role_entries.append(
            {
                "role_name": role_name,
                "role_side": role_side,
                "role_order": order,
                "source": None,
                "anchor_only": role_name == "default",
            }
        )
    return role_entries


def axis_id_for_run(relative_parts: tuple[str, ...]) -> str:
    if len(relative_parts) >= 5:
        model_slug, dataset_slug, variant, track_name, run_name = relative_parts[:5]
        return slugify(f"{model_slug}__{dataset_slug}__{variant}__{track_name}__{run_name}")
    return slugify("__".join(relative_parts))


def build_axis_payload(search_run_dir: Path, runs_root: Path) -> dict[str, Any]:
    manifest_path = search_run_dir / "meta" / "run_manifest.json"
    selection_path = search_run_dir / "results" / "final_selection.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing final selection: {selection_path}")

    manifest = read_json(manifest_path)
    selection = read_json(selection_path)
    relative_parts = search_run_dir.resolve().relative_to(runs_root.resolve()).parts
    axis_id = axis_id_for_run(relative_parts)
    model_slug = relative_parts[0] if len(relative_parts) > 0 else None
    dataset_slug = relative_parts[1] if len(relative_parts) > 1 else None
    variant = relative_parts[2] if len(relative_parts) > 2 else None
    track_name = relative_parts[3] if len(relative_parts) > 3 else None
    run_name = relative_parts[4] if len(relative_parts) > 4 else search_run_dir.name

    question_rows = read_questions(search_run_dir)
    question_rows_by_id = {str(row["question_id"]): row for row in question_rows if "question_id" in row}
    selected_question_ids = [str(item) for item in selection.get("question_ids", [])]
    ordered_questions = []
    for order, question_id in enumerate(selected_question_ids):
        row = question_rows_by_id.get(question_id, {"question_id": question_id, "question": None})
        ordered_questions.append(
            {
                "question_id": question_id,
                "question_text": row.get("question"),
                "question_order": order,
            }
        )

    role_entries = read_role_entries(search_run_dir, selection)
    payload = {
        "axis_id": axis_id,
        "source_run_dir": str(search_run_dir.resolve()),
        "source_run_relative": str(search_run_dir.resolve().relative_to(runs_root.resolve())),
        "model_slug": model_slug,
        "dataset_slug": dataset_slug,
        "variant": variant,
        "track_name": track_name,
        "run_name": run_name,
        "method": selection.get("method"),
        "layer_spec": selection.get("layer_spec"),
        "objective": selection.get("objective"),
        "threshold": selection.get("threshold"),
        "score": selection.get("score"),
        "completed_at": selection.get("completed_at"),
        "question_count": len(selected_question_ids),
        "honest_role_count": len(selection.get("honest_roles", [])),
        "deceptive_role_count": len(selection.get("deceptive_roles", [])),
        "honest_roles": [str(item) for item in selection.get("honest_roles", [])],
        "deceptive_roles": [str(item) for item in selection.get("deceptive_roles", [])],
        "question_ids": selected_question_ids,
        "questions": ordered_questions,
        "roles": role_entries,
        "paths": {
            "run_manifest": str(manifest_path.resolve()),
            "final_selection": str(selection_path.resolve()),
            "final_question_texts": str((search_run_dir / "results" / "final_question_texts.json").resolve()),
            "final_role_sets": str((search_run_dir / "results" / "final_role_sets.json").resolve()),
            "selected_questions": str((search_run_dir / "selected" / "questions.jsonl").resolve()),
            "selected_role_manifest": str((search_run_dir / "selected" / "role_manifest.json").resolve()),
            "selected_probe_config": str((search_run_dir / "selected" / "probe_config.json").resolve()),
        },
        "upstream": {
            "experiment_config_path": manifest.get("experiment_config_path"),
            "probe_config_path": manifest.get("probe_config_path"),
            "target_datasets": manifest.get("target_datasets"),
        },
    }
    return payload


def discover_search_runs(runs_root: Path, variants: set[str]) -> list[Path]:
    candidates = []
    for selection_path in sorted(runs_root.glob("**/results/final_selection.json")):
        search_run_dir = selection_path.parent.parent
        try:
            relative_parts = search_run_dir.resolve().relative_to(runs_root.resolve()).parts
        except ValueError:
            continue
        if len(relative_parts) < 5:
            continue
        variant = relative_parts[2]
        if variants and variant not in variants:
            continue
        candidates.append(search_run_dir)
    return candidates


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root(Path(__file__).resolve())
    runs_root = (args.runs_root.resolve() if args.runs_root else repo_root / "artifacts" / "runs" / "fixed-track-search")
    output_root = (args.output_root.resolve() if args.output_root else repo_root / "artifacts" / "analysis" / "axis-catalog")
    run_id = args.run_id or make_run_id()
    run_root = catalog_run_root(output_root, run_id)

    ensure_dir(run_root / "inputs")
    ensure_dir(run_root / "checkpoints")
    ensure_dir(run_root / "results" / "axes")
    ensure_dir(run_root / "logs")
    ensure_dir(run_root / "meta")

    progress_path = run_root / "checkpoints" / "progress.json"
    progress = load_progress(progress_path) if args.resume else {"completed_axis_ids": [], "updated_at": None}
    completed_axis_ids = set(progress.get("completed_axis_ids", []))

    write_analysis_manifest(
        run_root,
        {
            "analysis_kind": "axis-catalog",
            "repo_root": str(repo_root),
            "runs_root": str(runs_root),
            "output_root": str(output_root),
            "variant_filters": list(args.variant),
            "run_id": run_id,
            "resume": bool(args.resume),
            "force": bool(args.force),
            "created_at": utc_now_iso(),
        },
    )
    write_stage_status(run_root, "catalog", "running")

    run_paths = discover_search_runs(runs_root, set(args.variant))
    write_json(
        run_root / "inputs" / "source_runs.json",
        {"source_runs": [str(path.resolve()) for path in run_paths]},
    )

    axis_payloads: list[dict[str, Any]] = []
    for search_run_dir in run_paths:
        payload = build_axis_payload(search_run_dir, runs_root)
        axis_id = str(payload["axis_id"])
        if axis_id in completed_axis_ids and not args.force:
            axis_payloads.append(read_json(run_root / "results" / "axes" / f"{axis_id}.json"))
            continue
        write_json(run_root / "results" / "axes" / f"{axis_id}.json", payload)
        axis_payloads.append(payload)
        completed_axis_ids.add(axis_id)
        save_progress(progress_path, {"completed_axis_ids": sorted(completed_axis_ids)})

    catalog_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    for payload in axis_payloads:
        axis_id = str(payload["axis_id"])
        catalog_rows.append(
            {
                "axis_id": axis_id,
                "model_slug": payload.get("model_slug"),
                "dataset_slug": payload.get("dataset_slug"),
                "variant": payload.get("variant"),
                "track_name": payload.get("track_name"),
                "run_name": payload.get("run_name"),
                "method": payload.get("method"),
                "layer_spec": payload.get("layer_spec"),
                "objective": payload.get("objective"),
                "threshold": payload.get("threshold"),
                "score": payload.get("score"),
                "question_count": payload.get("question_count"),
                "honest_role_count": payload.get("honest_role_count"),
                "deceptive_role_count": payload.get("deceptive_role_count"),
                "source_run_dir": payload.get("source_run_dir"),
            }
        )
        for role_entry in payload.get("roles", []):
            role_rows.append(
                {
                    "axis_id": axis_id,
                    "role_name": role_entry.get("role_name"),
                    "role_side": role_entry.get("role_side"),
                    "role_order": role_entry.get("role_order"),
                    "source": role_entry.get("source"),
                    "anchor_only": role_entry.get("anchor_only"),
                }
            )
        for question_entry in payload.get("questions", []):
            question_rows.append(
                {
                    "axis_id": axis_id,
                    "question_id": question_entry.get("question_id"),
                    "question_order": question_entry.get("question_order"),
                    "question_text": question_entry.get("question_text"),
                }
            )

    if catalog_rows:
        write_csv(run_root / "results" / "axis_catalog.csv", list(catalog_rows[0].keys()), catalog_rows)
    if role_rows:
        write_csv(run_root / "results" / "axis_roles.csv", list(role_rows[0].keys()), role_rows)
    if question_rows:
        write_csv(run_root / "results" / "axis_questions.csv", list(question_rows[0].keys()), question_rows)
    write_json(
        run_root / "results" / "summary.json",
        {
            "axis_count": len(axis_payloads),
            "role_row_count": len(role_rows),
            "question_row_count": len(question_rows),
            "generated_at": utc_now_iso(),
        },
    )

    write_stage_status(
        run_root,
        "catalog",
        "completed",
        {
            "axis_count": len(axis_payloads),
            "role_row_count": len(role_rows),
            "question_row_count": len(question_rows),
        },
    )
    print(f"[axis-catalog] wrote analysis artifacts to {run_root}")


if __name__ == "__main__":
    main()
