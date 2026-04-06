#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from deception_honesty_axis.common import ensure_dir, find_repo_root, load_jsonl, make_run_id, read_json, slugify, utc_now_iso, write_json
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate fixed-track greedy-search artifacts into master evolution and family-analysis tables."
    )
    parser.add_argument("--repo-root", type=Path, default=None, help="Optional repo root; defaults to auto-detection.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Optional fixed-track runs root. Defaults to <repo>/artifacts/runs/fixed-track-search.",
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=None,
        help="Optional dataset-geometry root. Defaults to <repo>/artifacts/analysis/dataset-geometry.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional analysis output root. Defaults to <repo>/artifacts/analysis/greedy-search-evolution.",
    )
    parser.add_argument("--variant", action="append", default=[], help="Optional variant filter.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed analysis run id.")
    parser.add_argument("--resume", action="store_true", help="Resume an existing analysis run if checkpoints are present.")
    parser.add_argument("--force", action="store_true", help="Recompute per-run artifacts even if already saved.")
    parser.add_argument(
        "--cluster-count",
        type=int,
        default=3,
        help="Requested number of dataset/axis clusters when enough rows are available.",
    )
    return parser.parse_args()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed_axis_ids": [], "updated_at": None}
    return read_json(path)


def save_progress(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = utc_now_iso()
    write_json(path, payload)


def discover_search_runs(runs_root: Path, variants: set[str]) -> list[Path]:
    run_dirs: list[Path] = []
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
        run_dirs.append(search_run_dir)
    return run_dirs


def discover_geometry_runs(geometry_root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    if not geometry_root.exists():
        return mapping
    for reference_path in geometry_root.glob("**/inputs/source_run_reference.json"):
        try:
            payload = read_json(reference_path)
        except Exception:
            continue
        source_run_dir = payload.get("source_run_dir")
        if not source_run_dir:
            continue
        run_root = reference_path.parent.parent
        existing = mapping.get(str(source_run_dir))
        if existing is None or run_root.name > existing.name:
            mapping[str(source_run_dir)] = run_root
    return mapping


def safe_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    try:
        return int(value)
    except Exception:
        return None


def split_csv_field(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [item for item in text.split(",") if item]


def axis_id_for_run(relative_parts: tuple[str, ...]) -> str:
    if len(relative_parts) >= 5:
        model_slug, dataset_slug, variant, track_name, run_name = relative_parts[:5]
        return slugify(f"{model_slug}__{dataset_slug}__{variant}__{track_name}__{run_name}")
    return slugify("__".join(relative_parts))


def compact_axis_label(variant: str, method: str, objective: str | None) -> str:
    probe = "contrast" if "contrast" in method else "pc1" if "pc1" in method else method
    objective_label = "thr" if (objective and "threshold" in objective) else "mean"
    return f"{variant} | {probe} | {objective_label}"


def summarize_questions(search_run_dir: Path, final_selection: dict[str, Any]) -> list[dict[str, Any]]:
    selected_questions_path = search_run_dir / "selected" / "questions.jsonl"
    if selected_questions_path.exists():
        rows = load_jsonl(selected_questions_path)
        if rows:
            return [
                {
                    "question_id": str(row.get("question_id")),
                    "question_text": row.get("question"),
                    "question_order": order,
                }
                for order, row in enumerate(rows)
                if row.get("question_id") is not None
            ]
    final_questions_path = search_run_dir / "results" / "final_question_texts.json"
    final_questions = read_json(final_questions_path).get("questions", []) if final_questions_path.exists() else []
    ordered = []
    for order, row in enumerate(final_questions):
        ordered.append(
            {
                "question_id": str(row.get("question_id")),
                "question_text": row.get("question"),
                "question_order": order,
            }
        )
    if ordered:
        return ordered
    return [
        {"question_id": str(question_id), "question_text": None, "question_order": order}
        for order, question_id in enumerate(final_selection.get("question_ids", []))
    ]


def summarize_roles(search_run_dir: Path, final_selection: dict[str, Any]) -> list[dict[str, Any]]:
    selected_manifest_path = search_run_dir / "selected" / "role_manifest.json"
    role_side_by_name = {
        "default": "anchor",
        **{str(role): "deceptive" for role in final_selection.get("deceptive_roles", [])},
        **{str(role): "honest" for role in final_selection.get("honest_roles", [])},
    }
    if selected_manifest_path.exists():
        payload = read_json(selected_manifest_path)
        rows = []
        for order, entry in enumerate(payload.get("roles", [])):
            role_name = str(entry["name"])
            rows.append(
                {
                    "role_name": role_name,
                    "role_side": role_side_by_name.get(role_name, "unknown"),
                    "role_order": order,
                    "source": entry.get("source"),
                    "anchor_only": bool(entry.get("anchor_only", False)),
                }
            )
        if rows:
            return rows
    rows = [{"role_name": "default", "role_side": "anchor", "role_order": 0, "source": None, "anchor_only": True}]
    role_order = 1
    for role_name in final_selection.get("deceptive_roles", []):
        rows.append(
            {
                "role_name": str(role_name),
                "role_side": "deceptive",
                "role_order": role_order,
                "source": None,
                "anchor_only": False,
            }
        )
        role_order += 1
    for role_name in final_selection.get("honest_roles", []):
        rows.append(
            {
                "role_name": str(role_name),
                "role_side": "honest",
                "role_order": role_order,
                "source": None,
                "anchor_only": False,
            }
        )
        role_order += 1
    return rows


def attach_step_deltas(step_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    previous = None
    for row in sorted(step_rows, key=lambda item: safe_int(item.get("step")) or 0):
        new_row = dict(row)
        if previous is None:
            new_row["score_gain_mean_auroc"] = None
            new_row["score_gain_mean_delta"] = None
            new_row["score_gain_sum_delta"] = None
            new_row["score_gain_improved_count"] = None
            new_row["score_gain_count_gt_threshold"] = None
            new_row["score_gain_clipped_threshold_sum"] = None
        else:
            new_row["score_gain_mean_auroc"] = (
                safe_float(new_row.get("mean_auroc")) - safe_float(previous.get("mean_auroc"))
                if safe_float(new_row.get("mean_auroc")) is not None and safe_float(previous.get("mean_auroc")) is not None
                else None
            )
            new_row["score_gain_mean_delta"] = (
                safe_float(new_row.get("mean_delta")) - safe_float(previous.get("mean_delta"))
                if safe_float(new_row.get("mean_delta")) is not None and safe_float(previous.get("mean_delta")) is not None
                else None
            )
            new_row["score_gain_sum_delta"] = (
                safe_float(new_row.get("sum_delta")) - safe_float(previous.get("sum_delta"))
                if safe_float(new_row.get("sum_delta")) is not None and safe_float(previous.get("sum_delta")) is not None
                else None
            )
            new_row["score_gain_improved_count"] = (
                safe_int(new_row.get("improved_count")) - safe_int(previous.get("improved_count"))
                if safe_int(new_row.get("improved_count")) is not None and safe_int(previous.get("improved_count")) is not None
                else None
            )
            new_row["score_gain_count_gt_threshold"] = (
                safe_int(new_row.get("count_gt_threshold")) - safe_int(previous.get("count_gt_threshold"))
                if safe_int(new_row.get("count_gt_threshold")) is not None and safe_int(previous.get("count_gt_threshold")) is not None
                else None
            )
            new_row["score_gain_clipped_threshold_sum"] = (
                safe_float(new_row.get("clipped_threshold_sum")) - safe_float(previous.get("clipped_threshold_sum"))
                if safe_float(new_row.get("clipped_threshold_sum")) is not None and safe_float(previous.get("clipped_threshold_sum")) is not None
                else None
            )
        output.append(new_row)
        previous = new_row
    return output


def normalize_candidate_rows(candidate_rows: list[dict[str, Any]], step_lookup: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in candidate_rows:
        step = safe_int(row.get("step")) or 0
        baseline_step = step_lookup.get(step - 1)
        new_row = dict(row)
        new_row["step_baseline_mean_auroc"] = baseline_step.get("mean_auroc") if baseline_step else None
        new_row["step_baseline_mean_delta"] = baseline_step.get("mean_delta") if baseline_step else None
        new_row["step_baseline_sum_delta"] = baseline_step.get("sum_delta") if baseline_step else None
        new_row["candidate_gain_mean_auroc"] = (
            safe_float(row.get("mean_auroc")) - safe_float(baseline_step.get("mean_auroc"))
            if baseline_step and safe_float(row.get("mean_auroc")) is not None and safe_float(baseline_step.get("mean_auroc")) is not None
            else None
        )
        new_row["candidate_gain_mean_delta"] = (
            safe_float(row.get("mean_delta")) - safe_float(baseline_step.get("mean_delta"))
            if baseline_step and safe_float(row.get("mean_delta")) is not None and safe_float(baseline_step.get("mean_delta")) is not None
            else None
        )
        new_row["candidate_gain_sum_delta"] = (
            safe_float(row.get("sum_delta")) - safe_float(baseline_step.get("sum_delta"))
            if baseline_step and safe_float(row.get("sum_delta")) is not None and safe_float(baseline_step.get("sum_delta")) is not None
            else None
        )
        normalized.append(new_row)
    return normalized


def read_geometry_payload(geometry_run_dir: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if geometry_run_dir is None:
        return [], []
    geometry_rows = read_csv_rows(geometry_run_dir / "results" / "per_dataset_geometry.csv")
    role_rows = read_csv_rows(geometry_run_dir / "results" / "per_dataset_role_attribution.csv")
    return geometry_rows, role_rows


def build_similarity_table(matrix_df, entity_key: str) -> list[dict[str, Any]]:
    entities = list(matrix_df.index)
    values = matrix_df.to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(entities):
        left_vec = values[i]
        for j, right in enumerate(entities):
            if j <= i:
                continue
            right_vec = values[j]
            if np.allclose(left_vec, left_vec[0]) or np.allclose(right_vec, right_vec[0]):
                correlation = None
            else:
                correlation = float(np.corrcoef(left_vec, right_vec)[0, 1])
            denom = np.linalg.norm(left_vec) * np.linalg.norm(right_vec)
            cosine = None if float(denom) == 0.0 else float(np.dot(left_vec, right_vec) / denom)
            rows.append(
                {
                    f"{entity_key}_a": left,
                    f"{entity_key}_b": right,
                    "profile_correlation": correlation,
                    "profile_cosine": cosine,
                }
            )
    return rows


def maybe_cluster(matrix_df, entity_key: str, cluster_count: int) -> list[dict[str, Any]]:
    if len(matrix_df.index) < 2:
        return []
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return []

    n_clusters = min(cluster_count, len(matrix_df.index))
    if n_clusters < 2:
        return []
    scaler = StandardScaler()
    features = scaler.fit_transform(matrix_df.to_numpy(dtype=np.float64))
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(features)
    return [
        {entity_key: entity, "cluster": int(label)}
        for entity, label in zip(matrix_df.index.tolist(), labels.tolist(), strict=False)
    ]


def jaccard(left: set[str], right: set[str]) -> float | None:
    if not left and not right:
        return None
    union = left | right
    if not union:
        return None
    return float(len(left & right) / len(union))


def normalize_run(
    search_run_dir: Path,
    runs_root: Path,
    geometry_mapping: dict[str, Path],
) -> dict[str, Any]:
    relative_parts = search_run_dir.resolve().relative_to(runs_root.resolve()).parts
    model_slug, dataset_slug, variant, track_name, run_name = relative_parts[:5]
    axis_id = axis_id_for_run(relative_parts)

    run_manifest = read_json(search_run_dir / "meta" / "run_manifest.json")
    final_selection = read_json(search_run_dir / "results" / "final_selection.json")
    search_summary = read_json(search_run_dir / "results" / "search_summary.json")
    final_role_sets = read_json(search_run_dir / "results" / "final_role_sets.json")
    step_rows = attach_step_deltas(read_csv_rows(search_run_dir / "results" / "step_history.csv"))
    step_lookup = {safe_int(row.get("step")) or 0: row for row in step_rows}

    candidate_rows: list[dict[str, Any]] = []
    candidate_dir = search_run_dir / "results" / "candidate_moves"
    if candidate_dir.exists():
        for candidate_path in sorted(candidate_dir.glob("step_*.csv")):
            candidate_rows.extend(read_csv_rows(candidate_path))
    candidate_rows = normalize_candidate_rows(candidate_rows, step_lookup)

    final_dataset_rows = read_csv_rows(search_run_dir / "results" / "final_per_dataset.csv")
    baseline_dataset_rows = read_csv_rows(search_run_dir / "results" / "baseline_per_dataset.csv")
    geometry_run_dir = geometry_mapping.get(str(search_run_dir.resolve()))
    geometry_rows, role_attribution_rows = read_geometry_payload(geometry_run_dir)

    method = str(final_selection.get("method"))
    objective = str(final_selection.get("objective") or "")
    objective_label = "thr" if "threshold" in objective else "mean"
    axis_label = compact_axis_label(variant, method, objective)
    role_rows = summarize_roles(search_run_dir, final_selection)
    question_rows = summarize_questions(search_run_dir, final_selection)

    run_row = {
        "axis_id": axis_id,
        "axis_label": axis_label,
        "model_slug": model_slug,
        "dataset_slug": dataset_slug,
        "variant": variant,
        "track_name": track_name,
        "run_name": run_name,
        "method": method,
        "probe_family": "contrast" if "contrast" in method else "pc1" if "pc1" in method else method,
        "layer_spec": final_selection.get("layer_spec"),
        "objective": final_selection.get("objective") or "mean",
        "objective_label": objective_label,
        "threshold": final_selection.get("threshold"),
        "steps_completed": search_summary.get("steps_completed"),
        "baseline_mean_auroc": search_summary.get("baseline_score", {}).get("mean_auroc"),
        "final_mean_auroc": search_summary.get("final_score", {}).get("mean_auroc"),
        "final_mean_delta": search_summary.get("final_score", {}).get("mean_delta"),
        "final_sum_delta": search_summary.get("final_score", {}).get("sum_delta"),
        "final_improved_count": search_summary.get("final_score", {}).get("improved_count"),
        "final_nonnegative_count": search_summary.get("final_score", {}).get("nonnegative_count"),
        "final_count_gt_threshold": search_summary.get("final_score", {}).get("count_gt_threshold"),
        "question_count": final_selection.get("question_count"),
        "honest_role_count": len(final_role_sets.get("honest_roles", [])),
        "deceptive_role_count": len(final_role_sets.get("deceptive_roles", [])),
        "source_run_dir": str(search_run_dir.resolve()),
        "experiment_config_path": run_manifest.get("experiment_config_path"),
        "probe_config_path": run_manifest.get("probe_config_path"),
        "geometry_run_dir": str(geometry_run_dir) if geometry_run_dir else None,
    }

    normalized_steps = []
    for row in step_rows:
        normalized = dict(row)
        normalized.update(
            {
                "axis_id": axis_id,
                "axis_label": axis_label,
                "variant": variant,
                "track_name": track_name,
                "run_name": run_name,
                "method": method,
                "probe_family": run_row["probe_family"],
                "objective": run_row["objective"],
                "objective_label": objective_label,
                "threshold": final_selection.get("threshold"),
                "item_type": "question" if "question" in str(row.get("move_type")) else "role",
                "role_side_affected": (
                    "honest" if "honest" in str(row.get("move_type"))
                    else "deceptive" if "deceptive" in str(row.get("move_type"))
                    else None
                ),
            }
        )
        normalized_steps.append(normalized)

    normalized_candidates = []
    for row in candidate_rows:
        normalized = dict(row)
        normalized.update(
            {
                "axis_id": axis_id,
                "axis_label": axis_label,
                "variant": variant,
                "track_name": track_name,
                "run_name": run_name,
                "method": method,
                "probe_family": run_row["probe_family"],
                "objective": run_row["objective"],
                "objective_label": objective_label,
                "threshold": final_selection.get("threshold"),
                "item_type": "question" if "question" in str(row.get("move_type")) else "role",
                "role_side_affected": (
                    "honest" if "honest" in str(row.get("move_type"))
                    else "deceptive" if "deceptive" in str(row.get("move_type"))
                    else None
                ),
            }
        )
        normalized_candidates.append(normalized)

    normalized_final_dataset_rows = []
    for row in final_dataset_rows:
        normalized = dict(row)
        normalized.update(
            {
                "axis_id": axis_id,
                "axis_label": axis_label,
                "variant": variant,
                "track_name": track_name,
                "run_name": run_name,
                "method": method,
                "probe_family": run_row["probe_family"],
                "objective": run_row["objective"],
                "objective_label": objective_label,
                "threshold": final_selection.get("threshold"),
            }
        )
        normalized_final_dataset_rows.append(normalized)

    normalized_baseline_rows = []
    for row in baseline_dataset_rows:
        normalized = dict(row)
        normalized.update(
            {
                "axis_id": axis_id,
                "axis_label": axis_label,
                "variant": variant,
                "track_name": track_name,
                "run_name": run_name,
                "method": method,
                "probe_family": run_row["probe_family"],
                "objective": run_row["objective"],
                "objective_label": objective_label,
                "threshold": final_selection.get("threshold"),
            }
        )
        normalized_baseline_rows.append(normalized)

    normalized_role_rows = []
    for role_row in role_rows:
        normalized = dict(role_row)
        normalized.update(
            {
                "axis_id": axis_id,
                "axis_label": axis_label,
                "variant": variant,
                "method": method,
                "objective": run_row["objective"],
                "objective_label": objective_label,
                "track_name": track_name,
                "run_name": run_name,
            }
        )
        normalized_role_rows.append(normalized)

    normalized_question_rows = []
    for question_row in question_rows:
        normalized = dict(question_row)
        normalized.update(
            {
                "axis_id": axis_id,
                "axis_label": axis_label,
                "variant": variant,
                "method": method,
                "objective": run_row["objective"],
                "objective_label": objective_label,
                "track_name": track_name,
                "run_name": run_name,
            }
        )
        normalized_question_rows.append(normalized)

    normalized_geometry_rows = []
    for row in geometry_rows:
        normalized = dict(row)
        normalized.update(
            {
                "axis_id": axis_id,
                "axis_label": axis_label,
                "variant": variant,
                "method": method,
                "objective": run_row["objective"],
                "objective_label": objective_label,
                "track_name": track_name,
                "run_name": run_name,
                "geometry_run_dir": str(geometry_run_dir) if geometry_run_dir else None,
            }
        )
        normalized_geometry_rows.append(normalized)

    normalized_role_mix_rows = []
    for row in role_attribution_rows:
        normalized = dict(row)
        normalized.update(
            {
                "axis_id": axis_id,
                "axis_label": axis_label,
                "variant": variant,
                "method": method,
                "objective": run_row["objective"],
                "objective_label": objective_label,
                "track_name": track_name,
                "run_name": run_name,
                "geometry_run_dir": str(geometry_run_dir) if geometry_run_dir else None,
            }
        )
        normalized_role_mix_rows.append(normalized)

    return {
        "run": run_row,
        "steps": normalized_steps,
        "candidates": normalized_candidates,
        "final_dataset_rows": normalized_final_dataset_rows,
        "baseline_dataset_rows": normalized_baseline_rows,
        "axis_roles": normalized_role_rows,
        "axis_questions": normalized_question_rows,
        "geometry_rows": normalized_geometry_rows,
        "role_mix_rows": normalized_role_mix_rows,
    }


def build_drop_summaries(step_rows: list[dict[str, Any]], final_dataset_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted_rows = [row for row in step_rows if str(row.get("move_type")) != "start"]
    final_dataset_by_axis: dict[str, list[dict[str, Any]]] = {}
    for row in final_dataset_rows:
        final_dataset_by_axis.setdefault(str(row["axis_id"]), []).append(row)

    summary_by_item: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    dataset_assoc: list[dict[str, Any]] = []
    for row in accepted_rows:
        move_type = str(row.get("move_type"))
        item = str(row.get("item"))
        role_side = row.get("role_side_affected")
        key = (move_type, item, str(role_side) if role_side is not None else None)
        payload = summary_by_item.setdefault(
            key,
            {
                "move_type": move_type,
                "item": item,
                "item_type": row.get("item_type"),
                "role_side_affected": role_side,
                "times_selected": 0,
                "mean_step": [],
                "mean_score_gain_mean_auroc": [],
                "mean_score_gain_mean_delta": [],
                "mean_score_gain_sum_delta": [],
                "variants": set(),
                "axes": set(),
            },
        )
        payload["times_selected"] += 1
        if safe_float(row.get("step")) is not None:
            payload["mean_step"].append(float(row["step"]))
        for source_field, target_field in (
            ("score_gain_mean_auroc", "mean_score_gain_mean_auroc"),
            ("score_gain_mean_delta", "mean_score_gain_mean_delta"),
            ("score_gain_sum_delta", "mean_score_gain_sum_delta"),
        ):
            value = safe_float(row.get(source_field))
            if value is not None:
                payload[target_field].append(value)
        payload["variants"].add(str(row.get("variant")))
        payload["axes"].add(str(row.get("axis_id")))

        for dataset_row in final_dataset_by_axis.get(str(row["axis_id"]), []):
            dataset_assoc.append(
                {
                    "move_type": move_type,
                    "item": item,
                    "item_type": row.get("item_type"),
                    "role_side_affected": role_side,
                    "dataset": dataset_row.get("dataset"),
                    "dataset_label": dataset_row.get("dataset_label"),
                    "axis_id": row.get("axis_id"),
                    "variant": row.get("variant"),
                    "final_delta": safe_float(dataset_row.get("delta")),
                    "final_auroc": safe_float(dataset_row.get("auroc")),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for payload in summary_by_item.values():
        summary_rows.append(
            {
                "move_type": payload["move_type"],
                "item": payload["item"],
                "item_type": payload["item_type"],
                "role_side_affected": payload["role_side_affected"],
                "times_selected": payload["times_selected"],
                "mean_step": float(np.mean(payload["mean_step"])) if payload["mean_step"] else None,
                "mean_score_gain_mean_auroc": float(np.mean(payload["mean_score_gain_mean_auroc"])) if payload["mean_score_gain_mean_auroc"] else None,
                "mean_score_gain_mean_delta": float(np.mean(payload["mean_score_gain_mean_delta"])) if payload["mean_score_gain_mean_delta"] else None,
                "mean_score_gain_sum_delta": float(np.mean(payload["mean_score_gain_sum_delta"])) if payload["mean_score_gain_sum_delta"] else None,
                "variant_count": len(payload["variants"]),
                "axis_count": len(payload["axes"]),
            }
        )
    summary_rows.sort(key=lambda row: (-int(row["times_selected"]), -(safe_float(row["mean_score_gain_mean_auroc"]) or 0.0), str(row["item"])))

    dataset_summary: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in dataset_assoc:
        key = (str(row["item"]), str(row["move_type"]), str(row["dataset"]))
        payload = dataset_summary.setdefault(
            key,
            {
                "item": row["item"],
                "move_type": row["move_type"],
                "dataset": row["dataset"],
                "dataset_label": row["dataset_label"],
                "item_type": row["item_type"],
                "role_side_affected": row["role_side_affected"],
                "final_delta": [],
                "final_auroc": [],
                "axis_count": 0,
            },
        )
        if row["final_delta"] is not None:
            payload["final_delta"].append(float(row["final_delta"]))
        if row["final_auroc"] is not None:
            payload["final_auroc"].append(float(row["final_auroc"]))
        payload["axis_count"] += 1
    dataset_rows: list[dict[str, Any]] = []
    for payload in dataset_summary.values():
        dataset_rows.append(
            {
                "item": payload["item"],
                "move_type": payload["move_type"],
                "item_type": payload["item_type"],
                "role_side_affected": payload["role_side_affected"],
                "dataset": payload["dataset"],
                "dataset_label": payload["dataset_label"],
                "mean_final_delta_when_selected": float(np.mean(payload["final_delta"])) if payload["final_delta"] else None,
                "mean_final_auroc_when_selected": float(np.mean(payload["final_auroc"])) if payload["final_auroc"] else None,
                "axis_count": payload["axis_count"],
            }
        )
    return summary_rows, dataset_rows


def build_dataset_feature_tables(final_dataset_rows: list[dict[str, Any]], geometry_rows: list[dict[str, Any]], role_mix_rows: list[dict[str, Any]], cluster_count: int) -> dict[str, list[dict[str, Any]]]:
    import pandas as pd

    final_df = pd.DataFrame(final_dataset_rows)
    if final_df.empty:
        return {"features": [], "similarity": [], "clusters": [], "matrix": []}
    final_df["delta"] = pd.to_numeric(final_df["delta"], errors="coerce")
    final_df["auroc"] = pd.to_numeric(final_df["auroc"], errors="coerce")
    feature_rows = (
        final_df.groupby(["dataset", "dataset_label"], as_index=False)
        .agg(
            mean_final_delta=("delta", "mean"),
            std_final_delta=("delta", "std"),
            best_final_delta=("delta", "max"),
            mean_final_auroc=("auroc", "mean"),
            run_count=("axis_id", "count"),
        )
        .sort_values("dataset_label")
        .to_dict(orient="records")
    )

    if geometry_rows:
        geometry_df = pd.DataFrame(geometry_rows)
        if "separation_ratio" in geometry_df.columns:
            geometry_df["separation_ratio"] = pd.to_numeric(geometry_df["separation_ratio"], errors="coerce")
            geometry_df["auroc"] = pd.to_numeric(geometry_df.get("auroc"), errors="coerce")
            geometry_features = (
                geometry_df.groupby("dataset_label", as_index=False)
                .agg(
                    mean_separation_ratio=("separation_ratio", "mean"),
                    mean_geometry_auroc=("auroc", "mean"),
                )
            )
            by_label = {row["dataset_label"]: row for row in geometry_features.to_dict(orient="records")}
            for row in feature_rows:
                extra = by_label.get(row["dataset_label"], {})
                row["mean_separation_ratio"] = extra.get("mean_separation_ratio")
                row["mean_geometry_auroc"] = extra.get("mean_geometry_auroc")

    if role_mix_rows:
        role_df = pd.DataFrame(role_mix_rows)
        if "fraction_of_deceptive" in role_df.columns:
            role_df["fraction_of_deceptive"] = pd.to_numeric(role_df["fraction_of_deceptive"], errors="coerce")
            dominant = (
                role_df.sort_values(["dataset_label", "fraction_of_deceptive"], ascending=[True, False])
                .groupby("dataset_label", as_index=False)
                .first()[["dataset_label", "role", "fraction_of_deceptive"]]
                .rename(columns={"role": "dominant_role", "fraction_of_deceptive": "dominant_role_fraction"})
            )
            by_label = {row["dataset_label"]: row for row in dominant.to_dict(orient="records")}
            for row in feature_rows:
                extra = by_label.get(row["dataset_label"], {})
                row["dominant_role"] = extra.get("dominant_role")
                row["dominant_role_fraction"] = extra.get("dominant_role_fraction")

    matrix_df = final_df.pivot_table(index="dataset_label", columns="axis_label", values="delta")
    matrix_df = matrix_df.fillna(matrix_df.mean(axis=0)).fillna(0.0)
    similarity_rows = build_similarity_table(matrix_df, "dataset")
    cluster_rows = maybe_cluster(matrix_df, "dataset", cluster_count)
    matrix_rows = (
        matrix_df.reset_index()
        .rename(columns={"dataset_label": "dataset"})
        .to_dict(orient="records")
    )
    return {"features": feature_rows, "similarity": similarity_rows, "clusters": cluster_rows, "matrix": matrix_rows}


def build_axis_similarity_tables(final_dataset_rows: list[dict[str, Any]], axis_roles: list[dict[str, Any]], axis_questions: list[dict[str, Any]], cluster_count: int) -> dict[str, list[dict[str, Any]]]:
    import pandas as pd

    final_df = pd.DataFrame(final_dataset_rows)
    if final_df.empty:
        return {"similarity": [], "clusters": [], "matrix": []}
    final_df["delta"] = pd.to_numeric(final_df["delta"], errors="coerce")
    matrix_df = final_df.pivot_table(index="axis_label", columns="dataset_label", values="delta")
    matrix_df = matrix_df.fillna(matrix_df.mean(axis=0)).fillna(0.0)
    similarity_rows = build_similarity_table(matrix_df, "axis")

    roles_by_axis: dict[str, set[str]] = {}
    for row in axis_roles:
        roles_by_axis.setdefault(str(row["axis_label"]), set()).add(str(row["role_name"]))
    questions_by_axis: dict[str, set[str]] = {}
    for row in axis_questions:
        questions_by_axis.setdefault(str(row["axis_label"]), set()).add(str(row["question_id"]))

    by_pair: list[dict[str, Any]] = []
    axis_labels = list(matrix_df.index)
    for i, left in enumerate(axis_labels):
        for j, right in enumerate(axis_labels):
            if j <= i:
                continue
            by_pair.append(
                {
                    "axis_a": left,
                    "axis_b": right,
                    "role_jaccard": jaccard(roles_by_axis.get(left, set()), roles_by_axis.get(right, set())),
                    "question_jaccard": jaccard(questions_by_axis.get(left, set()), questions_by_axis.get(right, set())),
                }
            )
    pair_lookup = {(row["axis_a"], row["axis_b"]): row for row in by_pair}
    for row in similarity_rows:
        extra = pair_lookup.get((row["axis_a"], row["axis_b"]), {})
        row["role_jaccard"] = extra.get("role_jaccard")
        row["question_jaccard"] = extra.get("question_jaccard")

    cluster_rows = maybe_cluster(matrix_df, "axis", cluster_count)
    matrix_rows = matrix_df.reset_index().rename(columns={"axis_label": "axis"}).to_dict(orient="records")
    return {"similarity": similarity_rows, "clusters": cluster_rows, "matrix": matrix_rows}


def aggregate_run_payloads(per_run_payloads: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    data = {
        "search_runs": [],
        "search_steps": [],
        "search_candidates": [],
        "final_dataset_performance": [],
        "baseline_dataset_performance": [],
        "final_axes": [],
        "final_axis_roles": [],
        "final_axis_questions": [],
        "geometry_per_dataset": [],
        "role_mix_per_dataset": [],
    }
    for payload in per_run_payloads:
        data["search_runs"].append(payload["run"])
        data["search_steps"].extend(payload["steps"])
        data["search_candidates"].extend(payload["candidates"])
        data["final_dataset_performance"].extend(payload["final_dataset_rows"])
        data["baseline_dataset_performance"].extend(payload["baseline_dataset_rows"])
        data["final_axes"].append(payload["run"])
        data["final_axis_roles"].extend(payload["axis_roles"])
        data["final_axis_questions"].extend(payload["axis_questions"])
        data["geometry_per_dataset"].extend(payload["geometry_rows"])
        data["role_mix_per_dataset"].extend(payload["role_mix_rows"])
    return data


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root(Path(__file__).resolve())
    runs_root = args.runs_root.resolve() if args.runs_root else repo_root / "artifacts" / "runs" / "fixed-track-search"
    geometry_root = args.geometry_root.resolve() if args.geometry_root else repo_root / "artifacts" / "analysis" / "dataset-geometry"
    output_root = args.output_root.resolve() if args.output_root else repo_root / "artifacts" / "analysis" / "greedy-search-evolution"
    run_id = args.run_id or make_run_id()
    run_root = output_root / run_id

    for relative in ("inputs", "checkpoints", "results", "results/per_run", "meta", "logs"):
        ensure_dir(run_root / relative)

    progress_path = run_root / "checkpoints" / "progress.json"
    progress = load_progress(progress_path) if args.resume else {"completed_axis_ids": [], "updated_at": None}
    completed_axis_ids = set(progress.get("completed_axis_ids", []))

    run_paths = discover_search_runs(runs_root, set(args.variant))
    geometry_mapping = discover_geometry_runs(geometry_root)

    write_analysis_manifest(
        run_root,
        {
            "analysis_kind": "greedy-search-evolution",
            "repo_root": str(repo_root),
            "runs_root": str(runs_root),
            "geometry_root": str(geometry_root),
            "variant_filters": list(args.variant),
            "run_id": run_id,
            "resume": bool(args.resume),
            "force": bool(args.force),
            "cluster_count": args.cluster_count,
            "created_at": utc_now_iso(),
        },
    )
    write_stage_status(run_root, "analyze_greedy_search_evolution", "running", {"run_count": len(run_paths)})
    write_json(run_root / "inputs" / "source_runs.json", {"source_runs": [str(path.resolve()) for path in run_paths]})

    per_run_payloads: list[dict[str, Any]] = []
    for search_run_dir in run_paths:
        relative_parts = search_run_dir.resolve().relative_to(runs_root.resolve()).parts
        axis_id = axis_id_for_run(relative_parts)
        per_run_path = run_root / "results" / "per_run" / f"{axis_id}.json"
        if axis_id in completed_axis_ids and per_run_path.exists() and not args.force:
            per_run_payloads.append(read_json(per_run_path))
            continue
        payload = normalize_run(search_run_dir, runs_root, geometry_mapping)
        write_json(per_run_path, payload)
        per_run_payloads.append(payload)
        completed_axis_ids.add(axis_id)
        save_progress(progress_path, {"completed_axis_ids": sorted(completed_axis_ids)})

    aggregated = aggregate_run_payloads(per_run_payloads)
    drop_summary_rows, dataset_drop_rows = build_drop_summaries(
        aggregated["search_steps"],
        aggregated["final_dataset_performance"],
    )
    dataset_tables = build_dataset_feature_tables(
        aggregated["final_dataset_performance"],
        aggregated["geometry_per_dataset"],
        aggregated["role_mix_per_dataset"],
        args.cluster_count,
    )
    axis_tables = build_axis_similarity_tables(
        aggregated["final_dataset_performance"],
        aggregated["final_axis_roles"],
        aggregated["final_axis_questions"],
        args.cluster_count,
    )

    outputs = {
        "search_runs.csv": aggregated["search_runs"],
        "search_steps.csv": aggregated["search_steps"],
        "search_candidates.csv": aggregated["search_candidates"],
        "final_dataset_performance.csv": aggregated["final_dataset_performance"],
        "baseline_dataset_performance.csv": aggregated["baseline_dataset_performance"],
        "final_axes.csv": aggregated["final_axes"],
        "final_axis_roles.csv": aggregated["final_axis_roles"],
        "final_axis_questions.csv": aggregated["final_axis_questions"],
        "geometry_per_dataset.csv": aggregated["geometry_per_dataset"],
        "role_mix_per_dataset.csv": aggregated["role_mix_per_dataset"],
        "item_drop_summary.csv": drop_summary_rows,
        "dataset_association_after_drop.csv": dataset_drop_rows,
        "dataset_features.csv": dataset_tables["features"],
        "dataset_similarity.csv": dataset_tables["similarity"],
        "dataset_clusters.csv": dataset_tables["clusters"],
        "dataset_run_matrix.csv": dataset_tables["matrix"],
        "axis_similarity.csv": axis_tables["similarity"],
        "axis_clusters.csv": axis_tables["clusters"],
        "axis_dataset_matrix.csv": axis_tables["matrix"],
    }

    for filename, rows in outputs.items():
        if rows:
            write_csv(run_root / "results" / filename, list(rows[0].keys()), rows)

    write_json(
        run_root / "results" / "summary.json",
        {
            "search_run_count": len(aggregated["search_runs"]),
            "accepted_step_count": len(aggregated["search_steps"]),
            "candidate_row_count": len(aggregated["search_candidates"]),
            "final_axis_count": len(aggregated["final_axes"]),
            "dataset_count": len({row["dataset_label"] for row in aggregated["final_dataset_performance"]}),
            "generated_at": utc_now_iso(),
        },
    )

    write_stage_status(
        run_root,
        "analyze_greedy_search_evolution",
        "completed",
        {
            "search_run_count": len(aggregated["search_runs"]),
            "accepted_step_count": len(aggregated["search_steps"]),
            "candidate_row_count": len(aggregated["search_candidates"]),
        },
    )
    print(f"[greedy-search-evolution] wrote analysis artifacts to {run_root}")


if __name__ == "__main__":
    main()
