#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

from deception_honesty_axis.activation_row_transfer import (
    ensure_loaded_dataset_compatibility,
    load_activation_row_dataset,
)
from deception_honesty_axis.activation_row_transfer_config import load_activation_row_transfer_config
from deception_honesty_axis.common import ensure_dir, make_run_id, read_json, slugify, write_json
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.persona_coordinates import (
    PERSONA_COORDINATE_METHOD,
    coordinate_loaded_dataset,
    ensure_persona_coordinate_dataset_compatibility,
    load_persona_role_matrix,
)
from deception_honesty_axis.role_axis_transfer import (
    append_csv_row,
    compute_binary_metrics,
    fit_logistic_scores,
    read_completed_metric_keys,
    save_fit_artifact,
    save_metric_heatmaps,
    save_score_rows,
    write_fit_summary_csv,
    write_summary_csv,
)
from deception_honesty_axis.sycophancy_activations import normalize_activation_pooling


METRIC_FIELDS = [
    "method",
    "layer_spec",
    "layer_label",
    "source_dataset",
    "target_dataset",
    "train_split",
    "eval_split",
    "train_count",
    "auroc",
    "auprc",
    "balanced_accuracy",
    "f1",
    "accuracy",
    "count",
    "completed_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train pairwise logistic probes on persona-coordinate features: dot products between "
            "mean-pooled dataset activations and each stored role vector in a unified-axis run."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Activation-row transfer config JSON.")
    parser.add_argument(
        "--unified-axis-run-dir",
        type=Path,
        required=True,
        help="Unified axis run dir containing results/role_vectors.pt.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id for resume/replay.")
    parser.add_argument(
        "--include-anchor",
        action="store_true",
        help="Include the anchor/default role as a coordinate feature.",
    )
    parser.add_argument(
        "--role",
        dest="roles",
        action="append",
        default=None,
        help="Optional explicit role name to include. Repeat to define order. Defaults to safe roles then harmful roles.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=5,
        help="Print a progress update every N newly completed combinations.",
    )
    parser.add_argument(
        "--push-to-hub-repo-id",
        default=None,
        help="Optional HF dataset repo to upload this run folder into.",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Optional destination path in the HF artifacts repo. Defaults to the run path under artifacts/.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def unified_axis_expected_pooling(unified_axis_run_dir: Path) -> str | None:
    manifest_path = unified_axis_run_dir / "meta" / "run_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    pooling = manifest.get("expected_pooling")
    if pooling not in (None, ""):
        return normalize_activation_pooling(str(pooling))
    return None


def infer_unified_axis_slug(unified_axis_run_dir: Path) -> str:
    manifest_path = unified_axis_run_dir / "meta" / "run_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        axis_name = manifest.get("axis_name")
        variant_name = manifest.get("variant_name")
        if axis_name and variant_name:
            return slugify(f"{axis_name}-{variant_name}")
        if axis_name:
            return slugify(str(axis_name))
    parts = list(unified_axis_run_dir.parts)
    if len(parts) >= 2:
        return slugify("-".join(parts[-2:]))
    return slugify(unified_axis_run_dir.name)


def main() -> None:
    args = parse_args()
    config = load_activation_row_transfer_config(args.config.resolve())
    loaded_datasets = [
        load_activation_row_dataset(
            dataset_config,
            expected_pooling=config.expected_pooling,
            expected_layer_number=config.expected_layer_number,
        )
        for dataset_config in config.datasets
    ]
    raw_compatibility = ensure_loaded_dataset_compatibility(loaded_datasets)

    unified_axis_run_dir = args.unified_axis_run_dir.resolve()
    expected_axis_pooling = unified_axis_expected_pooling(unified_axis_run_dir)
    if expected_axis_pooling is not None and raw_compatibility["activation_pooling"] is not None:
        if normalize_activation_pooling(expected_axis_pooling) != normalize_activation_pooling(
            raw_compatibility["activation_pooling"]
        ):
            raise ValueError(
                f"Unified axis expects pooling {expected_axis_pooling!r}, "
                f"but loaded datasets resolved to {raw_compatibility['activation_pooling']!r}"
            )

    role_matrix = load_persona_role_matrix(
        unified_axis_run_dir,
        expected_layer_number=config.expected_layer_number,
        include_anchor=bool(args.include_anchor),
        role_order=args.roles,
    )
    if role_matrix.feature_dim != int(raw_compatibility["feature_dim"]):
        raise ValueError(
            f"Unified role-vector feature dim {role_matrix.feature_dim} does not match loaded datasets "
            f"feature dim {raw_compatibility['feature_dim']}"
        )

    run_id = args.run_id or make_run_id()
    axis_slug = infer_unified_axis_slug(unified_axis_run_dir)
    layer_label = f"L{role_matrix.activation_layer_number}" if role_matrix.activation_layer_number is not None else "single-layer"
    layer_spec = str(role_matrix.activation_layer_number or "single")
    variant_slug = slugify(
        f"{raw_compatibility['activation_pooling'] or config.expected_pooling or 'unknown'}-"
        f"layer-{layer_spec}-roles-{len(role_matrix.role_names)}"
    )
    run_root = (
        config.artifact_root
        / "runs"
        / "persona-coordinates-probing"
        / slugify(config.model_name)
        / slugify(config.behavior_name)
        / slugify(config.dataset_set_name)
        / axis_slug
        / variant_slug
        / run_id
    )
    for relative in ("inputs", "results", "logs", "checkpoints", "meta"):
        ensure_dir(run_root / relative)

    coordinate_root = ensure_dir(run_root / "results" / "coordinate_splits")
    coordinate_datasets = [
        coordinate_loaded_dataset(
            dataset,
            role_matrix=role_matrix,
            coordinate_root=coordinate_root,
        )
        for dataset in loaded_datasets
    ]
    coordinate_compatibility = ensure_persona_coordinate_dataset_compatibility(coordinate_datasets)
    dataset_summaries = [dataset.summary() for dataset in coordinate_datasets]
    dataset_names = [dataset.name for dataset in coordinate_datasets]

    write_analysis_manifest(
        run_root,
        {
            "config_path": str(config.path),
            "run_id": run_id,
            "experiment_name": "persona-coordinates-probing",
            "behavior_name": config.behavior_name,
            "model_name": config.model_name,
            "dataset_set_name": config.dataset_set_name,
            "unified_axis_run_dir": str(unified_axis_run_dir),
            "unified_axis_slug": axis_slug,
            "layer_spec": layer_spec,
            "layer_label": layer_label,
            "raw_feature_dim": raw_compatibility["feature_dim"],
            "coordinate_feature_dim": coordinate_compatibility["coordinate_feature_dim"],
            "role_names": coordinate_compatibility["role_names"],
            "role_sides": coordinate_compatibility["role_sides"],
            "expected_pooling": config.expected_pooling,
            "resolved_pooling": raw_compatibility["activation_pooling"],
            "resolved_layer_number": raw_compatibility["activation_layer_number"],
            "datasets": [dataset_config.manifest_row() for dataset_config in config.datasets],
            "dataset_summaries": dataset_summaries,
            "unified_axis_source": role_matrix.source_manifest,
        },
    )
    write_json(run_root / "inputs" / "config.json", config.raw)
    write_json(run_root / "inputs" / "persona_role_matrix_manifest.json", role_matrix.source_manifest)
    write_json(run_root / "inputs" / "coordinate_dataset_summaries.json", dataset_summaries)

    results_dir = run_root / "results"
    metrics_path = results_dir / "pairwise_metrics.csv"
    score_rows_path = results_dir / "per_example_scores.jsonl"
    fit_artifacts_path = results_dir / "source_fit_artifacts.jsonl"
    progress_path = run_root / "checkpoints" / "progress.json"
    completed_keys = read_completed_metric_keys(metrics_path)
    total_expected = len(coordinate_datasets) * len(coordinate_datasets)
    completed_now = 0

    write_stage_status(
        run_root,
        "evaluate_persona_coordinates_probe",
        "running",
        {
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "datasets": dataset_names,
            "unified_axis_slug": axis_slug,
            "role_count": len(role_matrix.role_names),
        },
    )
    write_json(
        progress_path,
        {
            "state": "running",
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "completed_this_run": 0,
            "unified_axis_slug": axis_slug,
            "role_count": len(role_matrix.role_names),
        },
    )

    if completed_keys:
        print(
            f"[persona-coordinates] resuming with {len(completed_keys)}/{total_expected} "
            "combinations already complete"
        )
    else:
        print(
            f"[persona-coordinates] starting fresh with {total_expected} combinations across "
            f"{len(coordinate_datasets)} datasets using {len(role_matrix.role_names)} role coordinates"
        )

    def update_progress() -> None:
        payload = {
            "state": "running",
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "completed_this_run": completed_now,
            "remaining_combinations": max(0, total_expected - len(completed_keys)),
            "unified_axis_slug": axis_slug,
            "role_count": len(role_matrix.role_names),
        }
        write_json(progress_path, payload)
        write_stage_status(run_root, "evaluate_persona_coordinates_probe", "running", payload)

    def maybe_print_progress(last_row: dict[str, Any]) -> None:
        if completed_now == 1 or (args.print_every > 0 and completed_now % args.print_every == 0):
            print(
                f"[persona-coordinates] {len(completed_keys)}/{total_expected} complete "
                f"({completed_now} new this run); "
                f"{last_row['source_dataset']} -> {last_row['target_dataset']} "
                f"AUROC={float(last_row['auroc']):.3f}"
            )

    for source_dataset in coordinate_datasets:
        for target_dataset in coordinate_datasets:
            metric_key = (PERSONA_COORDINATE_METHOD, layer_spec, source_dataset.name, target_dataset.name)
            if metric_key in completed_keys:
                continue

            fit_artifact, probabilities, predictions = fit_logistic_scores(
                source_dataset.train.features,
                source_dataset.train.labels,
                target_dataset.eval.features,
                random_seed=config.random_seed,
                max_iter=config.logistic_max_iter,
            )
            metrics = compute_binary_metrics(
                target_dataset.eval.labels,
                probabilities,
                predictions,
                probabilities=probabilities,
            )
            row = {
                "method": PERSONA_COORDINATE_METHOD,
                "layer_spec": layer_spec,
                "layer_label": layer_label,
                "source_dataset": source_dataset.name,
                "target_dataset": target_dataset.name,
                "train_split": source_dataset.train.source_manifest.get("split", source_dataset.train.split_name),
                "eval_split": target_dataset.eval.source_manifest.get("split", target_dataset.eval.split_name),
                "train_count": int(source_dataset.train.labels.shape[0]),
                "completed_at": utc_now_iso(),
                **metrics,
            }
            append_csv_row(metrics_path, METRIC_FIELDS, row)
            save_fit_artifact(
                fit_artifacts_path,
                {
                    "method": PERSONA_COORDINATE_METHOD,
                    "layer_spec": layer_spec,
                    "layer_label": layer_label,
                    "source_dataset": source_dataset.name,
                    "target_dataset": target_dataset.name,
                    "train_split": row["train_split"],
                    "eval_split": row["eval_split"],
                    "train_count": int(source_dataset.train.labels.shape[0]),
                    "eval_count": int(target_dataset.eval.labels.shape[0]),
                    "train_label_counts": source_dataset.train.summary()["label_counts"],
                    "eval_label_counts": target_dataset.eval.summary()["label_counts"],
                    "unified_axis_slug": axis_slug,
                    "role_names": role_matrix.role_names,
                    "role_sides": role_matrix.role_sides,
                    "coordinate_feature_dim": coordinate_compatibility["coordinate_feature_dim"],
                    **fit_artifact,
                },
            )
            save_score_rows(
                score_rows_path,
                [
                    {
                        "method": PERSONA_COORDINATE_METHOD,
                        "layer_spec": layer_spec,
                        "source_dataset": source_dataset.name,
                        "target_dataset": target_dataset.name,
                        "sample_id": sample_id,
                        "label": int(label),
                        "positive_score": float(probability),
                        "probability": float(probability),
                        "prediction": int(prediction),
                    }
                    for sample_id, label, probability, prediction in zip(
                        target_dataset.eval.sample_ids,
                        target_dataset.eval.labels,
                        probabilities,
                        predictions,
                        strict=False,
                    )
                ],
            )
            completed_keys.add(metric_key)
            completed_now += 1
            update_progress()
            maybe_print_progress(row)

    metric_rows: list[dict[str, str]] = []
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            metric_rows = list(csv.DictReader(handle))

    write_summary_csv(results_dir / "summary_by_method.csv", metric_rows)
    write_fit_summary_csv(results_dir / "fit_summary.csv", fit_artifacts_path)
    heatmap_paths = save_metric_heatmaps(
        results_dir / "plots",
        metric_rows,
        [PERSONA_COORDINATE_METHOD],
        [layer_spec],
        dataset_names,
    )
    write_json(
        results_dir / "results.json",
        {
            "record_count": len(metric_rows),
            "datasets": dataset_names,
            "method": PERSONA_COORDINATE_METHOD,
            "resolved_pooling": raw_compatibility["activation_pooling"],
            "resolved_layer_number": raw_compatibility["activation_layer_number"],
            "raw_feature_dim": raw_compatibility["feature_dim"],
            "coordinate_feature_dim": coordinate_compatibility["coordinate_feature_dim"],
            "role_names": role_matrix.role_names,
            "role_sides": role_matrix.role_sides,
            "unified_axis_run_dir": str(unified_axis_run_dir),
            "unified_axis_slug": axis_slug,
            "coordinate_splits_dir": str(coordinate_root),
            "heatmaps": heatmap_paths,
        },
    )
    write_json(
        progress_path,
        {
            "state": "completed",
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "completed_this_run": completed_now,
            "remaining_combinations": max(0, total_expected - len(completed_keys)),
            "unified_axis_slug": axis_slug,
            "role_count": len(role_matrix.role_names),
            "heatmaps": heatmap_paths,
        },
    )
    write_stage_status(
        run_root,
        "evaluate_persona_coordinates_probe",
        "completed",
        {
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "unified_axis_slug": axis_slug,
            "role_count": len(role_matrix.role_names),
            "heatmaps": heatmap_paths,
        },
    )

    if args.push_to_hub_repo_id:
        from huggingface_hub import upload_folder

        default_path_in_repo = str(run_root.relative_to(config.artifact_root))
        path_in_repo = args.path_in_repo or default_path_in_repo
        upload_folder(
            repo_id=args.push_to_hub_repo_id,
            repo_type="dataset",
            folder_path=str(run_root),
            path_in_repo=path_in_repo,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"[persona-coordinates] uploaded to hf://datasets/{args.push_to_hub_repo_id}/{path_in_repo}")

    print(
        f"[persona-coordinates] completed {completed_now} new combinations "
        f"({len(completed_keys)}/{total_expected} total); wrote results to {run_root}"
    )


if __name__ == "__main__":
    main()
