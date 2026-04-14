#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

from deception_honesty_axis.activation_row_projection_transfer import (
    PROJECTED_TRANSFER_METHOD,
    ensure_projected_dataset_compatibility,
    project_loaded_dataset,
    resolve_axis_layer_bundle,
)
from deception_honesty_axis.activation_row_transfer import (
    ensure_loaded_dataset_compatibility,
    load_activation_row_dataset,
)
from deception_honesty_axis.activation_row_transfer_config import load_activation_row_transfer_config
from deception_honesty_axis.common import ensure_dir, make_run_id, read_json, slugify, write_json
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.role_axis_transfer import (
    append_csv_row,
    compute_binary_metrics,
    fit_logistic_scores,
    load_role_axis_bundle,
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
        description="Train linear probes on PC-projected features from a role-axis bundle and evaluate transfer."
    )
    parser.add_argument("--config", type=Path, required=True, help="Activation-row transfer config JSON.")
    parser.add_argument(
        "--axis-bundle-run-dir",
        type=Path,
        required=True,
        help="Run dir containing results/axis_bundle.pt.",
    )
    parser.add_argument(
        "--max-pcs",
        type=int,
        default=None,
        help="Optional maximum number of leading PCs to use. Defaults to all stored PCs for the matched layer.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id for resume/replay.")
    parser.add_argument(
        "--print-every",
        type=int,
        default=5,
        help="Print a progress update every N newly completed combinations.",
    )
    parser.add_argument(
        "--push-to-hub-repo-id",
        default=None,
        help="Optional HF dataset repo to upload this projected transfer run into.",
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


def axis_expected_pooling(axis_bundle_dir: Path) -> str | None:
    manifest_path = axis_bundle_dir / "meta" / "run_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    pooling = manifest.get("expected_pooling")
    if pooling not in (None, ""):
        return normalize_activation_pooling(str(pooling))
    return None


def infer_axis_slug(axis_bundle_dir: Path) -> str:
    manifest_path = axis_bundle_dir / "meta" / "run_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        axis_name = manifest.get("axis_name")
        variant_name = manifest.get("variant_name")
        if axis_name and variant_name:
            return slugify(f"{axis_name}-{variant_name}")
        if axis_name:
            return slugify(str(axis_name))
    parts = list(axis_bundle_dir.parts)
    if len(parts) >= 2:
        return slugify("-".join(parts[-2:]))
    return slugify(axis_bundle_dir.name)


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

    axis_bundle_dir = args.axis_bundle_run_dir.resolve()
    axis_bundle = load_role_axis_bundle(axis_bundle_dir / "results" / "axis_bundle.pt")
    expected_axis_pooling = axis_expected_pooling(axis_bundle_dir)
    if expected_axis_pooling is not None and raw_compatibility["activation_pooling"] is not None:
        if normalize_activation_pooling(expected_axis_pooling) != normalize_activation_pooling(
            raw_compatibility["activation_pooling"]
        ):
            raise ValueError(
                f"Axis bundle expects pooling {expected_axis_pooling!r}, "
                f"but loaded datasets resolved to {raw_compatibility['activation_pooling']!r}"
            )

    layer_spec, layer_label, layer_bundle = resolve_axis_layer_bundle(
        axis_bundle,
        expected_layer_number=config.expected_layer_number,
    )
    axis_feature_dim = int(layer_bundle["pca_mean"].shape[0])
    if axis_feature_dim != int(raw_compatibility["feature_dim"]):
        raise ValueError(
            f"Axis bundle feature dim {axis_feature_dim} does not match loaded datasets "
            f"feature dim {raw_compatibility['feature_dim']}"
        )

    available_pc_count = int(layer_bundle["pc_components"].shape[0])
    selected_pc_count = available_pc_count if args.max_pcs is None else min(int(args.max_pcs), available_pc_count)
    if selected_pc_count <= 0:
        raise ValueError(f"Projected transfer requires at least one PC, got max_pcs={args.max_pcs!r}")

    run_id = args.run_id or make_run_id()
    axis_slug = infer_axis_slug(axis_bundle_dir)
    variant_slug = slugify(
        f"{raw_compatibility['activation_pooling'] or config.expected_pooling or 'unknown'}-"
        f"layer-{layer_spec}-pcs-{selected_pc_count}"
    )
    run_root = (
        config.artifact_root
        / "runs"
        / "activation-row-transfer-pc-projection"
        / slugify(config.model_name)
        / slugify(config.behavior_name)
        / slugify(config.dataset_set_name)
        / axis_slug
        / variant_slug
        / run_id
    )
    for relative in ("inputs", "results", "logs", "checkpoints", "meta"):
        ensure_dir(run_root / relative)

    projected_root = ensure_dir(run_root / "results" / "projected_splits")
    projected_datasets = [
        project_loaded_dataset(
            dataset,
            layer_bundle=layer_bundle,
            layer_spec=layer_spec,
            layer_label=layer_label,
            max_pcs=selected_pc_count,
            axis_bundle_run_dir=axis_bundle_dir,
            projected_root=projected_root,
        )
        for dataset in loaded_datasets
    ]
    projected_compatibility = ensure_projected_dataset_compatibility(projected_datasets)
    dataset_summaries = [dataset.summary() for dataset in projected_datasets]
    dataset_names = [dataset.name for dataset in projected_datasets]

    write_analysis_manifest(
        run_root,
        {
            "config_path": str(config.path),
            "run_id": run_id,
            "experiment_name": "activation-row-transfer-pc-projection",
            "behavior_name": config.behavior_name,
            "model_name": config.model_name,
            "dataset_set_name": config.dataset_set_name,
            "axis_bundle_run_dir": str(axis_bundle_dir),
            "axis_slug": axis_slug,
            "layer_spec": layer_spec,
            "layer_label": layer_label,
            "raw_feature_dim": raw_compatibility["feature_dim"],
            "projected_feature_dim": projected_compatibility["projected_feature_dim"],
            "available_pc_count": available_pc_count,
            "selected_pc_count": projected_compatibility["selected_pc_count"],
            "selected_pc_indices": projected_compatibility["selected_pc_indices"],
            "expected_pooling": config.expected_pooling,
            "resolved_pooling": raw_compatibility["activation_pooling"],
            "resolved_layer_number": raw_compatibility["activation_layer_number"],
            "datasets": [dataset_config.manifest_row() for dataset_config in config.datasets],
            "dataset_summaries": dataset_summaries,
        },
    )
    write_json(run_root / "inputs" / "config.json", config.raw)
    write_json(run_root / "inputs" / "projected_dataset_summaries.json", dataset_summaries)

    results_dir = run_root / "results"
    metrics_path = results_dir / "pairwise_metrics.csv"
    score_rows_path = results_dir / "per_example_scores.jsonl"
    fit_artifacts_path = results_dir / "source_fit_artifacts.jsonl"
    progress_path = run_root / "checkpoints" / "progress.json"
    completed_keys = read_completed_metric_keys(metrics_path)
    total_expected = len(projected_datasets) * len(projected_datasets)
    completed_now = 0

    write_stage_status(
        run_root,
        "evaluate_activation_row_transfer_pc_projection",
        "running",
        {
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "datasets": dataset_names,
            "axis_slug": axis_slug,
            "selected_pc_count": selected_pc_count,
        },
    )
    write_json(
        progress_path,
        {
            "state": "running",
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "completed_this_run": 0,
            "axis_slug": axis_slug,
            "selected_pc_count": selected_pc_count,
        },
    )

    if completed_keys:
        print(
            f"[activation-row-transfer-pc-projection] resuming with {len(completed_keys)}/{total_expected} "
            "combinations already complete"
        )
    else:
        print(
            f"[activation-row-transfer-pc-projection] starting fresh with {total_expected} combinations "
            f"across {len(projected_datasets)} datasets using {selected_pc_count}/{available_pc_count} PCs"
        )

    def update_progress() -> None:
        payload = {
            "state": "running",
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "completed_this_run": completed_now,
            "remaining_combinations": max(0, total_expected - len(completed_keys)),
            "axis_slug": axis_slug,
            "selected_pc_count": selected_pc_count,
        }
        write_json(progress_path, payload)
        write_stage_status(run_root, "evaluate_activation_row_transfer_pc_projection", "running", payload)

    def maybe_print_progress(last_row: dict[str, Any]) -> None:
        if completed_now == 1 or (args.print_every > 0 and completed_now % args.print_every == 0):
            print(
                f"[activation-row-transfer-pc-projection] {len(completed_keys)}/{total_expected} complete "
                f"({completed_now} new this run); "
                f"{last_row['source_dataset']} -> {last_row['target_dataset']} "
                f"AUROC={float(last_row['auroc']):.3f}"
            )

    for source_dataset in projected_datasets:
        for target_dataset in projected_datasets:
            metric_key = (PROJECTED_TRANSFER_METHOD, layer_spec, source_dataset.name, target_dataset.name)
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
                "method": PROJECTED_TRANSFER_METHOD,
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
                    "method": PROJECTED_TRANSFER_METHOD,
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
                    "axis_slug": axis_slug,
                    "selected_pc_count": selected_pc_count,
                    "projected_feature_dim": projected_compatibility["projected_feature_dim"],
                    **fit_artifact,
                },
            )
            save_score_rows(
                score_rows_path,
                [
                    {
                        "method": PROJECTED_TRANSFER_METHOD,
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
        [PROJECTED_TRANSFER_METHOD],
        [layer_spec],
        dataset_names,
    )
    write_json(
        results_dir / "results.json",
        {
            "record_count": len(metric_rows),
            "datasets": dataset_names,
            "method": PROJECTED_TRANSFER_METHOD,
            "resolved_pooling": raw_compatibility["activation_pooling"],
            "resolved_layer_number": raw_compatibility["activation_layer_number"],
            "raw_feature_dim": raw_compatibility["feature_dim"],
            "projected_feature_dim": projected_compatibility["projected_feature_dim"],
            "available_pc_count": available_pc_count,
            "selected_pc_count": selected_pc_count,
            "selected_pc_indices": projected_compatibility["selected_pc_indices"],
            "axis_bundle_run_dir": str(axis_bundle_dir),
            "axis_slug": axis_slug,
            "projected_splits_dir": str(projected_root),
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
            "axis_slug": axis_slug,
            "selected_pc_count": selected_pc_count,
            "heatmaps": heatmap_paths,
        },
    )
    write_stage_status(
        run_root,
        "evaluate_activation_row_transfer_pc_projection",
        "completed",
        {
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "axis_slug": axis_slug,
            "selected_pc_count": selected_pc_count,
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
        print(
            f"[activation-row-transfer-pc-projection] uploaded to hf://datasets/"
            f"{args.push_to_hub_repo_id}/{path_in_repo}"
        )

    print(
        f"[activation-row-transfer-pc-projection] completed {completed_now} new combinations "
        f"({len(completed_keys)}/{total_expected} total); wrote results to {run_root}"
    )


if __name__ == "__main__":
    main()
