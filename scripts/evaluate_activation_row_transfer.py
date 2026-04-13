#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from deception_honesty_axis.activation_row_transfer import (
    ACTIVATION_TRANSFER_METHODS,
    ensure_loaded_dataset_compatibility,
    load_activation_row_dataset,
)
from deception_honesty_axis.activation_row_transfer_config import load_activation_row_transfer_config
from deception_honesty_axis.common import ensure_dir, make_run_id, slugify, write_json
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
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
        description="Train linear probes on cached activation-row datasets and evaluate them across datasets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the activation-row transfer config JSON.",
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
        help="Optional HF dataset repo to upload this evaluation run folder into.",
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


def main() -> None:
    args = parse_args()
    config = load_activation_row_transfer_config(args.config.resolve())
    invalid_methods = [method for method in config.methods if method not in ACTIVATION_TRANSFER_METHODS]
    if invalid_methods:
        raise ValueError(f"Unsupported activation-row transfer methods requested: {invalid_methods}")

    loaded_datasets = [
        load_activation_row_dataset(
            dataset_config,
            expected_pooling=config.expected_pooling,
            expected_layer_number=config.expected_layer_number,
        )
        for dataset_config in config.datasets
    ]
    compatibility = ensure_loaded_dataset_compatibility(loaded_datasets)
    resolved_pooling = compatibility["activation_pooling"] or config.expected_pooling or "unknown"
    resolved_layer_number = compatibility["activation_layer_number"]
    if resolved_layer_number is None and config.expected_layer_number is not None:
        resolved_layer_number = config.expected_layer_number
    layer_spec = str(resolved_layer_number) if resolved_layer_number is not None else "unknown"
    layer_label = f"L{resolved_layer_number}" if resolved_layer_number is not None else "Unknown"

    run_id = args.run_id or make_run_id()
    variant_slug = slugify(f"{resolved_pooling}-layer-{layer_spec}")
    run_root = (
        config.artifact_root
        / "runs"
        / slugify(config.experiment_name)
        / slugify(config.model_name)
        / slugify(config.behavior_name)
        / slugify(config.dataset_set_name)
        / variant_slug
        / run_id
    )
    for relative in ("inputs", "results", "logs", "checkpoints", "meta"):
        ensure_dir(run_root / relative)

    dataset_summaries = [dataset.summary() for dataset in loaded_datasets]
    dataset_names = [dataset.name for dataset in loaded_datasets]
    write_analysis_manifest(
        run_root,
        {
            "config_path": str(config.path),
            "run_id": run_id,
            "experiment_name": config.experiment_name,
            "behavior_name": config.behavior_name,
            "model_name": config.model_name,
            "dataset_set_name": config.dataset_set_name,
            "methods": config.methods,
            "expected_pooling": config.expected_pooling,
            "expected_layer_number": config.expected_layer_number,
            "resolved_pooling": resolved_pooling,
            "resolved_layer_number": resolved_layer_number,
            "feature_dim": compatibility["feature_dim"],
            "datasets": [dataset_config.manifest_row() for dataset_config in config.datasets],
            "dataset_summaries": dataset_summaries,
        },
    )
    write_json(run_root / "inputs" / "config.json", config.raw)
    write_json(run_root / "inputs" / "dataset_summaries.json", dataset_summaries)

    results_dir = run_root / "results"
    metrics_path = results_dir / "pairwise_metrics.csv"
    score_rows_path = results_dir / "per_example_scores.jsonl"
    fit_artifacts_path = results_dir / "source_fit_artifacts.jsonl"
    progress_path = run_root / "checkpoints" / "progress.json"
    completed_keys = read_completed_metric_keys(metrics_path)
    total_expected = len(config.methods) * len(loaded_datasets) * len(loaded_datasets)
    completed_now = 0

    write_stage_status(
        run_root,
        "evaluate_activation_row_transfer",
        "running",
        {
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "datasets": dataset_names,
        },
    )
    write_json(
        progress_path,
        {
            "state": "running",
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "completed_this_run": 0,
        },
    )

    if completed_keys:
        print(
            f"[activation-row-transfer] resuming with {len(completed_keys)}/{total_expected} "
            "combinations already complete"
        )
    else:
        print(
            f"[activation-row-transfer] starting fresh with {total_expected} combinations "
            f"across {len(loaded_datasets)} datasets"
        )

    def update_progress() -> None:
        payload = {
            "state": "running",
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "completed_this_run": completed_now,
            "remaining_combinations": max(0, total_expected - len(completed_keys)),
        }
        write_json(progress_path, payload)
        write_stage_status(run_root, "evaluate_activation_row_transfer", "running", payload)

    def maybe_print_progress(last_row: dict[str, Any]) -> None:
        if completed_now == 1 or (args.print_every > 0 and completed_now % args.print_every == 0):
            print(
                f"[activation-row-transfer] {len(completed_keys)}/{total_expected} complete "
                f"({completed_now} new this run); "
                f"{last_row['method']} @ {last_row['layer_spec']} "
                f"{last_row['source_dataset']} -> {last_row['target_dataset']} "
                f"AUROC={float(last_row['auroc']):.3f}"
            )

    for source_dataset in loaded_datasets:
        for target_dataset in loaded_datasets:
            for method in config.methods:
                metric_key = (method, layer_spec, source_dataset.name, target_dataset.name)
                if metric_key in completed_keys:
                    continue

                if method == "activation_logistic":
                    fit_artifact, probabilities, predictions = fit_logistic_scores(
                        source_dataset.train.features,
                        source_dataset.train.labels,
                        target_dataset.eval.features,
                        random_seed=config.random_seed,
                        max_iter=config.logistic_max_iter,
                    )
                else:
                    raise ValueError(f"Unsupported activation-row transfer method {method!r}")

                metrics = compute_binary_metrics(
                    target_dataset.eval.labels,
                    probabilities,
                    predictions,
                    probabilities=probabilities,
                )
                row = {
                    "method": method,
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
                        "method": method,
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
                        **fit_artifact,
                    },
                )
                save_score_rows(
                    score_rows_path,
                    [
                        {
                            "method": method,
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
        config.methods,
        [layer_spec],
        dataset_names,
    )
    write_json(
        results_dir / "results.json",
        {
            "record_count": len(metric_rows),
            "datasets": dataset_names,
            "methods": config.methods,
            "resolved_pooling": resolved_pooling,
            "resolved_layer_number": resolved_layer_number,
            "feature_dim": compatibility["feature_dim"],
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
            "heatmaps": heatmap_paths,
        },
    )
    write_stage_status(
        run_root,
        "evaluate_activation_row_transfer",
        "completed",
        {
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
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
        print(f"[activation-row-transfer] uploaded to hf://datasets/{args.push_to_hub_repo_id}/{path_in_repo}")

    print(
        f"[activation-row-transfer] completed {completed_now} new combinations "
        f"({len(completed_keys)}/{total_expected} total); wrote results to {run_root}"
    )


if __name__ == "__main__":
    main()
