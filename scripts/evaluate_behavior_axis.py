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
from deception_honesty_axis.role_axis_transfer import (
    ZERO_SHOT_METHODS,
    ZERO_SHOT_SOURCE,
    append_csv_row,
    evaluate_zero_shot,
    load_role_axis_bundle,
    read_completed_metric_keys,
    save_metric_heatmaps,
    save_score_rows,
    save_zero_shot_grouped_barplots,
    scores_for_layer,
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
    "auroc",
    "auprc",
    "balanced_accuracy",
    "f1",
    "accuracy",
    "count",
    "completed_at",
]

ZERO_SHOT_METHOD_SPECS = {
    "contrast_zero_shot": ("contrast", None),
    "pc1_zero_shot": ("pc_scores", 0),
    "pc2_zero_shot": ("pc_scores", 1),
    "pc3_zero_shot": ("pc_scores", 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved role-axis bundle over a mixed dataset inventory loaded from cached features."
    )
    parser.add_argument("--config", type=Path, required=True, help="Feature transfer config JSON.")
    parser.add_argument("--axis-bundle-run-dir", type=Path, required=True, help="Run dir containing results/axis_bundle.pt.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["contrast_zero_shot", "pc1_zero_shot", "pc2_zero_shot", "pc3_zero_shot"],
        help="Zero-shot methods to evaluate.",
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


def axis_expected_pooling(axis_bundle_dir: Path) -> str | None:
    manifest_path = axis_bundle_dir / "meta" / "run_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    pooling = manifest.get("expected_pooling")
    if pooling not in (None, ""):
        return normalize_activation_pooling(str(pooling))
    role_experiment_config_path = manifest.get("role_experiment_config_path")
    if role_experiment_config_path:
        from deception_honesty_axis.config import load_config

        config_path = Path(str(role_experiment_config_path))
        if config_path.exists():
            return normalize_activation_pooling(str(load_config(config_path).analysis.get("pooling") or ""))
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
    invalid_methods = [method for method in args.methods if method not in ZERO_SHOT_METHODS]
    if invalid_methods:
        raise ValueError(f"Only zero-shot methods are supported here; invalid={invalid_methods}")

    config = load_activation_row_transfer_config(args.config.resolve())
    loaded_datasets = [
        load_activation_row_dataset(
            dataset_config,
            expected_pooling=config.expected_pooling,
            expected_layer_number=config.expected_layer_number,
        )
        for dataset_config in config.datasets
    ]
    compatibility = ensure_loaded_dataset_compatibility(loaded_datasets)

    axis_bundle_dir = args.axis_bundle_run_dir.resolve()
    axis_bundle = load_role_axis_bundle(axis_bundle_dir / "results" / "axis_bundle.pt")
    expected_axis_pooling = axis_expected_pooling(axis_bundle_dir)
    if expected_axis_pooling is not None and compatibility["activation_pooling"] is not None:
        if normalize_activation_pooling(expected_axis_pooling) != normalize_activation_pooling(
            compatibility["activation_pooling"]
        ):
            raise ValueError(
                f"Axis bundle expects pooling {expected_axis_pooling!r}, "
                f"but loaded datasets resolved to {compatibility['activation_pooling']!r}"
            )

    layer_specs = list(axis_bundle["layers"].keys())
    if config.expected_layer_number is not None:
        compatible_specs = []
        for layer_spec in layer_specs:
            layer_numbers = [int(value) for value in (axis_bundle["layers"][layer_spec].get("layer_numbers") or [])]
            if len(layer_numbers) == 1 and int(layer_numbers[0]) == int(config.expected_layer_number):
                compatible_specs.append(layer_spec)
                continue
            try:
                parsed_layer_spec = int(str(layer_spec))
            except (TypeError, ValueError):
                parsed_layer_spec = None
            if parsed_layer_spec == int(config.expected_layer_number):
                compatible_specs.append(layer_spec)
        if not compatible_specs:
            raise ValueError(
                f"No axis bundle layers match expected layer {config.expected_layer_number}; "
                f"bundle_layers={layer_specs!r}"
            )
        layer_specs = compatible_specs

    dataset_names = [dataset.name for dataset in loaded_datasets]
    axis_slug = infer_axis_slug(axis_bundle_dir)
    run_id = args.run_id or make_run_id()
    run_root = (
        config.artifact_root
        / "runs"
        / "behavior-axis-evaluation"
        / slugify(config.model_name)
        / slugify(config.behavior_name)
        / slugify(config.dataset_set_name)
        / axis_slug
        / run_id
    )
    for relative in ("inputs", "results", "logs", "checkpoints", "meta"):
        ensure_dir(run_root / relative)

    write_analysis_manifest(
        run_root,
        {
            "config_path": str(config.path),
            "run_id": run_id,
            "axis_bundle_run_dir": str(axis_bundle_dir),
            "axis_slug": axis_slug,
            "methods": args.methods,
            "layer_specs": layer_specs,
            "resolved_pooling": compatibility["activation_pooling"],
            "resolved_layer_number": compatibility["activation_layer_number"],
            "feature_dim": compatibility["feature_dim"],
            "datasets": [dataset_config.manifest_row() for dataset_config in config.datasets],
            "dataset_summaries": [dataset.summary() for dataset in loaded_datasets],
        },
    )
    write_json(run_root / "inputs" / "config.json", config.raw)

    results_dir = run_root / "results"
    metrics_path = results_dir / "pairwise_metrics.csv"
    score_rows_path = results_dir / "per_example_scores.jsonl"
    progress_path = run_root / "checkpoints" / "progress.json"
    completed_keys = read_completed_metric_keys(metrics_path)
    total_expected = len(layer_specs) * len(dataset_names) * len(args.methods)
    completed_now = 0

    write_stage_status(
        run_root,
        "evaluate_behavior_axis",
        "running",
        {
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "datasets": dataset_names,
            "axis_slug": axis_slug,
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
        },
    )

    if completed_keys:
        print(
            f"[behavior-axis-eval] resuming with {len(completed_keys)}/{total_expected} "
            "combinations already complete"
        )
    else:
        print(
            f"[behavior-axis-eval] starting fresh with {total_expected} combinations "
            f"across {len(loaded_datasets)} datasets"
        )

    def update_progress() -> None:
        payload = {
            "state": "running",
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "completed_this_run": completed_now,
            "remaining_combinations": max(0, total_expected - len(completed_keys)),
            "axis_slug": axis_slug,
        }
        write_json(progress_path, payload)
        write_stage_status(run_root, "evaluate_behavior_axis", "running", payload)

    def maybe_print_progress(last_row: dict[str, Any]) -> None:
        if completed_now == 1 or (args.print_every > 0 and completed_now % args.print_every == 0):
            print(
                f"[behavior-axis-eval] {len(completed_keys)}/{total_expected} complete "
                f"({completed_now} new this run); "
                f"{last_row['method']} @ {last_row['layer_spec']} "
                f"{last_row['target_dataset']} AUROC={float(last_row['auroc']):.3f}"
            )

    for layer_spec in layer_specs:
        layer_bundle = axis_bundle["layers"][layer_spec]
        layer_label = str(layer_bundle.get("layer_label", layer_spec))
        for dataset in loaded_datasets:
            layer_scores = scores_for_layer(dataset.eval.features, layer_bundle)
            for method in args.methods:
                metric_key = (method, layer_spec, ZERO_SHOT_SOURCE, dataset.name)
                if metric_key in completed_keys:
                    continue
                score_key, component_index = ZERO_SHOT_METHOD_SPECS[method]
                if component_index is None:
                    honest_like_scores = layer_scores[score_key]
                else:
                    if layer_scores[score_key].shape[1] <= component_index:
                        continue
                    honest_like_scores = layer_scores[score_key][:, component_index]
                metrics, deceptive_scores, probabilities = evaluate_zero_shot(
                    dataset.eval.labels,
                    honest_like_scores,
                )
                row = {
                    "method": method,
                    "layer_spec": layer_spec,
                    "layer_label": layer_label,
                    "source_dataset": ZERO_SHOT_SOURCE,
                    "target_dataset": dataset.name,
                    "train_split": axis_slug,
                    "eval_split": dataset.eval.split_name,
                    "completed_at": utc_now_iso(),
                    **metrics,
                }
                append_csv_row(metrics_path, METRIC_FIELDS, row)
                save_score_rows(
                    score_rows_path,
                    [
                        {
                            "method": method,
                            "layer_spec": layer_spec,
                            "source_dataset": ZERO_SHOT_SOURCE,
                            "target_dataset": dataset.name,
                            "sample_id": sample_id,
                            "label": int(label),
                            "deceptive_score": float(score),
                            "probability": float(probability),
                            "prediction": int(probability > 0.5),
                        }
                        for sample_id, label, score, probability in zip(
                            dataset.eval.sample_ids,
                            dataset.eval.labels,
                            deceptive_scores,
                            probabilities,
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
    heatmap_paths = save_metric_heatmaps(
        results_dir / "plots",
        metric_rows,
        args.methods,
        layer_specs,
        dataset_names,
    )
    grouped_bar_paths = save_zero_shot_grouped_barplots(
        results_dir / "plots",
        metric_rows,
        args.methods,
        layer_specs,
        dataset_names,
    )
    write_json(
        results_dir / "results.json",
        {
            "record_count": len(metric_rows),
            "datasets": dataset_names,
            "methods": args.methods,
            "resolved_pooling": compatibility["activation_pooling"],
            "resolved_layer_number": compatibility["activation_layer_number"],
            "feature_dim": compatibility["feature_dim"],
            "heatmaps": heatmap_paths,
            "grouped_barplots": grouped_bar_paths,
            "axis_slug": axis_slug,
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
            "grouped_barplots": grouped_bar_paths,
            "axis_slug": axis_slug,
        },
    )
    write_stage_status(
        run_root,
        "evaluate_behavior_axis",
        "completed",
        {
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "heatmaps": heatmap_paths,
            "grouped_barplots": grouped_bar_paths,
            "axis_slug": axis_slug,
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
        print(f"[behavior-axis-eval] uploaded to hf://datasets/{args.push_to_hub_repo_id}/{path_in_repo}")

    print(
        f"[behavior-axis-eval] completed {completed_now} new combinations "
        f"({len(completed_keys)}/{total_expected} total); wrote results to {run_root}"
    )


if __name__ == "__main__":
    main()
