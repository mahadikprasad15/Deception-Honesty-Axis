#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from deception_honesty_axis.common import make_run_id, read_json, utc_now_iso
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.role_axis_transfer import (
    SUPPORTED_METHODS,
    TRAINED_METHODS,
    ZERO_SHOT_METHODS,
    ZERO_SHOT_SOURCE,
    append_csv_row,
    compute_binary_metrics,
    evaluate_zero_shot,
    fit_logistic_scores,
    load_completion_mean_split,
    load_role_axis_bundle,
    read_completed_metric_keys,
    resolve_layer_specs,
    save_fit_artifact,
    save_metric_heatmaps,
    save_score_rows,
    save_transfer_lineplots,
    scores_for_layer,
    write_fit_summary_csv,
    write_summary_csv,
)
from deception_honesty_axis.transfer_config import load_transfer_config


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate role-derived directions as cross-dataset deception probes.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/probes/role_axis_transfer_v1.json"),
        help="Path to the role-axis transfer config file.",
    )
    parser.add_argument("--axis-bundle-run-dir", type=Path, required=True, help="Role-axis bundle run directory.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id.")
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print a progress update every N newly completed combinations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transfer_config = load_transfer_config(args.config.resolve())
    experiment_config = transfer_config.role_experiment_config
    run_id = args.run_id or make_run_id()
    run_root = (
        experiment_config.artifact_root
        / "runs"
        / "role-axis-transfer"
        / experiment_config.model_slug
        / experiment_config.dataset_slug
        / experiment_config.role_set_slug
        / transfer_config.pooling
        / run_id
    )
    axis_bundle_dir = args.axis_bundle_run_dir.resolve()
    axis_bundle = load_role_axis_bundle(axis_bundle_dir / "results" / "axis_bundle.pt")
    resolved_specs = resolve_layer_specs(axis_bundle["layer_count"], transfer_config.layer_specs)

    invalid_methods = [method for method in transfer_config.methods if method not in SUPPORTED_METHODS]
    if invalid_methods:
        raise ValueError(f"Unsupported methods requested: {invalid_methods}")
    if transfer_config.pooling != "completion_mean":
        raise ValueError("This v1 evaluator only supports completion_mean pooling")

    write_analysis_manifest(
        run_root,
        {
            "config_path": str(transfer_config.path),
            "role_experiment_config_path": str(experiment_config.path),
            "axis_bundle_run_dir": str(axis_bundle_dir),
            "target_activations_root": str(transfer_config.target_activations_root),
            "datasets": transfer_config.target_datasets,
            "methods": transfer_config.methods,
            "layer_specs": transfer_config.layer_specs,
            "source_split": transfer_config.source_split,
            "eval_split": transfer_config.eval_split,
        },
    )

    results_dir = run_root / "results"
    metrics_path = results_dir / "pairwise_metrics.csv"
    score_rows_path = results_dir / "per_example_scores.jsonl"
    fit_artifacts_path = results_dir / "source_fit_artifacts.jsonl"
    progress_path = run_root / "checkpoints" / "progress.json"
    completed_keys = read_completed_metric_keys(metrics_path)

    total_expected = (
        len(ZERO_SHOT_METHODS & set(transfer_config.methods)) * len(transfer_config.target_datasets) * len(resolved_specs)
        + len(TRAINED_METHODS & set(transfer_config.methods)) * len(transfer_config.target_datasets) * len(transfer_config.target_datasets) * len(resolved_specs)
    )
    starting_completed = len(completed_keys)
    run_root.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)
    write_stage_status(
        run_root,
        "evaluate_role_axis_transfer",
        "running",
        {
            "expected_combinations": total_expected,
            "completed_combinations": starting_completed,
            "remaining_combinations": max(0, total_expected - starting_completed),
        },
    )
    progress_path.write_text(
        json.dumps(
            {
                "state": "running",
                "expected_combinations": total_expected,
                "completed_combinations": starting_completed,
                "completed_this_run": 0,
                "remaining_combinations": max(0, total_expected - starting_completed),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if starting_completed:
        print(
            f"[role-axis-transfer] resuming with {starting_completed}/{total_expected} "
            "combinations already complete"
        )
    else:
        print(
            f"[role-axis-transfer] starting fresh with {total_expected} total "
            f"combinations across {len(resolved_specs)} layer specs"
        )

    split_cache: dict[tuple[str, str], Any] = {}
    scores_cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    def get_split(dataset_name: str, split_name: str):  # noqa: ANN202
        key = (dataset_name, split_name)
        cached = split_cache.get(key)
        if cached is not None:
            return cached
        split_dir = transfer_config.target_activations_root / dataset_name / split_name
        print(f"[role-axis-transfer] loading {dataset_name}/{split_name} from {split_dir}")
        cached = load_completion_mean_split(split_dir, resolved_specs)
        split_cache[key] = cached
        return cached

    def get_scored_split(dataset_name: str, split_name: str, spec_key: str):  # noqa: ANN202
        cache_key = (dataset_name, split_name, spec_key)
        cached = scores_cache.get(cache_key)
        if cached is not None:
            return cached
        split_payload = get_split(dataset_name, split_name)
        features = split_payload.features_by_spec[spec_key]
        layer_scores = scores_for_layer(features, axis_bundle["layers"][spec_key])
        cached = {
            "dataset": dataset_name,
            "split": split_name,
            "sample_ids": split_payload.sample_ids,
            "labels": split_payload.labels,
            "contrast": layer_scores["contrast"],
            "pc_scores": layer_scores["pc_scores"],
        }
        scores_cache[cache_key] = cached
        return cached

    completed_now = 0

    def update_progress() -> None:
        progress_payload = {
            "state": "running",
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "completed_this_run": completed_now,
            "remaining_combinations": max(0, total_expected - len(completed_keys)),
        }
        progress_path.write_text(json.dumps(progress_payload, indent=2) + "\n", encoding="utf-8")
        write_stage_status(
            run_root,
            "evaluate_role_axis_transfer",
            "running",
            progress_payload,
        )

    def maybe_print_progress(last_row: dict[str, object]) -> None:
        if completed_now == 1 or (args.print_every > 0 and completed_now % args.print_every == 0):
            print(
                f"[role-axis-transfer] {len(completed_keys)}/{total_expected} complete "
                f"({completed_now} new this run); "
                f"{last_row['method']} @ {last_row['layer_spec']} "
                f"{last_row['source_dataset']} -> {last_row['target_dataset']} "
                f"AUROC={float(last_row['auroc']):.3f}"
            )

    for spec in resolved_specs:
        layer_bundle = axis_bundle["layers"][spec.key]
        for target_dataset in transfer_config.target_datasets:
            target_payload = get_scored_split(target_dataset, transfer_config.eval_split, spec.key)

            if "contrast_zero_shot" in transfer_config.methods:
                metric_key = ("contrast_zero_shot", spec.key, ZERO_SHOT_SOURCE, target_dataset)
                if metric_key not in completed_keys:
                    metrics, deceptive_scores, probabilities = evaluate_zero_shot(
                        target_payload["labels"],
                        target_payload["contrast"],
                    )
                    row = {
                        "method": "contrast_zero_shot",
                        "layer_spec": spec.key,
                        "layer_label": spec.label,
                        "source_dataset": ZERO_SHOT_SOURCE,
                        "target_dataset": target_dataset,
                        "train_split": "",
                        "eval_split": transfer_config.eval_split,
                        "completed_at": utc_now_iso(),
                        **metrics,
                    }
                    append_csv_row(metrics_path, METRIC_FIELDS, row)
                    completed_keys.add(metric_key)
                    completed_now += 1
                    update_progress()
                    maybe_print_progress(row)
                    save_score_rows(
                        score_rows_path,
                        [
                            {
                                "method": "contrast_zero_shot",
                                "layer_spec": spec.key,
                                "source_dataset": ZERO_SHOT_SOURCE,
                                "target_dataset": target_dataset,
                                "sample_id": sample_id,
                                "label": int(label),
                                "deceptive_score": float(score),
                                "probability": float(probability),
                                "prediction": int(probability >= 0.5),
                            }
                            for sample_id, label, score, probability in zip(
                                target_payload["sample_ids"],
                                target_payload["labels"],
                                deceptive_scores,
                                probabilities,
                                strict=False,
                            )
                        ],
                    )

            if "pc1_zero_shot" in transfer_config.methods:
                metric_key = ("pc1_zero_shot", spec.key, ZERO_SHOT_SOURCE, target_dataset)
                if metric_key not in completed_keys:
                    metrics, deceptive_scores, probabilities = evaluate_zero_shot(
                        target_payload["labels"],
                        target_payload["pc_scores"][:, 0],
                    )
                    row = {
                        "method": "pc1_zero_shot",
                        "layer_spec": spec.key,
                        "layer_label": spec.label,
                        "source_dataset": ZERO_SHOT_SOURCE,
                        "target_dataset": target_dataset,
                        "train_split": "",
                        "eval_split": transfer_config.eval_split,
                        "completed_at": utc_now_iso(),
                        **metrics,
                    }
                    append_csv_row(metrics_path, METRIC_FIELDS, row)
                    completed_keys.add(metric_key)
                    completed_now += 1
                    update_progress()
                    maybe_print_progress(row)
                    save_score_rows(
                        score_rows_path,
                        [
                            {
                                "method": "pc1_zero_shot",
                                "layer_spec": spec.key,
                                "source_dataset": ZERO_SHOT_SOURCE,
                                "target_dataset": target_dataset,
                                "sample_id": sample_id,
                                "label": int(label),
                                "deceptive_score": float(score),
                                "probability": float(probability),
                                "prediction": int(probability >= 0.5),
                            }
                            for sample_id, label, score, probability in zip(
                                target_payload["sample_ids"],
                                target_payload["labels"],
                                deceptive_scores,
                                probabilities,
                                strict=False,
                            )
                        ],
                    )

        for source_dataset in transfer_config.target_datasets:
            source_payload = get_scored_split(source_dataset, transfer_config.source_split, spec.key)
            for target_dataset in transfer_config.target_datasets:
                target_payload = get_scored_split(target_dataset, transfer_config.eval_split, spec.key)

                if "contrast_logistic" in transfer_config.methods:
                    metric_key = ("contrast_logistic", spec.key, source_dataset, target_dataset)
                    if metric_key not in completed_keys:
                        fit_artifact, probabilities, predictions = fit_logistic_scores(
                            source_payload["contrast"].reshape(-1, 1),
                            source_payload["labels"],
                            target_payload["contrast"].reshape(-1, 1),
                            random_seed=transfer_config.random_seed,
                            max_iter=transfer_config.logistic_max_iter,
                        )
                        metrics = compute_binary_metrics(
                            target_payload["labels"],
                            probabilities,
                            predictions,
                            probabilities=probabilities,
                        )
                        row = {
                            "method": "contrast_logistic",
                            "layer_spec": spec.key,
                            "layer_label": spec.label,
                            "source_dataset": source_dataset,
                            "target_dataset": target_dataset,
                            "train_split": transfer_config.source_split,
                            "eval_split": transfer_config.eval_split,
                            "completed_at": utc_now_iso(),
                            **metrics,
                        }
                        append_csv_row(metrics_path, METRIC_FIELDS, row)
                        completed_keys.add(metric_key)
                        completed_now += 1
                        update_progress()
                        maybe_print_progress(row)
                        save_fit_artifact(
                            fit_artifacts_path,
                            {
                                "method": "contrast_logistic",
                                "layer_spec": spec.key,
                                "source_dataset": source_dataset,
                                "target_dataset": target_dataset,
                                **fit_artifact,
                            },
                        )
                        save_score_rows(
                            score_rows_path,
                            [
                                {
                                    "method": "contrast_logistic",
                                    "layer_spec": spec.key,
                                    "source_dataset": source_dataset,
                                    "target_dataset": target_dataset,
                                    "sample_id": sample_id,
                                    "label": int(label),
                                    "deceptive_score": float(probability),
                                    "probability": float(probability),
                                    "prediction": int(prediction),
                                }
                                for sample_id, label, probability, prediction in zip(
                                    target_payload["sample_ids"],
                                    target_payload["labels"],
                                    probabilities,
                                    predictions,
                                    strict=False,
                                )
                            ],
                        )

                if "pc123_linear" in transfer_config.methods:
                    metric_key = ("pc123_linear", spec.key, source_dataset, target_dataset)
                    if metric_key not in completed_keys:
                        fit_artifact, probabilities, predictions = fit_logistic_scores(
                            source_payload["pc_scores"],
                            source_payload["labels"],
                            target_payload["pc_scores"],
                            random_seed=transfer_config.random_seed,
                            max_iter=transfer_config.logistic_max_iter,
                        )
                        metrics = compute_binary_metrics(
                            target_payload["labels"],
                            probabilities,
                            predictions,
                            probabilities=probabilities,
                        )
                        row = {
                            "method": "pc123_linear",
                            "layer_spec": spec.key,
                            "layer_label": spec.label,
                            "source_dataset": source_dataset,
                            "target_dataset": target_dataset,
                            "train_split": transfer_config.source_split,
                            "eval_split": transfer_config.eval_split,
                            "completed_at": utc_now_iso(),
                            **metrics,
                        }
                        append_csv_row(metrics_path, METRIC_FIELDS, row)
                        completed_keys.add(metric_key)
                        completed_now += 1
                        update_progress()
                        maybe_print_progress(row)
                        save_fit_artifact(
                            fit_artifacts_path,
                            {
                                "method": "pc123_linear",
                                "layer_spec": spec.key,
                                "source_dataset": source_dataset,
                                "target_dataset": target_dataset,
                                **fit_artifact,
                            },
                        )
                        save_score_rows(
                            score_rows_path,
                            [
                                {
                                    "method": "pc123_linear",
                                    "layer_spec": spec.key,
                                    "source_dataset": source_dataset,
                                    "target_dataset": target_dataset,
                                    "sample_id": sample_id,
                                    "label": int(label),
                                    "deceptive_score": float(probability),
                                    "probability": float(probability),
                                    "prediction": int(prediction),
                                }
                                for sample_id, label, probability, prediction in zip(
                                    target_payload["sample_ids"],
                                    target_payload["labels"],
                                    probabilities,
                                    predictions,
                                    strict=False,
                                )
                            ],
                        )

    metric_rows = []
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            metric_rows = list(__import__("csv").DictReader(handle))
    write_summary_csv(results_dir / "summary_by_method.csv", metric_rows)
    write_fit_summary_csv(results_dir / "fit_summary.csv", fit_artifacts_path)
    heatmap_paths = save_metric_heatmaps(
        results_dir / "plots",
        metric_rows,
        transfer_config.methods,
        [spec.key for spec in resolved_specs],
        transfer_config.target_datasets,
    )
    lineplot_paths = save_transfer_lineplots(
        results_dir / "plots",
        metric_rows,
        [spec.key for spec in resolved_specs],
        transfer_config.target_datasets,
    )
    write_stage_status(
        run_root,
        "evaluate_role_axis_transfer",
        "completed",
        {
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "heatmaps": heatmap_paths,
            "lineplots": lineplot_paths,
        },
    )
    progress_path.write_text(
        json.dumps(
            {
                "state": "completed",
                "expected_combinations": total_expected,
                "completed_combinations": len(completed_keys),
                "completed_this_run": completed_now,
                "remaining_combinations": max(0, total_expected - len(completed_keys)),
                "heatmaps": heatmap_paths,
                "lineplots": lineplot_paths,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[role-axis-transfer] completed {completed_now} new combinations "
        f"({len(completed_keys)}/{total_expected} total); wrote results to {run_root}"
    )


if __name__ == "__main__":
    main()
