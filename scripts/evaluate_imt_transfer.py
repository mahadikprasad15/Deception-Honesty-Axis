#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deception_honesty_axis.common import append_jsonl, load_jsonl, make_run_id, utc_now_iso
from deception_honesty_axis.imt_config import imt_run_root, load_imt_config
from deception_honesty_axis.imt_recovery import (
    IMT_AXIS_ORDER,
    append_csv_row,
    evaluate_imt_axes_on_split,
    load_imt_axis_bundle,
    read_completed_imt_metric_keys,
    sigmoid_scores,
    single_layer_spec,
)
from deception_honesty_axis.metadata import write_stage_status
from deception_honesty_axis.role_axis_transfer import compute_binary_metrics, load_completion_mean_split


METRIC_FIELDS = [
    "axis_name",
    "axis_variant",
    "layer_spec",
    "layer_label",
    "target_dataset",
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
    parser = argparse.ArgumentParser(description="Evaluate recovered IMT axes on target deception datasets.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the IMT recovery config.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id.")
    parser.add_argument("--print-every", type=int, default=8, help="Print progress every N new combinations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_imt_config(args.config.resolve())
    run_id = args.run_id or make_run_id()
    run_root = imt_run_root(config, run_id)
    bundle_path = run_root / "results" / "axis_bundle.pt"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing axis bundle: {bundle_path}")

    bundle = load_imt_axis_bundle(bundle_path)
    source_mean = np.asarray(bundle["source_mean"], dtype=np.float32)
    raw_axes = np.asarray(bundle["raw_axes"], dtype=np.float32)
    orth_axes = np.asarray(bundle["orth_axes"], dtype=np.float32)
    layer_number = int(bundle["layer_number"])
    spec = single_layer_spec(layer_number)

    results_dir = run_root / "results" / "eval"
    metrics_path = results_dir / "pairwise_metrics.csv"
    score_rows_path = results_dir / "per_example_scores.jsonl"
    projection_rows_path = results_dir / "per_example_projections.jsonl"
    progress_path = run_root / "checkpoints" / "imt_transfer_progress.json"
    completed_keys = read_completed_imt_metric_keys(metrics_path)
    total_expected = len(config.target_datasets) * len(IMT_AXIS_ORDER) * 2
    completed_now = 0

    write_stage_status(
        run_root,
        "evaluate_imt_transfer",
        "running",
        {
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "remaining_combinations": max(0, total_expected - len(completed_keys)),
        },
    )

    split_cache: dict[str, Any] = {}
    projection_written: set[str] = {
        str(row["target_dataset"])
        for row in load_jsonl(projection_rows_path)
        if row.get("target_dataset")
    }

    def update_progress() -> None:
        payload = {
            "state": "running",
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "completed_this_run": completed_now,
            "remaining_combinations": max(0, total_expected - len(completed_keys)),
        }
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        write_stage_status(run_root, "evaluate_imt_transfer", "running", payload)

    def maybe_print(row: dict[str, object]) -> None:
        if completed_now == 1 or (args.print_every > 0 and completed_now % args.print_every == 0):
            print(
                f"[imt-transfer] {len(completed_keys)}/{total_expected} complete "
                f"({completed_now} new this run); {row['axis_name']} {row['axis_variant']} "
                f"-> {row['target_dataset']} AUROC={float(row['auroc']):.3f}"
            )

    for dataset_name in config.target_datasets:
        split_payload = split_cache.get(dataset_name)
        if split_payload is None:
            split_dir = config.target_activations_root / dataset_name / config.eval_split
            print(f"[imt-transfer] loading {dataset_name}/{config.eval_split} from {split_dir}")
            split_payload = load_completion_mean_split(split_dir, [spec])
            split_cache[dataset_name] = split_payload

        projections = evaluate_imt_axes_on_split(
            split_payload,
            source_mean=source_mean,
            raw_axes=raw_axes,
            orth_axes=orth_axes,
        )

        if dataset_name not in projection_written:
            projection_rows = []
            for row_index, sample_id in enumerate(split_payload.sample_ids):
                row = {
                    "target_dataset": dataset_name,
                    "sample_id": sample_id,
                    "label": int(split_payload.labels[row_index]),
                }
                for axis_index, axis_name in enumerate(IMT_AXIS_ORDER):
                    row[f"raw_{axis_name}"] = float(projections["raw"][row_index, axis_index])
                    row[f"orth_{axis_name}"] = float(projections["orth"][row_index, axis_index])
                projection_rows.append(row)
            append_jsonl(projection_rows_path, projection_rows)
            projection_written.add(dataset_name)

        for axis_variant, variant_projections in projections.items():
            for axis_index, axis_name in enumerate(IMT_AXIS_ORDER):
                metric_key = (axis_name, axis_variant, dataset_name)
                if metric_key in completed_keys:
                    continue
                deceptive_scores = variant_projections[:, axis_index]
                predictions = (deceptive_scores > 0.0).astype(np.int64)
                probabilities = sigmoid_scores(deceptive_scores)
                metrics = compute_binary_metrics(
                    split_payload.labels,
                    deceptive_scores,
                    predictions,
                    probabilities=probabilities,
                )
                row = {
                    "axis_name": axis_name,
                    "axis_variant": axis_variant,
                    "layer_spec": spec.key,
                    "layer_label": spec.label,
                    "target_dataset": dataset_name,
                    "eval_split": config.eval_split,
                    "completed_at": utc_now_iso(),
                    **metrics,
                }
                append_csv_row(metrics_path, METRIC_FIELDS, row)
                score_rows = [
                    {
                        "axis_name": axis_name,
                        "axis_variant": axis_variant,
                        "target_dataset": dataset_name,
                        "sample_id": sample_id,
                        "label": int(label),
                        "deceptive_score": float(score),
                        "probability": float(prob),
                    }
                    for sample_id, label, score, prob in zip(
                        split_payload.sample_ids,
                        split_payload.labels,
                        deceptive_scores,
                        probabilities,
                        strict=False,
                    )
                ]
                append_jsonl(score_rows_path, score_rows)
                completed_keys.add(metric_key)
                completed_now += 1
                update_progress()
                maybe_print(row)

    progress_payload = {
        "state": "completed",
        "expected_combinations": total_expected,
        "completed_combinations": len(completed_keys),
        "completed_this_run": completed_now,
        "remaining_combinations": max(0, total_expected - len(completed_keys)),
    }
    progress_path.write_text(json.dumps(progress_payload, indent=2) + "\n", encoding="utf-8")
    write_stage_status(run_root, "evaluate_imt_transfer", "completed", progress_payload)
    print(f"[imt-transfer] wrote metrics under {results_dir}")


if __name__ == "__main__":
    main()
