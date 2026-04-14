#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
from typing import Any

from deception_honesty_axis.activation_row_targets import (
    ensure_pooling_compatibility,
    ensure_single_layer_compatibility,
    grouped_indices,
    load_activation_rows,
    validate_and_stack_activation_rows,
)
from deception_honesty_axis.common import ensure_dir, make_run_id, read_json, slugify, write_json
from deception_honesty_axis.config import load_config
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
        description="Evaluate a saved role-axis bundle on cached sycophancy output activations."
    )
    parser.add_argument("--axis-bundle-run-dir", type=Path, required=True, help="Run dir containing results/axis_bundle.pt.")
    parser.add_argument("--activation-dataset-repo-id", default=None, help="HF Dataset repo with activation rows.")
    parser.add_argument("--activation-dataset-dir", type=Path, default=None, help="Local datasets.save_to_disk directory.")
    parser.add_argument("--activation-jsonl", type=Path, default=None, help="Local records.jsonl from extraction run.")
    parser.add_argument("--split", default="train", help="HF Dataset split to load when using --activation-dataset-repo-id.")
    parser.add_argument(
        "--group-field",
        default="dataset_source",
        help="Activation-row field used to define evaluation groups. 'all' is always added automatically.",
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"), help="Repo artifact root.")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-3B-Instruct", help="Model slug for run path.")
    parser.add_argument("--axis-name", default="sycophancy-pilot-v1", help="Axis slug for run path.")
    parser.add_argument("--target-name", default=None, help="Optional target identifier used in artifact naming and manifests.")
    parser.add_argument("--run-id", default=None, help="Optional fixed run id.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["contrast_zero_shot", "pc1_zero_shot", "pc2_zero_shot", "pc3_zero_shot"],
        help="Zero-shot methods to evaluate.",
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
    role_experiment_config_path = read_json(manifest_path).get("role_experiment_config_path")
    if not role_experiment_config_path:
        return None
    config_path = Path(str(role_experiment_config_path))
    if not config_path.exists():
        return None
    return str(load_config(config_path).analysis.get("pooling") or "")


def infer_target_name(args: argparse.Namespace, source_manifest: dict[str, Any], dataset_names: list[str]) -> str | None:
    if args.target_name:
        return str(args.target_name)
    repo_id = source_manifest.get("repo_id")
    split = source_manifest.get("split")
    if repo_id and split:
        return f"{str(repo_id).split('/')[-1]}-{split}"
    named_datasets = sorted(name for name in dataset_names if name != "all")
    if len(named_datasets) == 1:
        return named_datasets[0]
    return None


def main() -> None:
    args = parse_args()
    invalid_methods = [method for method in args.methods if method not in ZERO_SHOT_METHODS]
    if invalid_methods:
        raise ValueError(f"Only zero-shot methods are supported here; invalid={invalid_methods}")

    axis_bundle_dir = args.axis_bundle_run_dir.resolve()
    axis_bundle_path = axis_bundle_dir / "results" / "axis_bundle.pt"
    axis_bundle = load_role_axis_bundle(axis_bundle_path)
    layer_specs = list(axis_bundle["layers"].keys())
    layer_labels = {
        key: str(payload.get("layer_label", key))
        for key, payload in axis_bundle["layers"].items()
    }
    expected_pooling = axis_expected_pooling(axis_bundle_dir)

    rows, source_manifest = load_activation_rows(
        activation_dataset_repo_id=args.activation_dataset_repo_id,
        activation_dataset_dir=args.activation_dataset_dir,
        activation_jsonl=args.activation_jsonl,
        split=args.split,
    )
    stacked = validate_and_stack_activation_rows(
        rows,
        group_field=args.group_field,
        source_manifest=source_manifest,
    )
    ensure_pooling_compatibility(
        activation_pooling=stacked.activation_pooling,
        expected_pooling=expected_pooling,
        context="Sycophancy activation rows",
    )
    compatible_layer_specs: list[str] = []
    skipped_layer_specs: list[str] = []
    for layer_spec in layer_specs:
        layer_numbers = [int(value) for value in axis_bundle["layers"][layer_spec].get("layer_numbers", [])]
        try:
            ensure_single_layer_compatibility(
                activation_layer_number=stacked.activation_layer_number,
                expected_layer_numbers=layer_numbers,
                context="Sycophancy activation rows",
            )
        except ValueError:
            skipped_layer_specs.append(layer_spec)
            continue
        compatible_layer_specs.append(layer_spec)
    if not compatible_layer_specs:
        raise ValueError(
            "No axis layer specs are compatible with the cached activation rows; "
            f"activation_layer_number={stacked.activation_layer_number!r}, bundle_layers={layer_specs!r}"
        )
    layer_specs = compatible_layer_specs
    layer_labels = {key: layer_labels[key] for key in layer_specs}

    features = stacked.features
    labels = stacked.labels
    sample_ids = stacked.sample_ids
    dataset_sources = stacked.group_values
    groups = grouped_indices(dataset_sources)
    dataset_names = list(groups.keys())
    target_name = infer_target_name(args, source_manifest, dataset_names)

    run_id = args.run_id or make_run_id()
    run_path = (
        args.artifact_root
        / "runs"
        / "sycophancy-activation-probe"
        / slugify(args.model_name)
        / slugify(args.axis_name)
    )
    if target_name:
        run_path = run_path / slugify(target_name)
    run_root = ensure_dir(run_path / run_id)
    for relative in ("inputs", "results", "logs", "checkpoints", "meta"):
        ensure_dir(run_root / relative)

    write_analysis_manifest(
        run_root,
        {
            "run_id": run_id,
            "target_name": target_name,
            "axis_bundle_run_dir": str(axis_bundle_dir),
            "axis_bundle_path": str(axis_bundle_path),
            "activation_source": source_manifest,
            "activation_group_field": args.group_field,
            "activation_pooling": stacked.activation_pooling,
            "activation_layer_number": stacked.activation_layer_number,
            "axis_expected_pooling": expected_pooling,
            "methods": args.methods,
            "layer_specs": layer_specs,
            "skipped_incompatible_layer_specs": skipped_layer_specs,
            "record_count": int(labels.shape[0]),
            "label_counts": dict(Counter(int(value) for value in labels.tolist())),
            "dataset_source_counts": dict(Counter(dataset_sources)),
        },
    )
    write_json(run_root / "inputs" / "activation_source.json", source_manifest)
    write_stage_status(
        run_root,
        "evaluate_sycophancy_activation_probe",
        "running",
        {"record_count": int(labels.shape[0]), "datasets": dataset_names, "methods": args.methods, "target_name": target_name},
    )

    metrics_path = run_root / "results" / "pairwise_metrics.csv"
    score_rows_path = run_root / "results" / "per_example_scores.jsonl"
    completed_keys = read_completed_metric_keys(metrics_path)
    total_expected = len(layer_specs) * len(dataset_names) * len(args.methods)
    completed_now = 0

    write_json(
        run_root / "checkpoints" / "progress.json",
        {
            "state": "running",
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
        },
    )

    for layer_spec in layer_specs:
        layer_bundle = axis_bundle["layers"][layer_spec]
        layer_scores = scores_for_layer(features, layer_bundle)
        for dataset_name, indices in groups.items():
            group_labels = labels[indices]
            group_sample_ids = [sample_ids[index] for index in indices.tolist()]
            for method in args.methods:
                metric_key = (method, layer_spec, ZERO_SHOT_SOURCE, dataset_name)
                if metric_key in completed_keys:
                    continue
                score_key, component_index = ZERO_SHOT_METHOD_SPECS[method]
                if component_index is None:
                    honest_like_scores = layer_scores[score_key][indices]
                else:
                    if layer_scores[score_key].shape[1] <= component_index:
                        continue
                    honest_like_scores = layer_scores[score_key][indices, component_index]

                metrics, deceptive_scores, probabilities = evaluate_zero_shot(group_labels, honest_like_scores)
                row = {
                    "method": method,
                    "layer_spec": layer_spec,
                    "layer_label": layer_labels[layer_spec],
                    "source_dataset": ZERO_SHOT_SOURCE,
                    "target_dataset": dataset_name,
                    "train_split": "",
                    "eval_split": "all",
                    "completed_at": utc_now_iso(),
                    **metrics,
                }
                append_csv_row(metrics_path, METRIC_FIELDS, row)
                completed_keys.add(metric_key)
                completed_now += 1
                save_score_rows(
                    score_rows_path,
                    [
                        {
                            "method": method,
                            "layer_spec": layer_spec,
                            "source_dataset": ZERO_SHOT_SOURCE,
                            "target_dataset": dataset_name,
                            "sample_id": sample_id,
                            "label": int(label),
                            "deceptive_score": float(score),
                            "probability": float(probability),
                            "prediction": int(probability >= 0.5),
                        }
                        for sample_id, label, score, probability in zip(
                            group_sample_ids,
                            group_labels,
                            deceptive_scores,
                            probabilities,
                            strict=False,
                        )
                    ],
                )
                print(
                    f"[sycophancy-probe] {len(completed_keys)}/{total_expected} "
                    f"{method} @ {layer_spec} -> {dataset_name} AUROC={metrics['auroc']:.3f}"
                )
                write_json(
                    run_root / "checkpoints" / "progress.json",
                    {
                        "state": "running",
                        "expected_combinations": total_expected,
                        "completed_combinations": len(completed_keys),
                        "completed_this_run": completed_now,
                    },
                )

    import csv

    metric_rows = []
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    write_summary_csv(run_root / "results" / "summary_by_method.csv", metric_rows)
    heatmap_paths = save_metric_heatmaps(
        run_root / "results" / "plots",
        metric_rows,
        args.methods,
        layer_specs,
        dataset_names,
    )
    grouped_bar_paths = save_zero_shot_grouped_barplots(
        run_root / "results" / "plots",
        metric_rows,
        args.methods,
        layer_specs,
        dataset_names,
    )
    write_json(
        run_root / "checkpoints" / "progress.json",
        {
            "state": "completed",
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "completed_this_run": completed_now,
            "heatmaps": heatmap_paths,
            "grouped_bar_plots": grouped_bar_paths,
        },
    )
    write_stage_status(
        run_root,
        "evaluate_sycophancy_activation_probe",
        "completed",
        {
            "target_name": target_name,
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "heatmaps": heatmap_paths,
            "grouped_bar_plots": grouped_bar_paths,
        },
    )

    if args.push_to_hub_repo_id:
        from huggingface_hub import upload_folder

        default_path_in_repo = str(run_root.relative_to(args.artifact_root))
        path_in_repo = args.path_in_repo or default_path_in_repo
        upload_folder(
            repo_id=args.push_to_hub_repo_id,
            repo_type="dataset",
            folder_path=str(run_root),
            path_in_repo=path_in_repo,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"[sycophancy-probe] uploaded to hf://datasets/{args.push_to_hub_repo_id}/{path_in_repo}")

    print(f"[sycophancy-probe] completed run_id={run_id} root={run_root}")


if __name__ == "__main__":
    main()
