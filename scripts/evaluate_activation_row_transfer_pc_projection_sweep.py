#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

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
from deception_honesty_axis.common import ensure_dir, make_run_id, read_json, write_json
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.role_axis_transfer import (
    DATASET_LABELS,
    append_csv_row,
    compute_binary_metrics,
    fit_logistic_scores,
    load_role_axis_bundle,
    save_fit_artifact,
    save_score_rows,
)
from deception_honesty_axis.sycophancy_activations import normalize_activation_pooling


METRIC_FIELDS = [
    "method",
    "selected_pc_count",
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
        description="Train cumulative PC-projected probes across every k prefix of a behavior-specific axis bundle."
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
        help="Optional maximum number of leading PCs to sweep. Defaults to all stored PCs for the matched layer.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit cumulative PC counts to evaluate. Defaults to 1..max_pcs.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id for resume/replay.")
    parser.add_argument(
        "--print-every",
        type=int,
        default=20,
        help="Print a progress update every N newly completed (k, source, target) combinations.",
    )
    parser.add_argument(
        "--push-to-hub-repo-id",
        default=None,
        help="Optional HF dataset repo to upload this sweep run into.",
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
        bundle_name = manifest.get("bundle_name")
        bundle_slug = manifest.get("bundle_slug")
        if bundle_name:
            return normalize_path_slug(str(bundle_name))
        if bundle_slug:
            return normalize_path_slug(str(bundle_slug))
        axis_name = manifest.get("axis_name")
        variant_name = manifest.get("variant_name")
        if axis_name and variant_name:
            return normalize_path_slug(f"{axis_name}-{variant_name}")
        if axis_name:
            return normalize_path_slug(str(axis_name))
    parts = list(axis_bundle_dir.parts)
    if len(parts) >= 2:
        return normalize_path_slug("-".join(parts[-2:]))
    return normalize_path_slug(axis_bundle_dir.name)


def normalize_path_slug(text: str) -> str:
    from deception_honesty_axis.common import slugify

    return slugify(text)


def resolve_k_values(
    available_pc_count: int,
    *,
    max_pcs: int | None,
    explicit_k_values: list[int] | None,
) -> list[int]:
    max_selected_pc_count = available_pc_count if max_pcs is None else min(int(max_pcs), available_pc_count)
    if max_selected_pc_count <= 0:
        raise ValueError(f"Projected transfer sweep requires at least one PC, got max_pcs={max_pcs!r}")

    if explicit_k_values:
        k_values = sorted({int(value) for value in explicit_k_values})
        if any(value <= 0 for value in k_values):
            raise ValueError(f"k_values must be positive, got {explicit_k_values!r}")
        if any(value > max_selected_pc_count for value in k_values):
            raise ValueError(
                f"k_values must be <= selected max pc count {max_selected_pc_count}, got {explicit_k_values!r}"
            )
        return k_values
    return list(range(1, max_selected_pc_count + 1))


def read_completed_sweep_metric_keys(path: Path) -> set[tuple[str, str, int, str, str]]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            (
                row["method"],
                row["layer_spec"],
                int(row["selected_pc_count"]),
                row["source_dataset"],
                row["target_dataset"],
            )
            for row in rows
        }


def load_metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["selected_pc_count"] = int(row["selected_pc_count"])
        for metric_name in ("auroc", "auprc", "balanced_accuracy", "f1", "accuracy", "count"):
            if row.get(metric_name) not in (None, ""):
                row[metric_name] = float(row[metric_name])
    return rows


def write_fit_summary_with_k(path: Path, fit_artifacts_path: Path) -> None:
    fieldnames = [
        "method",
        "selected_pc_count",
        "layer_spec",
        "source_dataset",
        "target_dataset",
        "coef",
        "intercept",
        "coef_l2_norm",
        "coef_dim",
    ]
    ensure_dir(path.parent)
    if not fit_artifacts_path.exists():
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
        return

    rows: list[dict[str, Any]] = []
    with fit_artifacts_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            coef = np.asarray(record.get("coef", []), dtype=np.float32)
            rows.append(
                {
                    "method": record.get("method"),
                    "selected_pc_count": int(record.get("selected_pc_count") or coef.size),
                    "layer_spec": record.get("layer_spec"),
                    "source_dataset": record.get("source_dataset"),
                    "target_dataset": record.get("target_dataset"),
                    "coef": json.dumps([float(value) for value in coef.tolist()]),
                    "intercept": json.dumps(record.get("intercept", [])),
                    "coef_l2_norm": float(np.linalg.norm(coef)) if coef.size else 0.0,
                    "coef_dim": int(coef.size),
                }
            )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_by_k(path: Path, metric_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "selected_pc_count",
        "row_count",
        "mean_auroc",
        "mean_auprc",
        "diag_count",
        "diag_mean_auroc",
        "offdiag_count",
        "offdiag_mean_auroc",
    ]
    summary: dict[int, dict[str, Any]] = {}
    for row in metric_rows:
        k = int(row["selected_pc_count"])
        bucket = summary.setdefault(
            k,
            {
                "selected_pc_count": k,
                "row_count": 0,
                "mean_auroc_sum": 0.0,
                "mean_auprc_sum": 0.0,
                "diag_count": 0,
                "diag_auroc_sum": 0.0,
                "offdiag_count": 0,
                "offdiag_auroc_sum": 0.0,
            },
        )
        bucket["row_count"] += 1
        bucket["mean_auroc_sum"] += float(row["auroc"])
        bucket["mean_auprc_sum"] += float(row["auprc"])
        if row["source_dataset"] == row["target_dataset"]:
            bucket["diag_count"] += 1
            bucket["diag_auroc_sum"] += float(row["auroc"])
        else:
            bucket["offdiag_count"] += 1
            bucket["offdiag_auroc_sum"] += float(row["auroc"])

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for k in sorted(summary):
            bucket = summary[k]
            writer.writerow(
                {
                    "selected_pc_count": k,
                    "row_count": bucket["row_count"],
                    "mean_auroc": bucket["mean_auroc_sum"] / bucket["row_count"] if bucket["row_count"] else None,
                    "mean_auprc": bucket["mean_auprc_sum"] / bucket["row_count"] if bucket["row_count"] else None,
                    "diag_count": bucket["diag_count"],
                    "diag_mean_auroc": (
                        bucket["diag_auroc_sum"] / bucket["diag_count"] if bucket["diag_count"] else None
                    ),
                    "offdiag_count": bucket["offdiag_count"],
                    "offdiag_mean_auroc": (
                        bucket["offdiag_auroc_sum"] / bucket["offdiag_count"] if bucket["offdiag_count"] else None
                    ),
                }
            )


def write_dataset_offdiag_summaries(
    output_dir: Path,
    metric_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    source_path = output_dir / "source_offdiag_summary_by_k.csv"
    target_path = output_dir / "target_offdiag_summary_by_k.csv"
    fieldnames = ["selected_pc_count", "dataset_name", "dataset_label", "pair_count", "offdiag_mean_auroc"]

    source_summary: dict[tuple[int, str], list[float]] = {}
    target_summary: dict[tuple[int, str], list[float]] = {}
    for row in metric_rows:
        if row["source_dataset"] == row["target_dataset"]:
            continue
        k = int(row["selected_pc_count"])
        source_summary.setdefault((k, str(row["source_dataset"])), []).append(float(row["auroc"]))
        target_summary.setdefault((k, str(row["target_dataset"])), []).append(float(row["auroc"]))

    for path, summary in ((source_path, source_summary), (target_path, target_summary)):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for (k, dataset_name), values in sorted(summary.items()):
                writer.writerow(
                    {
                        "selected_pc_count": k,
                        "dataset_name": dataset_name,
                        "dataset_label": DATASET_LABELS.get(dataset_name, dataset_name),
                        "pair_count": len(values),
                        "offdiag_mean_auroc": float(sum(values) / len(values)) if values else None,
                    }
                )
    return source_path, target_path


def save_heatmap(
    output_path: Path,
    matrix: np.ndarray,
    *,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    colorbar_label: str,
    vmin: float,
    vmax: float,
    cmap: str,
    center: float | None = None,
) -> str:
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(max(9, len(col_labels) * 1.7), max(6, len(row_labels) * 0.95)))
    image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    if center is not None:
        from matplotlib.colors import TwoSlopeNorm

        image.set_norm(TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax))

    ax.set_xticks(range(len(col_labels)), col_labels, rotation=35, ha="right", fontsize=11)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=11)
    ax.set_title(title, fontsize=15, fontweight="bold")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = float(matrix[row_index, col_index])
            if np.isnan(value):
                continue
            if center is not None:
                text_color = "white" if abs(value - center) > 0.09 else "black"
            else:
                text_color = "white" if value < (vmin + vmax) / 2.0 else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.02f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=text_color,
            )
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output_path)


def save_projected_heatmaps_by_k(
    output_dir: Path,
    metric_rows: list[dict[str, Any]],
    *,
    layer_spec: str,
    dataset_names: list[str],
    k_values: list[int],
) -> list[str]:
    dataset_labels = [DATASET_LABELS.get(dataset_name, dataset_name) for dataset_name in dataset_names]
    dataset_index = {dataset_name: index for index, dataset_name in enumerate(dataset_names)}
    output_paths: list[str] = []
    for k in k_values:
        rows = [
            row
            for row in metric_rows
            if row["method"] == PROJECTED_TRANSFER_METHOD
            and str(row["layer_spec"]) == str(layer_spec)
            and int(row["selected_pc_count"]) == int(k)
        ]
        if not rows:
            continue
        matrix = np.full((len(dataset_names), len(dataset_names)), np.nan, dtype=np.float32)
        for row in rows:
            row_index = dataset_index[str(row["source_dataset"])]
            col_index = dataset_index[str(row["target_dataset"])]
            matrix[row_index, col_index] = float(row["auroc"])
        output_paths.append(
            save_heatmap(
                output_dir / f"k-{k:03d}__auroc_heatmap.png",
                matrix,
                row_labels=dataset_labels,
                col_labels=dataset_labels,
                title=f"Cumulative Projected AUROC @ {layer_spec} (k={k})",
                colorbar_label="AUROC",
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
            )
        )
    return output_paths


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
    k_values = resolve_k_values(
        available_pc_count,
        max_pcs=args.max_pcs,
        explicit_k_values=args.k_values,
    )
    max_selected_pc_count = max(k_values)

    run_id = args.run_id or make_run_id()
    axis_slug = infer_axis_slug(axis_bundle_dir)
    variant_slug = normalize_path_slug(
        f"{raw_compatibility['activation_pooling'] or config.expected_pooling or 'unknown'}-"
        f"layer-{layer_spec}-cumulative-pcs-1-to-{max_selected_pc_count}"
    )
    run_root = (
        config.artifact_root
        / "runs"
        / "activation-row-transfer-pc-projection-sweep"
        / normalize_path_slug(config.model_name)
        / normalize_path_slug(config.behavior_name)
        / normalize_path_slug(config.dataset_set_name)
        / axis_slug
        / variant_slug
        / run_id
    )
    for relative in ("inputs", "results", "logs", "checkpoints", "meta"):
        ensure_dir(run_root / relative)

    projected_root = ensure_dir(run_root / "results" / "full_projected_splits")
    projected_datasets = [
        project_loaded_dataset(
            dataset,
            layer_bundle=layer_bundle,
            layer_spec=layer_spec,
            layer_label=layer_label,
            max_pcs=max_selected_pc_count,
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
            "experiment_name": "activation-row-transfer-pc-projection-sweep",
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
            "k_values": k_values,
            "max_selected_pc_count": max_selected_pc_count,
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
    completed_keys = read_completed_sweep_metric_keys(metrics_path)
    total_expected = len(projected_datasets) * len(projected_datasets) * len(k_values)
    completed_now = 0

    write_stage_status(
        run_root,
        "evaluate_activation_row_transfer_pc_projection_sweep",
        "running",
        {
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "datasets": dataset_names,
            "axis_slug": axis_slug,
            "k_values": k_values,
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
            "k_values": k_values,
        },
    )

    if completed_keys:
        print(
            f"[activation-row-transfer-pc-projection-sweep] resuming with {len(completed_keys)}/{total_expected} "
            "combinations already complete"
        )
    else:
        print(
            f"[activation-row-transfer-pc-projection-sweep] starting fresh with {total_expected} combinations "
            f"across {len(projected_datasets)} datasets using cumulative PCs {k_values[0]}..{k_values[-1]}/"
            f"{available_pc_count}"
        )

    def update_progress() -> None:
        payload = {
            "state": "running",
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "completed_this_run": completed_now,
            "remaining_combinations": max(0, total_expected - len(completed_keys)),
            "axis_slug": axis_slug,
            "k_values": k_values,
        }
        write_json(progress_path, payload)
        write_stage_status(run_root, "evaluate_activation_row_transfer_pc_projection_sweep", "running", payload)

    def maybe_print_progress(last_row: dict[str, Any]) -> None:
        if completed_now == 1 or (args.print_every > 0 and completed_now % args.print_every == 0):
            print(
                f"[activation-row-transfer-pc-projection-sweep] {len(completed_keys)}/{total_expected} complete "
                f"({completed_now} new this run); "
                f"k={int(last_row['selected_pc_count'])} "
                f"{last_row['source_dataset']} -> {last_row['target_dataset']} "
                f"AUROC={float(last_row['auroc']):.3f}"
            )

    for selected_pc_count in k_values:
        for source_dataset in projected_datasets:
            for target_dataset in projected_datasets:
                metric_key = (
                    PROJECTED_TRANSFER_METHOD,
                    layer_spec,
                    int(selected_pc_count),
                    source_dataset.name,
                    target_dataset.name,
                )
                if metric_key in completed_keys:
                    continue

                fit_artifact, probabilities, predictions = fit_logistic_scores(
                    source_dataset.train.features[:, :selected_pc_count],
                    source_dataset.train.labels,
                    target_dataset.eval.features[:, :selected_pc_count],
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
                    "selected_pc_count": int(selected_pc_count),
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
                        "selected_pc_count": int(selected_pc_count),
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
                        "projected_feature_dim": int(selected_pc_count),
                        **fit_artifact,
                    },
                )
                save_score_rows(
                    score_rows_path,
                    [
                        {
                            "method": PROJECTED_TRANSFER_METHOD,
                            "selected_pc_count": int(selected_pc_count),
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

    metric_rows = load_metric_rows(metrics_path)
    write_summary_by_k(results_dir / "summary_by_k.csv", metric_rows)
    write_fit_summary_with_k(results_dir / "fit_summary.csv", fit_artifacts_path)
    source_summary_path, target_summary_path = write_dataset_offdiag_summaries(results_dir, metric_rows)
    heatmap_paths = save_projected_heatmaps_by_k(
        results_dir / "plots",
        metric_rows,
        layer_spec=layer_spec,
        dataset_names=dataset_names,
        k_values=k_values,
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
            "k_values": k_values,
            "max_selected_pc_count": max_selected_pc_count,
            "selected_pc_indices": projected_compatibility["selected_pc_indices"],
            "axis_bundle_run_dir": str(axis_bundle_dir),
            "axis_slug": axis_slug,
            "projected_splits_dir": str(projected_root),
            "heatmaps": heatmap_paths,
            "summary_by_k_path": str(results_dir / "summary_by_k.csv"),
            "source_offdiag_summary_by_k_path": str(source_summary_path),
            "target_offdiag_summary_by_k_path": str(target_summary_path),
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
            "k_values": k_values,
            "heatmaps": heatmap_paths,
        },
    )
    write_stage_status(
        run_root,
        "evaluate_activation_row_transfer_pc_projection_sweep",
        "completed",
        {
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "axis_slug": axis_slug,
            "k_values": k_values,
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
        )
        print(
            "[activation-row-transfer-pc-projection-sweep] uploaded to "
            f"hf://datasets/{args.push_to_hub_repo_id}/{path_in_repo}"
        )

    print(
        f"[activation-row-transfer-pc-projection-sweep] completed {completed_now} new combinations "
        f"({len(completed_keys)}/{total_expected} total); wrote results to {run_root}"
    )


if __name__ == "__main__":
    main()
