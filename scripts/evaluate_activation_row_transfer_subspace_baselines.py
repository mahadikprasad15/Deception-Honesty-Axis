#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from deception_honesty_axis.activation_row_transfer import (
    LoadedActivationRowDataset,
    ensure_loaded_dataset_compatibility,
    load_activation_row_dataset,
)
from deception_honesty_axis.activation_row_transfer_config import load_activation_row_transfer_config
from deception_honesty_axis.common import ensure_dir, make_run_id, slugify, write_json
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.role_axis_transfer import (
    DATASET_LABELS,
    append_csv_row,
    compute_binary_metrics,
    fit_logistic_scores,
    save_fit_artifact,
    save_score_rows,
)


DATASET_PCA_METHOD = "dataset_pca_logistic"
RANDOM_SUBSPACE_METHOD = "random_orthonormal_logistic"
SUPPORTED_METHODS = {DATASET_PCA_METHOD, RANDOM_SUBSPACE_METHOD}

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
    "seed_count",
    "random_seeds",
    "auroc",
    "auroc_std",
    "auprc",
    "auprc_std",
    "balanced_accuracy",
    "balanced_accuracy_std",
    "f1",
    "f1_std",
    "accuracy",
    "accuracy_std",
    "count",
    "completed_at",
]

SEED_METRIC_FIELDS = [
    "method",
    "selected_pc_count",
    "seed",
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


@dataclass(frozen=True)
class SourceSubspace:
    method: str
    selected_pc_count: int
    seed: int | None
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: list[float] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate source-dataset PCA and random orthonormal subspace baselines for activation-row transfer."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Activation-row transfer config JSON.")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=[DATASET_PCA_METHOD, RANDOM_SUBSPACE_METHOD],
        help=f"Baseline methods to run. Supported: {sorted(SUPPORTED_METHODS)}",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="*",
        default=[1, 2, 3],
        help="Subspace dimensions to evaluate. Defaults to 1 2 3.",
    )
    parser.add_argument(
        "--random-seeds",
        type=int,
        nargs="*",
        default=[11, 23, 37, 41, 53],
        help="Seeds for the random orthonormal baseline.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id for resume/replay.")
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print a progress update every N newly completed method/k/source/target combinations.",
    )
    parser.add_argument(
        "--push-to-hub-repo-id",
        default=None,
        help="Optional HF dataset repo to upload this baseline run into.",
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


def validate_methods(methods: list[str]) -> list[str]:
    resolved = [str(method) for method in methods]
    invalid = sorted(set(resolved) - SUPPORTED_METHODS)
    if invalid:
        raise ValueError(f"Unsupported subspace baseline methods: {invalid}; supported={sorted(SUPPORTED_METHODS)}")
    return resolved


def validate_k_values(k_values: list[int], feature_dim: int, train_counts: list[int]) -> list[int]:
    resolved = sorted({int(value) for value in k_values})
    if not resolved or any(value <= 0 for value in resolved):
        raise ValueError(f"k-values must be positive integers, got {k_values!r}")
    max_possible = min(int(feature_dim), min(int(count) for count in train_counts))
    if max(resolved) > max_possible:
        raise ValueError(
            f"Requested k={max(resolved)} but PCA/random subspaces can use at most {max_possible} dimensions "
            f"for feature_dim={feature_dim} and train_counts={train_counts!r}"
        )
    return resolved


def read_completed_subspace_metric_keys(path: Path) -> set[tuple[str, str, int, str, str]]:
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
        for key in (
            "auroc",
            "auroc_std",
            "auprc",
            "auprc_std",
            "balanced_accuracy",
            "balanced_accuracy_std",
            "f1",
            "f1_std",
            "accuracy",
            "accuracy_std",
            "count",
        ):
            if row.get(key) not in (None, ""):
                row[key] = float(row[key])
    return rows


def fit_source_pca_subspaces(source_dataset: LoadedActivationRowDataset, max_k: int) -> dict[int, SourceSubspace]:
    from sklearn.decomposition import PCA

    features = source_dataset.train.features.astype(np.float32, copy=False)
    pca = PCA(n_components=max_k, svd_solver="full")
    pca.fit(features)
    mean = pca.mean_.astype(np.float32, copy=False)
    components = pca.components_.astype(np.float32, copy=False)
    explained = [float(value) for value in pca.explained_variance_ratio_.tolist()]
    return {
        k: SourceSubspace(
            method=DATASET_PCA_METHOD,
            selected_pc_count=k,
            seed=None,
            mean=mean,
            components=components[:k].copy(),
            explained_variance_ratio=explained[:k],
        )
        for k in range(1, max_k + 1)
    }


def random_orthonormal_subspace(feature_dim: int, max_k: int, *, seed: int) -> SourceSubspace:
    rng = np.random.default_rng(int(seed))
    matrix = rng.normal(size=(feature_dim, max_k)).astype(np.float32)
    q, _ = np.linalg.qr(matrix, mode="reduced")
    return SourceSubspace(
        method=RANDOM_SUBSPACE_METHOD,
        selected_pc_count=max_k,
        seed=int(seed),
        mean=np.zeros(feature_dim, dtype=np.float32),
        components=q.T.astype(np.float32, copy=False),
        explained_variance_ratio=None,
    )


def project_features(features: np.ndarray, subspace: SourceSubspace) -> np.ndarray:
    centered = features.astype(np.float32, copy=False) - subspace.mean.reshape(1, -1)
    return centered @ subspace.components.T


def aggregate_metric_records(seed_records: list[dict[str, Any]]) -> dict[str, float]:
    if not seed_records:
        raise ValueError("Cannot aggregate empty metric records")
    payload: dict[str, float] = {}
    for metric_name in ("auroc", "auprc", "balanced_accuracy", "f1", "accuracy"):
        values = np.asarray([float(row[metric_name]) for row in seed_records], dtype=np.float32)
        payload[metric_name] = float(np.mean(values))
        payload[f"{metric_name}_std"] = float(np.std(values, ddof=0))
    payload["count"] = float(seed_records[0]["count"])
    return payload


def write_summary_by_method_k(path: Path, metric_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "method",
        "selected_pc_count",
        "row_count",
        "mean_auroc",
        "mean_auprc",
        "diag_count",
        "diag_mean_auroc",
        "offdiag_count",
        "offdiag_mean_auroc",
    ]
    summary: dict[tuple[str, int], dict[str, Any]] = {}
    for row in metric_rows:
        key = (str(row["method"]), int(row["selected_pc_count"]))
        bucket = summary.setdefault(
            key,
            {
                "method": key[0],
                "selected_pc_count": key[1],
                "row_count": 0,
                "auroc_sum": 0.0,
                "auprc_sum": 0.0,
                "diag_count": 0,
                "diag_auroc_sum": 0.0,
                "offdiag_count": 0,
                "offdiag_auroc_sum": 0.0,
            },
        )
        bucket["row_count"] += 1
        bucket["auroc_sum"] += float(row["auroc"])
        bucket["auprc_sum"] += float(row["auprc"])
        if row["source_dataset"] == row["target_dataset"]:
            bucket["diag_count"] += 1
            bucket["diag_auroc_sum"] += float(row["auroc"])
        else:
            bucket["offdiag_count"] += 1
            bucket["offdiag_auroc_sum"] += float(row["auroc"])

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(summary):
            bucket = summary[key]
            writer.writerow(
                {
                    "method": bucket["method"],
                    "selected_pc_count": bucket["selected_pc_count"],
                    "row_count": bucket["row_count"],
                    "mean_auroc": bucket["auroc_sum"] / bucket["row_count"] if bucket["row_count"] else None,
                    "mean_auprc": bucket["auprc_sum"] / bucket["row_count"] if bucket["row_count"] else None,
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


def write_fit_summary_with_k(path: Path, fit_artifacts_path: Path) -> None:
    fieldnames = [
        "method",
        "selected_pc_count",
        "seed",
        "layer_spec",
        "source_dataset",
        "target_dataset",
        "coef",
        "intercept",
        "coef_l2_norm",
        "coef_dim",
        "explained_variance_ratio",
    ]
    ensure_dir(path.parent)
    rows: list[dict[str, Any]] = []
    if fit_artifacts_path.exists():
        with fit_artifacts_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                coef = np.asarray(record.get("coef", []), dtype=np.float32)
                rows.append(
                    {
                        "method": record.get("method"),
                        "selected_pc_count": record.get("selected_pc_count"),
                        "seed": record.get("seed"),
                        "layer_spec": record.get("layer_spec"),
                        "source_dataset": record.get("source_dataset"),
                        "target_dataset": record.get("target_dataset"),
                        "coef": json.dumps([float(value) for value in coef.tolist()]),
                        "intercept": json.dumps(record.get("intercept", [])),
                        "coef_l2_norm": float(np.linalg.norm(coef)) if coef.size else 0.0,
                        "coef_dim": int(coef.size),
                        "explained_variance_ratio": json.dumps(record.get("explained_variance_ratio")),
                    }
                )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_heatmap(
    output_path: Path,
    matrix: np.ndarray,
    *,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    colorbar_label: str,
) -> str:
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(max(9, len(col_labels) * 1.7), max(6, len(row_labels) * 0.95)))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=35, ha="right", fontsize=11)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=11)
    ax.set_title(title, fontsize=15, fontweight="bold")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = float(matrix[row_index, col_index])
            if np.isnan(value):
                continue
            ax.text(
                col_index,
                row_index,
                f"{value:.02f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white" if value < 0.55 else "black",
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


def save_baseline_heatmaps_by_method_k(
    output_dir: Path,
    metric_rows: list[dict[str, Any]],
    *,
    layer_spec: str,
    dataset_names: list[str],
    methods: list[str],
    k_values: list[int],
) -> list[str]:
    dataset_labels = [DATASET_LABELS.get(dataset_name, dataset_name) for dataset_name in dataset_names]
    dataset_index = {dataset_name: index for index, dataset_name in enumerate(dataset_names)}
    output_paths: list[str] = []
    for method in methods:
        for k in k_values:
            rows = [
                row
                for row in metric_rows
                if row["method"] == method
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
                    output_dir / f"{method}__k-{k:03d}__auroc_heatmap.png",
                    matrix,
                    row_labels=dataset_labels,
                    col_labels=dataset_labels,
                    title=f"{method.replace('_', ' ').title()} AUROC @ {layer_spec} (k={k})",
                    colorbar_label="AUROC",
                )
            )
    return output_paths


def run_logistic_on_subspace(
    *,
    source_dataset: LoadedActivationRowDataset,
    target_dataset: LoadedActivationRowDataset,
    subspace: SourceSubspace,
    random_seed: int,
    max_iter: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    train_projected = project_features(source_dataset.train.features, subspace)
    eval_projected = project_features(target_dataset.eval.features, subspace)
    return fit_logistic_scores(
        train_projected,
        source_dataset.train.labels,
        eval_projected,
        random_seed=random_seed,
        max_iter=max_iter,
    )


def main() -> None:
    args = parse_args()
    methods = validate_methods(args.methods)
    if not args.random_seeds:
        raise ValueError("--random-seeds must contain at least one seed")

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
    resolved_pooling = compatibility["activation_pooling"] or config.expected_pooling or "unknown"
    resolved_layer_number = compatibility["activation_layer_number"]
    if resolved_layer_number is None and config.expected_layer_number is not None:
        resolved_layer_number = config.expected_layer_number
    layer_spec = str(resolved_layer_number) if resolved_layer_number is not None else "unknown"
    layer_label = f"L{resolved_layer_number}" if resolved_layer_number is not None else "Unknown"

    k_values = validate_k_values(
        args.k_values,
        int(compatibility["feature_dim"]),
        [int(dataset.train.labels.shape[0]) for dataset in loaded_datasets],
    )
    max_k = max(k_values)
    random_seeds = [int(seed) for seed in args.random_seeds]

    run_id = args.run_id or make_run_id()
    variant_slug = slugify(
        f"{resolved_pooling}-layer-{layer_spec}-k-{'-'.join(str(value) for value in k_values)}"
    )
    run_root = (
        config.artifact_root
        / "runs"
        / "activation-row-transfer-subspace-baselines"
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
            "experiment_name": "activation-row-transfer-subspace-baselines",
            "behavior_name": config.behavior_name,
            "model_name": config.model_name,
            "dataset_set_name": config.dataset_set_name,
            "methods": methods,
            "k_values": k_values,
            "random_seeds": random_seeds,
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
    seed_metrics_path = results_dir / "per_seed_pairwise_metrics.csv"
    score_rows_path = results_dir / "per_example_scores.jsonl"
    fit_artifacts_path = results_dir / "source_fit_artifacts.jsonl"
    progress_path = run_root / "checkpoints" / "progress.json"
    completed_keys = read_completed_subspace_metric_keys(metrics_path)
    total_expected = len(methods) * len(k_values) * len(loaded_datasets) * len(loaded_datasets)
    completed_now = 0

    write_stage_status(
        run_root,
        "evaluate_activation_row_transfer_subspace_baselines",
        "running",
        {
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "datasets": dataset_names,
            "methods": methods,
            "k_values": k_values,
            "random_seeds": random_seeds,
        },
    )
    write_json(
        progress_path,
        {
            "state": "running",
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "completed_this_run": 0,
            "methods": methods,
            "k_values": k_values,
            "random_seeds": random_seeds,
        },
    )

    if completed_keys:
        print(
            "[activation-row-transfer-subspace-baselines] resuming with "
            f"{len(completed_keys)}/{total_expected} combinations already complete"
        )
    else:
        print(
            "[activation-row-transfer-subspace-baselines] starting fresh with "
            f"{total_expected} combinations across {len(loaded_datasets)} datasets"
        )

    source_pca_cache: dict[str, dict[int, SourceSubspace]] = {}
    random_cache: dict[tuple[int, int], SourceSubspace] = {}

    def update_progress() -> None:
        payload = {
            "state": "running",
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "completed_this_run": completed_now,
            "remaining_combinations": max(0, total_expected - len(completed_keys)),
            "methods": methods,
            "k_values": k_values,
            "random_seeds": random_seeds,
        }
        write_json(progress_path, payload)
        write_stage_status(run_root, "evaluate_activation_row_transfer_subspace_baselines", "running", payload)

    def maybe_print_progress(last_row: dict[str, Any]) -> None:
        if completed_now == 1 or (args.print_every > 0 and completed_now % args.print_every == 0):
            print(
                "[activation-row-transfer-subspace-baselines] "
                f"{len(completed_keys)}/{total_expected} complete ({completed_now} new this run); "
                f"{last_row['method']} k={int(last_row['selected_pc_count'])} "
                f"{last_row['source_dataset']} -> {last_row['target_dataset']} "
                f"AUROC={float(last_row['auroc']):.3f}"
            )

    for source_dataset in loaded_datasets:
        if DATASET_PCA_METHOD in methods:
            source_pca_cache[source_dataset.name] = fit_source_pca_subspaces(source_dataset, max_k)
        for selected_pc_count in k_values:
            for target_dataset in loaded_datasets:
                if DATASET_PCA_METHOD in methods:
                    metric_key = (
                        DATASET_PCA_METHOD,
                        layer_spec,
                        int(selected_pc_count),
                        source_dataset.name,
                        target_dataset.name,
                    )
                    if metric_key not in completed_keys:
                        subspace = source_pca_cache[source_dataset.name][selected_pc_count]
                        fit_artifact, probabilities, predictions = run_logistic_on_subspace(
                            source_dataset=source_dataset,
                            target_dataset=target_dataset,
                            subspace=subspace,
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
                            "method": DATASET_PCA_METHOD,
                            "selected_pc_count": int(selected_pc_count),
                            "layer_spec": layer_spec,
                            "layer_label": layer_label,
                            "source_dataset": source_dataset.name,
                            "target_dataset": target_dataset.name,
                            "train_split": source_dataset.train.source_manifest.get(
                                "split", source_dataset.train.split_name
                            ),
                            "eval_split": target_dataset.eval.source_manifest.get(
                                "split", target_dataset.eval.split_name
                            ),
                            "train_count": int(source_dataset.train.labels.shape[0]),
                            "seed_count": 0,
                            "random_seeds": "",
                            "completed_at": utc_now_iso(),
                            **metrics,
                            "auroc_std": "",
                            "auprc_std": "",
                            "balanced_accuracy_std": "",
                            "f1_std": "",
                            "accuracy_std": "",
                        }
                        append_csv_row(metrics_path, METRIC_FIELDS, row)
                        save_fit_artifact(
                            fit_artifacts_path,
                            {
                                "method": DATASET_PCA_METHOD,
                                "selected_pc_count": int(selected_pc_count),
                                "seed": None,
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
                                "subspace_mean": subspace.mean.tolist(),
                                "subspace_components": subspace.components.tolist(),
                                "explained_variance_ratio": subspace.explained_variance_ratio,
                                **fit_artifact,
                            },
                        )
                        save_score_rows(
                            score_rows_path,
                            [
                                {
                                    "method": DATASET_PCA_METHOD,
                                    "selected_pc_count": int(selected_pc_count),
                                    "seed": None,
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

                if RANDOM_SUBSPACE_METHOD in methods:
                    metric_key = (
                        RANDOM_SUBSPACE_METHOD,
                        layer_spec,
                        int(selected_pc_count),
                        source_dataset.name,
                        target_dataset.name,
                    )
                    if metric_key in completed_keys:
                        continue

                    seed_records: list[dict[str, Any]] = []
                    seed_score_rows: list[dict[str, Any]] = []
                    for seed in random_seeds:
                        max_seed_key = (int(seed), max_k)
                        if max_seed_key not in random_cache:
                            random_cache[max_seed_key] = random_orthonormal_subspace(
                                int(compatibility["feature_dim"]),
                                max_k,
                                seed=seed,
                            )
                        max_subspace = random_cache[max_seed_key]
                        subspace = SourceSubspace(
                            method=RANDOM_SUBSPACE_METHOD,
                            selected_pc_count=int(selected_pc_count),
                            seed=int(seed),
                            mean=max_subspace.mean,
                            components=max_subspace.components[:selected_pc_count].copy(),
                            explained_variance_ratio=None,
                        )
                        fit_artifact, probabilities, predictions = run_logistic_on_subspace(
                            source_dataset=source_dataset,
                            target_dataset=target_dataset,
                            subspace=subspace,
                            random_seed=config.random_seed,
                            max_iter=config.logistic_max_iter,
                        )
                        seed_metrics = compute_binary_metrics(
                            target_dataset.eval.labels,
                            probabilities,
                            predictions,
                            probabilities=probabilities,
                        )
                        seed_row = {
                            "method": RANDOM_SUBSPACE_METHOD,
                            "selected_pc_count": int(selected_pc_count),
                            "seed": int(seed),
                            "layer_spec": layer_spec,
                            "layer_label": layer_label,
                            "source_dataset": source_dataset.name,
                            "target_dataset": target_dataset.name,
                            "train_split": source_dataset.train.source_manifest.get(
                                "split", source_dataset.train.split_name
                            ),
                            "eval_split": target_dataset.eval.source_manifest.get(
                                "split", target_dataset.eval.split_name
                            ),
                            "train_count": int(source_dataset.train.labels.shape[0]),
                            "completed_at": utc_now_iso(),
                            **seed_metrics,
                        }
                        append_csv_row(seed_metrics_path, SEED_METRIC_FIELDS, seed_row)
                        save_fit_artifact(
                            fit_artifacts_path,
                            {
                                "method": RANDOM_SUBSPACE_METHOD,
                                "selected_pc_count": int(selected_pc_count),
                                "seed": int(seed),
                                "layer_spec": layer_spec,
                                "layer_label": layer_label,
                                "source_dataset": source_dataset.name,
                                "target_dataset": target_dataset.name,
                                "train_split": seed_row["train_split"],
                                "eval_split": seed_row["eval_split"],
                                "train_count": int(source_dataset.train.labels.shape[0]),
                                "eval_count": int(target_dataset.eval.labels.shape[0]),
                                "train_label_counts": source_dataset.train.summary()["label_counts"],
                                "eval_label_counts": target_dataset.eval.summary()["label_counts"],
                                "subspace_mean": subspace.mean.tolist(),
                                "subspace_components": subspace.components.tolist(),
                                "explained_variance_ratio": None,
                                **fit_artifact,
                            },
                        )
                        seed_records.append(seed_row)
                        seed_score_rows.extend(
                            [
                                {
                                    "method": RANDOM_SUBSPACE_METHOD,
                                    "selected_pc_count": int(selected_pc_count),
                                    "seed": int(seed),
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
                            ]
                        )

                    aggregated = aggregate_metric_records(seed_records)
                    row = {
                        "method": RANDOM_SUBSPACE_METHOD,
                        "selected_pc_count": int(selected_pc_count),
                        "layer_spec": layer_spec,
                        "layer_label": layer_label,
                        "source_dataset": source_dataset.name,
                        "target_dataset": target_dataset.name,
                        "train_split": source_dataset.train.source_manifest.get("split", source_dataset.train.split_name),
                        "eval_split": target_dataset.eval.source_manifest.get("split", target_dataset.eval.split_name),
                        "train_count": int(source_dataset.train.labels.shape[0]),
                        "seed_count": len(random_seeds),
                        "random_seeds": json.dumps(random_seeds),
                        "completed_at": utc_now_iso(),
                        **aggregated,
                    }
                    append_csv_row(metrics_path, METRIC_FIELDS, row)
                    save_score_rows(score_rows_path, seed_score_rows)
                    completed_keys.add(metric_key)
                    completed_now += 1
                    update_progress()
                    maybe_print_progress(row)

    metric_rows = load_metric_rows(metrics_path)
    write_summary_by_method_k(results_dir / "summary_by_method_k.csv", metric_rows)
    write_fit_summary_with_k(results_dir / "fit_summary.csv", fit_artifacts_path)
    heatmap_paths = save_baseline_heatmaps_by_method_k(
        results_dir / "plots",
        metric_rows,
        layer_spec=layer_spec,
        dataset_names=dataset_names,
        methods=methods,
        k_values=k_values,
    )
    write_json(
        results_dir / "results.json",
        {
            "record_count": len(metric_rows),
            "datasets": dataset_names,
            "methods": methods,
            "k_values": k_values,
            "random_seeds": random_seeds,
            "resolved_pooling": resolved_pooling,
            "resolved_layer_number": resolved_layer_number,
            "feature_dim": compatibility["feature_dim"],
            "heatmaps": heatmap_paths,
            "summary_by_method_k_path": str(results_dir / "summary_by_method_k.csv"),
            "per_seed_pairwise_metrics_path": str(seed_metrics_path) if seed_metrics_path.exists() else None,
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
            "methods": methods,
            "k_values": k_values,
            "random_seeds": random_seeds,
            "heatmaps": heatmap_paths,
        },
    )
    write_stage_status(
        run_root,
        "evaluate_activation_row_transfer_subspace_baselines",
        "completed",
        {
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "methods": methods,
            "k_values": k_values,
            "random_seeds": random_seeds,
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
            "[activation-row-transfer-subspace-baselines] uploaded to "
            f"hf://datasets/{args.push_to_hub_repo_id}/{path_in_repo}"
        )

    print(
        "[activation-row-transfer-subspace-baselines] completed "
        f"{completed_now} new combinations ({len(completed_keys)}/{total_expected} total); "
        f"wrote results to {run_root}"
    )


if __name__ == "__main__":
    main()
