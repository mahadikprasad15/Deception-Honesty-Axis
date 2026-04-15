#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deception_honesty_axis.activation_row_transfer import (
    ensure_loaded_dataset_compatibility,
    load_activation_row_dataset,
)
from deception_honesty_axis.activation_row_transfer_config import load_activation_row_transfer_config
from deception_honesty_axis.common import ensure_dir, make_run_id, read_json, slugify, write_json
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.role_axis_transfer import (
    DATASET_LABELS,
    ZERO_SHOT_SOURCE,
    append_csv_row,
    compute_binary_metrics,
    evaluate_zero_shot,
    load_role_axis_bundle,
    read_completed_metric_keys,
    scores_for_layer,
)
from deception_honesty_axis.sycophancy_activations import normalize_activation_pooling


METRIC_FIELDS = [
    "method",
    "pc_index",
    "pc_label",
    "layer_spec",
    "layer_label",
    "source_dataset",
    "target_dataset",
    "target_behavior",
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
    parser = argparse.ArgumentParser(
        description="Evaluate every stored PC in a role-axis bundle across a mixed dataset inventory."
    )
    parser.add_argument("--config", type=Path, required=True, help="Feature transfer config JSON.")
    parser.add_argument("--axis-bundle-run-dir", type=Path, required=True, help="Run dir containing results/axis_bundle.pt.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id.")
    parser.add_argument("--top-k", type=int, default=12, help="Number of top PCs to use for detailed profile and cosine plots.")
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=0.65,
        help="Threshold for including a PC in a dataset's top-PC set when computing Jaccard overlap.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
        help="Print a progress update every N newly completed dataset/PC combinations.",
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
        bundle_name = manifest.get("bundle_name")
        bundle_slug = manifest.get("bundle_slug")
        if bundle_name:
            return slugify(f"{bundle_name}-pc-sweep")
        if bundle_slug:
            return slugify(f"{bundle_slug}-pc-sweep")
        axis_name = manifest.get("axis_name")
        variant_name = manifest.get("variant_name")
        if axis_name and variant_name:
            return slugify(f"{axis_name}-{variant_name}-pc-sweep")
        if axis_name:
            return slugify(f"{axis_name}-pc-sweep")
    parts = list(axis_bundle_dir.parts)
    if len(parts) >= 2:
        return slugify("-".join(parts[-2:] + ["pc-sweep"]))
    return slugify(f"{axis_bundle_dir.name}-pc-sweep")


def _dataset_label(dataset_name: str) -> str:
    return DATASET_LABELS.get(dataset_name, dataset_name)


def _pc_method(pc_index: int) -> str:
    return f"pc{pc_index}_zero_shot"


def _pc_label(pc_index: int) -> str:
    return f"PC{pc_index}"


def _cosine_matrix(components: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(components, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    normalized = components / safe
    return normalized @ normalized.T


def _behavior_separators(dataset_behaviors: list[str | None]) -> list[int]:
    separators: list[int] = []
    for index in range(1, len(dataset_behaviors)):
        if dataset_behaviors[index] != dataset_behaviors[index - 1]:
            separators.append(index)
    return separators


def _build_pc_summary(metric_rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(metric_rows)
    if df.empty:
        return pd.DataFrame()
    df["pc_index"] = df["pc_index"].astype(int)
    for column in ("auroc", "auprc", "balanced_accuracy", "f1", "accuracy", "count"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    grouped = df.groupby(["layer_spec", "pc_index", "pc_label"], as_index=False)
    summary = grouped.agg(
        mean_auroc=("auroc", "mean"),
        mean_auprc=("auprc", "mean"),
        std_auroc=("auroc", "std"),
    )
    for behavior_name in sorted(value for value in df["target_behavior"].dropna().unique()):
        behavior_rows = df[df["target_behavior"] == behavior_name]
        if behavior_rows.empty:
            continue
        behavior_summary = (
            behavior_rows.groupby(["layer_spec", "pc_index"], as_index=False)["auroc"]
            .mean()
            .rename(columns={"auroc": f"mean_auroc_{slugify(behavior_name)}"})
        )
        summary = summary.merge(behavior_summary, on=["layer_spec", "pc_index"], how="left")
    return summary


def _save_plot_heatmap(
    output_path: Path,
    matrix: pd.DataFrame,
    *,
    title: str,
    center: float | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    behavior_separators: list[int] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(max(12, matrix.shape[1] * 0.32), max(5, matrix.shape[0] * 0.65)))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".02f",
        linewidths=0.5,
        cbar_kws={"label": "AUROC" if center == 0.5 or center is None else "Value"},
    )
    if behavior_separators:
        for separator in behavior_separators:
            ax.axhline(separator, color="black", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("PC")
    ax.set_ylabel("Dataset")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_pc_behavior_scatter(
    output_path: Path,
    summary_df: pd.DataFrame,
    *,
    deception_column: str | None,
    sycophancy_column: str | None,
    top_k: int,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if deception_column is None or sycophancy_column is None:
        return
    if deception_column not in summary_df.columns or sycophancy_column not in summary_df.columns:
        return

    plot_df = summary_df.copy()
    variance_columns = [
        column
        for column in plot_df.columns
        if column.startswith("mean_auroc_")
    ]
    if variance_columns:
        plot_df["behavior_mean_spread"] = plot_df[variance_columns].std(axis=1).fillna(0.0)
    else:
        plot_df["behavior_mean_spread"] = 0.0

    top_labels = set(plot_df.sort_values("mean_auroc", ascending=False).head(max(1, top_k))["pc_label"].tolist())

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(
        plot_df[deception_column],
        plot_df[sycophancy_column],
        c=plot_df["behavior_mean_spread"],
        s=60 + 220 * plot_df["mean_auroc"].clip(0.0, 1.0),
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
    )
    min_bound = min(float(plot_df[deception_column].min()), float(plot_df[sycophancy_column].min()), 0.45)
    max_bound = max(float(plot_df[deception_column].max()), float(plot_df[sycophancy_column].max()), 0.95)
    ax.plot([min_bound, max_bound], [min_bound, max_bound], linestyle="--", color="#6c757d", linewidth=1.5)
    ax.set_xlim(min_bound, max_bound)
    ax.set_ylim(min_bound, max_bound)
    ax.set_xlabel("Mean Deception AUROC")
    ax.set_ylabel("Mean Sycophancy AUROC")
    ax.set_title(title)
    for _, row in plot_df.iterrows():
        if row["pc_label"] in top_labels:
            ax.annotate(
                row["pc_label"],
                (row[deception_column], row[sycophancy_column]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=10,
            )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Across-Behavior Mean Spread")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_top_pc_profiles(
    output_path: Path,
    ordered_matrix: pd.DataFrame,
    *,
    top_pc_labels: list[str],
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not top_pc_labels:
        return
    plot_df = ordered_matrix[top_pc_labels]
    ncols = 3
    nrows = int(np.ceil(len(top_pc_labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(4.5 * nrows, 6)), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(-1)
    palette = sns.color_palette("crest", n_colors=plot_df.shape[0])

    for ax, pc_label in zip(axes_array, top_pc_labels, strict=False):
        values = plot_df[pc_label].to_numpy(dtype=np.float32)
        ax.bar(np.arange(plot_df.shape[0]), values, color=palette)
        ax.set_title(pc_label, fontsize=12, fontweight="bold")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(np.arange(plot_df.shape[0]))
        ax.set_xticklabels(plot_df.index.tolist(), rotation=35, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes_array[len(top_pc_labels):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=18, fontweight="bold")
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_pc_landscape_artifacts(
    *,
    output_dir: Path,
    metric_rows: list[dict[str, Any]],
    layer_spec: str,
    dataset_order: list[str],
    dataset_behavior_map: dict[str, str | None],
    layer_bundle: dict[str, Any],
    top_k: int,
    jaccard_threshold: float,
) -> dict[str, str]:
    ensure_dir(output_dir)
    layer_rows = [row for row in metric_rows if str(row["layer_spec"]) == str(layer_spec)]
    if not layer_rows:
        return {}

    summary_df = _build_pc_summary(layer_rows)
    if summary_df.empty:
        return {}
    summary_df["pc_index"] = summary_df["pc_index"].astype(int)
    summary_df = summary_df.sort_values(["mean_auroc", "pc_index"], ascending=[False, True]).reset_index(drop=True)
    sorted_pc_labels = summary_df["pc_label"].tolist()
    sorted_pc_indices = summary_df["pc_index"].tolist()

    metrics_df = pd.DataFrame(layer_rows)
    metrics_df["pc_index"] = metrics_df["pc_index"].astype(int)
    metrics_df["auroc"] = pd.to_numeric(metrics_df["auroc"], errors="coerce")
    ordered_matrix = (
        metrics_df.assign(dataset_label=metrics_df["target_dataset"].map(_dataset_label))
        .pivot(index="dataset_label", columns="pc_label", values="auroc")
        .loc[[_dataset_label(name) for name in dataset_order], sorted_pc_labels]
    )
    ordered_matrix.to_csv(output_dir / f"pc_auroc_matrix__{layer_spec}.csv")

    dataset_behaviors = [dataset_behavior_map.get(name) for name in dataset_order]
    behavior_separators = _behavior_separators(dataset_behaviors)
    heatmap_path = output_dir / f"pc_auroc_heatmap__{layer_spec}.png"
    _save_plot_heatmap(
        heatmap_path,
        ordered_matrix,
        title=f"All-PC AUROC by Dataset @ Layer {layer_spec}",
        center=0.5,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        behavior_separators=behavior_separators,
    )

    deception_column = next((column for column in summary_df.columns if column == "mean_auroc_deception"), None)
    sycophancy_column = next((column for column in summary_df.columns if column == "mean_auroc_sycophancy"), None)
    scatter_path = output_dir / f"pc_behavior_scatter__{layer_spec}.png"
    _save_pc_behavior_scatter(
        scatter_path,
        summary_df,
        deception_column=deception_column,
        sycophancy_column=sycophancy_column,
        top_k=top_k,
        title=f"PC Behavior Means @ Layer {layer_spec}",
    )

    top_sets = {
        dataset_label: {
            pc_label
            for pc_label, value in ordered_matrix.loc[dataset_label].items()
            if float(value) >= float(jaccard_threshold)
        }
        for dataset_label in ordered_matrix.index
    }
    jaccard = np.zeros((len(ordered_matrix.index), len(ordered_matrix.index)), dtype=np.float32)
    for row_index, left_name in enumerate(ordered_matrix.index):
        for col_index, right_name in enumerate(ordered_matrix.index):
            left = top_sets[left_name]
            right = top_sets[right_name]
            union = left | right
            jaccard[row_index, col_index] = float(len(left & right) / len(union)) if union else 1.0
    jaccard_df = pd.DataFrame(jaccard, index=ordered_matrix.index, columns=ordered_matrix.index)
    jaccard_df.to_csv(output_dir / f"dataset_pc_jaccard__{layer_spec}.csv")
    jaccard_path = output_dir / f"dataset_pc_jaccard_heatmap__{layer_spec}.png"
    _save_plot_heatmap(
        jaccard_path,
        jaccard_df,
        title=f"Dataset Top-PC Jaccard @ Layer {layer_spec} (threshold={jaccard_threshold:.2f})",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        behavior_separators=behavior_separators,
    )

    top_k = max(1, min(int(top_k), len(sorted_pc_indices)))
    top_pc_labels = sorted_pc_labels[:top_k]
    top_pc_indices = [pc_index - 1 for pc_index in sorted_pc_indices[:top_k]]
    components = layer_bundle["pc_components"].detach().cpu().numpy().astype(np.float32)
    top_components = components[top_pc_indices]
    cosine_df = pd.DataFrame(
        _cosine_matrix(top_components),
        index=top_pc_labels,
        columns=top_pc_labels,
    )
    cosine_df.to_csv(output_dir / f"top_pc_cosine__{layer_spec}.csv")
    cosine_path = output_dir / f"top_pc_cosine_heatmap__{layer_spec}.png"
    _save_plot_heatmap(
        cosine_path,
        cosine_df,
        title=f"Top-{top_k} PC Cosine Similarity @ Layer {layer_spec}",
        center=0.0,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
    )

    profiles_path = output_dir / f"top_pc_profiles__{layer_spec}.png"
    _save_top_pc_profiles(
        profiles_path,
        ordered_matrix,
        top_pc_labels=top_pc_labels,
        title=f"Top-{top_k} PC Dataset Profiles @ Layer {layer_spec}",
    )

    summary_df.to_csv(output_dir / f"pc_summary__{layer_spec}.csv", index=False)
    summary_df.to_json(output_dir / f"pc_summary__{layer_spec}.json", orient="records", indent=2)

    return {
        "pc_auroc_matrix": str(output_dir / f"pc_auroc_matrix__{layer_spec}.csv"),
        "pc_summary_csv": str(output_dir / f"pc_summary__{layer_spec}.csv"),
        "pc_auroc_heatmap": str(heatmap_path),
        "pc_behavior_scatter": str(scatter_path),
        "dataset_pc_jaccard_heatmap": str(jaccard_path),
        "top_pc_cosine_heatmap": str(cosine_path),
        "top_pc_profiles": str(profiles_path),
    }


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
            layer_numbers = [int(value) for value in axis_bundle["layers"][layer_spec].get("layer_numbers", [])]
            if len(layer_numbers) == 1 and int(layer_numbers[0]) == int(config.expected_layer_number):
                compatible_specs.append(layer_spec)
        if not compatible_specs:
            raise ValueError(
                f"No axis bundle layers match expected layer {config.expected_layer_number}; "
                f"bundle_layers={layer_specs!r}"
            )
        layer_specs = compatible_specs

    dataset_names = [dataset.name for dataset in loaded_datasets]
    dataset_behavior_map = {dataset_config.name: dataset_config.behavior for dataset_config in config.datasets}
    axis_slug = infer_axis_slug(axis_bundle_dir)
    run_id = args.run_id or make_run_id()
    run_root = (
        config.artifact_root
        / "runs"
        / "behavior-axis-pc-sweep"
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
            "layer_specs": layer_specs,
            "resolved_pooling": compatibility["activation_pooling"],
            "resolved_layer_number": compatibility["activation_layer_number"],
            "feature_dim": compatibility["feature_dim"],
            "dataset_order": dataset_names,
            "dataset_behaviors": dataset_behavior_map,
            "datasets": [dataset_config.manifest_row() for dataset_config in config.datasets],
            "dataset_summaries": [dataset.summary() for dataset in loaded_datasets],
            "top_k": args.top_k,
            "jaccard_threshold": args.jaccard_threshold,
        },
    )
    write_json(run_root / "inputs" / "config.json", config.raw)

    results_dir = run_root / "results"
    metrics_path = results_dir / "pairwise_metrics.csv"
    progress_path = run_root / "checkpoints" / "progress.json"
    completed_keys = read_completed_metric_keys(metrics_path)
    total_expected = 0
    for layer_spec in layer_specs:
        pc_count = int(axis_bundle["layers"][layer_spec]["pc_components"].shape[0])
        total_expected += len(dataset_names) * pc_count
    completed_now = 0

    write_stage_status(
        run_root,
        "evaluate_behavior_axis_pc_sweep",
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
            f"[behavior-axis-pc-sweep] resuming with {len(completed_keys)}/{total_expected} "
            "combinations already complete"
        )
    else:
        print(
            f"[behavior-axis-pc-sweep] starting fresh with {total_expected} combinations "
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
        write_stage_status(run_root, "evaluate_behavior_axis_pc_sweep", "running", payload)

    def maybe_print_progress(last_row: dict[str, Any]) -> None:
        if completed_now == 1 or (args.print_every > 0 and completed_now % args.print_every == 0):
            print(
                f"[behavior-axis-pc-sweep] {len(completed_keys)}/{total_expected} complete "
                f"({completed_now} new this run); {last_row['pc_label']} @ {last_row['layer_spec']} "
                f"{last_row['target_dataset']} AUROC={float(last_row['auroc']):.3f}"
            )

    for layer_spec in layer_specs:
        layer_bundle = axis_bundle["layers"][layer_spec]
        layer_label = str(layer_bundle.get("layer_label", layer_spec))
        pc_count = int(layer_bundle["pc_components"].shape[0])
        for dataset in loaded_datasets:
            layer_scores = scores_for_layer(dataset.eval.features, layer_bundle)
            pc_scores = layer_scores["pc_scores"]
            for pc_index_zero_based in range(pc_count):
                pc_index = pc_index_zero_based + 1
                method = _pc_method(pc_index)
                metric_key = (method, layer_spec, ZERO_SHOT_SOURCE, dataset.name)
                if metric_key in completed_keys:
                    continue
                honest_like_scores = pc_scores[:, pc_index_zero_based]
                metrics, _deceptive_scores, _probabilities = evaluate_zero_shot(
                    dataset.eval.labels,
                    honest_like_scores,
                )
                row = {
                    "method": method,
                    "pc_index": pc_index,
                    "pc_label": _pc_label(pc_index),
                    "layer_spec": layer_spec,
                    "layer_label": layer_label,
                    "source_dataset": ZERO_SHOT_SOURCE,
                    "target_dataset": dataset.name,
                    "target_behavior": dataset_behavior_map.get(dataset.name),
                    "train_split": axis_slug,
                    "eval_split": dataset.eval.split_name,
                    "completed_at": utc_now_iso(),
                    **metrics,
                }
                append_csv_row(metrics_path, METRIC_FIELDS, row)
                completed_keys.add(metric_key)
                completed_now += 1
                update_progress()
                maybe_print_progress(row)

    metric_rows: list[dict[str, Any]] = []
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            metric_rows = list(csv.DictReader(handle))

    plots_dir = results_dir / "plots"
    plot_artifacts: dict[str, dict[str, str]] = {}
    for layer_spec in layer_specs:
        plot_artifacts[layer_spec] = _save_pc_landscape_artifacts(
            output_dir=plots_dir,
            metric_rows=metric_rows,
            layer_spec=layer_spec,
            dataset_order=dataset_names,
            dataset_behavior_map=dataset_behavior_map,
            layer_bundle=axis_bundle["layers"][layer_spec],
            top_k=args.top_k,
            jaccard_threshold=args.jaccard_threshold,
        )

    summary_df = _build_pc_summary(metric_rows)
    if not summary_df.empty:
        summary_df.to_csv(results_dir / "summary_by_pc.csv", index=False)

    write_json(
        results_dir / "results.json",
        {
            "record_count": len(metric_rows),
            "datasets": dataset_names,
            "resolved_pooling": compatibility["activation_pooling"],
            "resolved_layer_number": compatibility["activation_layer_number"],
            "feature_dim": compatibility["feature_dim"],
            "axis_slug": axis_slug,
            "top_k": args.top_k,
            "jaccard_threshold": args.jaccard_threshold,
            "plot_artifacts": plot_artifacts,
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
            "plot_artifacts": plot_artifacts,
        },
    )
    write_stage_status(
        run_root,
        "evaluate_behavior_axis_pc_sweep",
        "completed",
        {
            "expected_combinations": total_expected,
            "completed_combinations": len(completed_keys),
            "axis_slug": axis_slug,
            "plot_artifacts": plot_artifacts,
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
        print(f"[behavior-axis-pc-sweep] uploaded to hf://datasets/{args.push_to_hub_repo_id}/{path_in_repo}")

    print(
        f"[behavior-axis-pc-sweep] completed {completed_now} new combinations "
        f"({len(completed_keys)}/{total_expected} total); wrote results to {run_root}"
    )


if __name__ == "__main__":
    main()
