#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from deception_honesty_axis.common import ensure_dir, find_repo_root, make_run_id, read_json, slugify, write_json
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.paper_plotting import (
    PAPER_FIGURE_FORMATS,
    TRANSFER_SERIES_COLORS,
    TRANSFER_SERIES_LABELS,
    TRANSFER_SERIES_ORDER,
    ZERO_SHOT_METHOD_COLORS,
    ZERO_SHOT_METHOD_LABELS,
    ZERO_SHOT_METHOD_ORDER,
    behavior_for_axis,
    canonical_axis_display_name,
    canonical_dataset_label,
    canonical_role_label,
    canonical_role_side,
    dataset_order_for_behavior,
    figure_output_paths,
    read_csv_rows,
)
from deception_honesty_axis.role_axis_transfer import load_role_axis_bundle


@dataclass(frozen=True)
class AxisFigureSpec:
    axis_name: str
    role_manifest_path: Path
    axis_bundle_run_dir: Path
    zero_shot_eval_run_dir: Path
    baseline_transfer_run_dir: Path
    projected_transfer_run_dirs: dict[int, Path]
    behavior: str | None
    dataset_order: list[str] | None
    layer_spec: str | None
    include_diagonal_pairs: bool


@dataclass(frozen=True)
class PaperFigureConfig:
    artifact_root: Path
    figure_set_name: str
    model_name: str
    axes: list[AxisFigureSpec]
    raw: dict[str, Any]
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready axis figures from saved run artifacts.")
    parser.add_argument("--config", type=Path, required=True, help="Paper figure manifest JSON.")
    parser.add_argument("--run-id", default=None, help="Optional fixed run id.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing completed figure run.")
    return parser.parse_args()


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def load_paper_figure_config(path: Path) -> PaperFigureConfig:
    manifest_path = path.resolve()
    raw = read_json(manifest_path)
    repo_root = find_repo_root(manifest_path.parent)
    artifact_root = _resolve_path(repo_root, str(raw.get("artifacts", {}).get("root", "artifacts")))
    experiment = raw.get("experiment", {})
    axes: list[AxisFigureSpec] = []
    for axis_row in raw.get("axes", []):
        projected_raw = axis_row.get("projected_transfer_run_dirs", {})
        projected_paths = {
            int(key): _resolve_path(repo_root, str(value))
            for key, value in projected_raw.items()
        }
        axes.append(
            AxisFigureSpec(
                axis_name=str(axis_row["axis_name"]),
                role_manifest_path=_resolve_path(repo_root, str(axis_row["role_manifest"])),
                axis_bundle_run_dir=_resolve_path(repo_root, str(axis_row["axis_bundle_run_dir"])),
                zero_shot_eval_run_dir=_resolve_path(repo_root, str(axis_row["zero_shot_eval_run_dir"])),
                baseline_transfer_run_dir=_resolve_path(repo_root, str(axis_row["baseline_transfer_run_dir"])),
                projected_transfer_run_dirs=projected_paths,
                behavior=str(axis_row["behavior"]) if axis_row.get("behavior") not in (None, "") else None,
                dataset_order=[str(item) for item in axis_row.get("dataset_order", [])] or None,
                layer_spec=str(axis_row["layer_spec"]) if axis_row.get("layer_spec") not in (None, "") else None,
                include_diagonal_pairs=bool(axis_row.get("include_diagonal_pairs", False)),
            )
        )
    if not axes:
        raise ValueError("Paper figure manifest requires a non-empty axes list")
    return PaperFigureConfig(
        artifact_root=artifact_root,
        figure_set_name=str(experiment.get("figure_set", "paper-axis-figures")),
        model_name=str(experiment.get("model_name", "unknown-model")),
        axes=axes,
        raw=raw,
        path=manifest_path,
    )


def choose_layer_spec(axis_bundle: dict[str, Any], requested: str | None) -> str:
    layer_specs = list(axis_bundle["layers"].keys())
    if requested is None:
        if len(layer_specs) != 1:
            raise ValueError(f"Axis bundle has multiple layers {layer_specs!r}; set layer_spec in the manifest")
        return str(layer_specs[0])
    requested_str = str(requested)
    if requested_str in axis_bundle["layers"]:
        return requested_str
    for layer_spec in layer_specs:
        layer_numbers = [int(value) for value in (axis_bundle["layers"][layer_spec].get("layer_numbers") or [])]
        if len(layer_numbers) == 1 and requested_str == str(layer_numbers[0]):
            return str(layer_spec)
    raise ValueError(f"Requested layer_spec={requested!r} is unavailable in axis bundle layers {layer_specs!r}")


def role_side_lookup(role_manifest_path: Path) -> dict[str, str]:
    manifest = read_json(role_manifest_path)
    lookup: dict[str, str] = {}
    for row in manifest.get("roles", []):
        role_name = str(row["name"])
        role_side = canonical_role_side(row)
        if role_side is not None:
            lookup[role_name] = role_side
    return lookup


def ordered_dataset_names(rows: list[dict[str, str]], preferred_order: list[str] | None, behavior_name: str | None) -> list[str]:
    observed = sorted({str(row["target_dataset"]) for row in rows})
    if preferred_order:
        ordered = [dataset for dataset in preferred_order if dataset in observed]
        ordered.extend(dataset for dataset in observed if dataset not in ordered)
        return ordered
    return dataset_order_for_behavior(behavior_name, observed)


def load_metric_rows(run_dir: Path, *, method_name: str | None = None) -> list[dict[str, str]]:
    rows = read_csv_rows(run_dir / "results" / "pairwise_metrics.csv")
    if method_name is None:
        return rows
    return [row for row in rows if str(row.get("method")) == method_name]


def _apply_paper_style(ax) -> None:  # noqa: ANN001
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, axis="y", color="#d9e2ec", linewidth=0.8, alpha=0.75)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa5b1")
    ax.spines["bottom"].set_color("#9aa5b1")


def _save_figure_outputs(fig, output_path: Path) -> dict[str, str]:  # noqa: ANN001
    ensure_dir(output_path.parent)
    path_map = figure_output_paths(output_path, PAPER_FIGURE_FORMATS)
    fig.savefig(path_map["png"], dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(path_map["pdf"], bbox_inches="tight", facecolor="white")
    return {fmt: str(path) for fmt, path in path_map.items()}


def save_role_pc_scatter(
    output_path: Path,
    *,
    axis_display_name: str,
    axis_bundle: dict[str, Any],
    layer_spec: str,
    role_side_by_name: dict[str, str],
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    layer_bundle = axis_bundle["layers"][layer_spec]
    role_points = []
    for role_name, projection in layer_bundle["role_projections"].items():
        side = role_side_by_name.get(str(role_name))
        if side is None:
            continue
        role_points.append(
            {
                "role_name": str(role_name),
                "role_label": canonical_role_label(str(role_name)),
                "side": side,
                "pc1": float(projection[0]),
                "pc2": float(projection[1]),
            }
        )

    anchor_projection = layer_bundle.get("anchor_projection", [0.0, 0.0])
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    _apply_paper_style(ax)
    side_meta = {
        "harmful": {"label": "Harmful Roles", "color": "#c2410c", "marker": "o"},
        "harmless": {"label": "Harmless Roles", "color": "#1d4ed8", "marker": "s"},
    }
    for side_name in ("harmless", "harmful"):
        side_rows = [row for row in role_points if row["side"] == side_name]
        if not side_rows:
            continue
        ax.scatter(
            [row["pc1"] for row in side_rows],
            [row["pc2"] for row in side_rows],
            s=95,
            alpha=0.92,
            color=side_meta[side_name]["color"],
            marker=side_meta[side_name]["marker"],
            edgecolors="white",
            linewidths=1.1,
            label=side_meta[side_name]["label"],
            zorder=3,
        )
        for row in side_rows:
            ax.annotate(
                row["role_label"],
                (row["pc1"], row["pc2"]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=9.5,
                color="#243b53",
            )

    ax.scatter(
        [float(anchor_projection[0])],
        [float(anchor_projection[1])],
        s=180,
        color="#475569",
        marker="*",
        edgecolors="white",
        linewidths=1.2,
        label="Default",
        zorder=4,
    )
    ax.annotate(
        "Default",
        (float(anchor_projection[0]), float(anchor_projection[1])),
        xytext=(8, 6),
        textcoords="offset points",
        fontsize=10,
        color="#102a43",
        fontweight="bold",
    )
    ax.axhline(0.0, color="#9aa5b1", linewidth=1.0, alpha=0.9)
    ax.axvline(0.0, color="#9aa5b1", linewidth=1.0, alpha=0.9)
    ax.set_title(f"{axis_display_name} Role Separation (PC1 vs PC2)", fontsize=15, fontweight="bold")
    ax.set_xlabel("PC1 Projection", fontsize=12)
    ax.set_ylabel("PC2 Projection", fontsize=12)
    ax.legend(frameon=False)
    fig.tight_layout()
    saved_paths = _save_figure_outputs(fig, output_path)
    plt.close(fig)
    return saved_paths


def save_zero_shot_bar_chart(
    output_path: Path,
    *,
    axis_display_name: str,
    rows: list[dict[str, str]],
    dataset_order: list[str],
    layer_spec: str,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    metric_index = {
        (str(row["target_dataset"]), str(row["method"])): float(row["auroc"])
        for row in rows
        if str(row.get("source_dataset")) == "__zero_shot__" and str(row.get("layer_spec")) == layer_spec
    }
    required_keys = [
        (dataset_name, method_name)
        for dataset_name in dataset_order
        for method_name in ZERO_SHOT_METHOD_ORDER
    ]
    missing = [key for key in required_keys if key not in metric_index]
    if missing:
        preview = ", ".join(f"{dataset}/{method}" for dataset, method in missing[:8])
        raise ValueError(f"Missing zero-shot metrics for {axis_display_name}: {preview}")

    fig, ax = plt.subplots(figsize=(max(11.5, len(dataset_order) * 1.35), 6.5))
    _apply_paper_style(ax)
    x_values = np.arange(len(dataset_order), dtype=np.float32)
    width = 0.82 / len(ZERO_SHOT_METHOD_ORDER)
    for method_index, method_name in enumerate(ZERO_SHOT_METHOD_ORDER):
        offsets = x_values - 0.41 + width * 0.5 + method_index * width
        values = [metric_index[(dataset_name, method_name)] for dataset_name in dataset_order]
        bars = ax.bar(
            offsets,
            values,
            width=width,
            color=ZERO_SHOT_METHOD_COLORS[method_name],
            label=ZERO_SHOT_METHOD_LABELS[method_name],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        for bar, value in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color="#334e68",
            )

    ax.set_xticks(x_values)
    ax.set_xticklabels([canonical_dataset_label(name) for name in dataset_order], rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color="#7b8794", linestyle="--", linewidth=1.2)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title(f"{axis_display_name} Zero-Shot Evaluation", fontsize=15, fontweight="bold")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    saved_paths = _save_figure_outputs(fig, output_path)
    plt.close(fig)
    return saved_paths


def save_transfer_heatmap(
    output_path: Path,
    *,
    axis_display_name: str,
    rows: list[dict[str, str]],
    dataset_order: list[str],
    title_suffix: str,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    matrix = np.full((len(dataset_order), len(dataset_order)), np.nan, dtype=np.float32)
    source_index = {name: idx for idx, name in enumerate(dataset_order)}
    target_index = {name: idx for idx, name in enumerate(dataset_order)}
    for row in rows:
        source_name = str(row["source_dataset"])
        target_name = str(row["target_dataset"])
        if source_name not in source_index or target_name not in target_index:
            continue
        matrix[source_index[source_name], target_index[target_name]] = float(row["auroc"])

    fig, ax = plt.subplots(figsize=(max(8.5, len(dataset_order) * 1.35), max(6.2, len(dataset_order) * 0.9)))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(dataset_order)), [canonical_dataset_label(name) for name in dataset_order], rotation=35, ha="right")
    ax.set_yticks(range(len(dataset_order)), [canonical_dataset_label(name) for name in dataset_order])
    ax.set_xlabel("Evaluation Dataset", fontsize=12)
    ax.set_ylabel("Training Dataset", fontsize=12)
    ax.set_title(f"{axis_display_name} {title_suffix}", fontsize=15, fontweight="bold")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if np.isnan(value):
                continue
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9.5,
                color="white" if value < 0.58 else "#102a43",
                fontweight="bold",
            )
    fig.colorbar(image, ax=ax, label="AUROC", fraction=0.046, pad=0.04)
    fig.tight_layout()
    saved_paths = _save_figure_outputs(fig, output_path)
    plt.close(fig)
    return saved_paths


def save_projection_pair_bars(
    output_path: Path,
    *,
    axis_display_name: str,
    dataset_order: list[str],
    baseline_rows: list[dict[str, str]],
    projected_rows_by_pc: dict[int, list[dict[str, str]]],
    include_diagonal_pairs: bool,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    pair_order = [
        (source_name, target_name)
        for source_name in dataset_order
        for target_name in dataset_order
        if include_diagonal_pairs or source_name != target_name
    ]
    series_index = {
        "baseline": {
            (str(row["source_dataset"]), str(row["target_dataset"])): float(row["auroc"])
            for row in baseline_rows
        }
    }
    for pc_count, rows in projected_rows_by_pc.items():
        series_index[f"pc{pc_count}"] = {
            (str(row["source_dataset"]), str(row["target_dataset"])): float(row["auroc"])
            for row in rows
        }

    required_series = ["baseline", "pc1", "pc2", "pc3"]
    missing: list[str] = []
    for series_name in required_series:
        if series_name not in series_index:
            missing.append(series_name)
            continue
        for pair in pair_order:
            if pair not in series_index[series_name]:
                missing.append(f"{series_name}:{pair[0]}->{pair[1]}")
                if len(missing) >= 8:
                    break
        if len(missing) >= 8:
            break
    if missing:
        raise ValueError(f"Missing projected transfer metrics for {axis_display_name}: {', '.join(missing)}")

    fig_width = max(14.0, len(pair_order) * 0.48)
    fig, ax = plt.subplots(figsize=(fig_width, 6.8))
    _apply_paper_style(ax)
    x_values = np.arange(len(pair_order), dtype=np.float32)
    width = 0.82 / len(TRANSFER_SERIES_ORDER)
    for series_index_value, series_name in enumerate(TRANSFER_SERIES_ORDER):
        offsets = x_values - 0.41 + width * 0.5 + series_index_value * width
        values = [series_index[series_name][pair] for pair in pair_order]
        ax.bar(
            offsets,
            values,
            width=width,
            color=TRANSFER_SERIES_COLORS[series_name],
            label=TRANSFER_SERIES_LABELS[series_name],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

    ax.set_xticks(x_values)
    ax.set_xticklabels(
        [f"{canonical_dataset_label(source)} -> {canonical_dataset_label(target)}" for source, target in pair_order],
        rotation=75,
        ha="right",
        fontsize=8.3,
    )
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color="#7b8794", linestyle="--", linewidth=1.2)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title(f"{axis_display_name} Cross-Dataset Transfer: Baseline vs PC Projections", fontsize=15, fontweight="bold")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    saved_paths = _save_figure_outputs(fig, output_path)
    plt.close(fig)
    return saved_paths


def main() -> None:
    args = parse_args()
    config = load_paper_figure_config(args.config)
    run_id = args.run_id or make_run_id()
    run_root = (
        config.artifact_root
        / "runs"
        / "paper-axis-figures"
        / slugify(config.model_name)
        / slugify(config.figure_set_name)
        / run_id
    )
    summary_path = run_root / "results" / "summary.json"
    if summary_path.exists() and not args.force:
        print(f"[paper-axis-figures] existing completed run found at {run_root}; skipping")
        return

    for relative in ("inputs", "results", "results/plots", "meta", "logs", "checkpoints"):
        ensure_dir(run_root / relative)

    write_analysis_manifest(
        run_root,
        {
            "config_path": str(config.path),
            "run_id": run_id,
            "figure_set_name": config.figure_set_name,
            "model_name": config.model_name,
            "axes": [
                {
                    "axis_name": axis.axis_name,
                    "role_manifest": str(axis.role_manifest_path),
                    "axis_bundle_run_dir": str(axis.axis_bundle_run_dir),
                    "zero_shot_eval_run_dir": str(axis.zero_shot_eval_run_dir),
                    "baseline_transfer_run_dir": str(axis.baseline_transfer_run_dir),
                    "projected_transfer_run_dirs": {str(key): str(value) for key, value in axis.projected_transfer_run_dirs.items()},
                    "behavior": axis.behavior,
                    "dataset_order": axis.dataset_order,
                    "layer_spec": axis.layer_spec,
                    "include_diagonal_pairs": axis.include_diagonal_pairs,
                }
                for axis in config.axes
            ],
        },
    )
    write_json(run_root / "inputs" / "config.json", config.raw)
    write_stage_status(run_root, "build_paper_axis_figures", "running", {"axis_count": len(config.axes)})

    summary_rows: list[dict[str, Any]] = []
    for axis in config.axes:
        axis_display_name = canonical_axis_display_name(axis.axis_name)
        behavior_name = axis.behavior or behavior_for_axis(axis.axis_name)
        axis_bundle = load_role_axis_bundle(axis.axis_bundle_run_dir / "results" / "axis_bundle.pt")
        layer_spec = choose_layer_spec(axis_bundle, axis.layer_spec)
        zero_shot_rows = load_metric_rows(axis.zero_shot_eval_run_dir)
        dataset_order = ordered_dataset_names(zero_shot_rows, axis.dataset_order, behavior_name)
        role_side_by_name = role_side_lookup(axis.role_manifest_path)

        axis_slug = slugify(axis_display_name)
        role_scatter_path = run_root / "results" / "plots" / f"{axis_slug}__pc1_pc2_roles.png"
        zero_shot_bar_path = run_root / "results" / "plots" / f"{axis_slug}__zero_shot_bar_chart.png"
        baseline_heatmap_path = run_root / "results" / "plots" / f"{axis_slug}__baseline_heatmap.png"
        projection_bar_path = run_root / "results" / "plots" / f"{axis_slug}__baseline_vs_projection_pairs.png"

        role_scatter_outputs = save_role_pc_scatter(
            role_scatter_path,
            axis_display_name=axis_display_name,
            axis_bundle=axis_bundle,
            layer_spec=layer_spec,
            role_side_by_name=role_side_by_name,
        )
        zero_shot_outputs = save_zero_shot_bar_chart(
            zero_shot_bar_path,
            axis_display_name=axis_display_name,
            rows=zero_shot_rows,
            dataset_order=dataset_order,
            layer_spec=layer_spec,
        )

        baseline_rows = load_metric_rows(axis.baseline_transfer_run_dir, method_name="activation_logistic")
        baseline_heatmap_outputs = save_transfer_heatmap(
            baseline_heatmap_path,
            axis_display_name=axis_display_name,
            rows=baseline_rows,
            dataset_order=dataset_order,
            title_suffix="Cross-Dataset Probe Transfer Baseline",
        )

        projected_rows_by_pc = {
            pc_count: load_metric_rows(run_dir, method_name="activation_logistic_pc_projection")
            for pc_count, run_dir in axis.projected_transfer_run_dirs.items()
        }
        projection_bar_outputs = save_projection_pair_bars(
            projection_bar_path,
            axis_display_name=axis_display_name,
            dataset_order=dataset_order,
            baseline_rows=baseline_rows,
            projected_rows_by_pc=projected_rows_by_pc,
            include_diagonal_pairs=axis.include_diagonal_pairs,
        )

        summary_rows.append(
            {
                "axis_name": axis.axis_name,
                "axis_display_name": axis_display_name,
                "behavior": behavior_name,
                "layer_spec": layer_spec,
                "dataset_order": dataset_order,
                "role_scatter_plot": role_scatter_outputs,
                "zero_shot_bar_chart": zero_shot_outputs,
                "baseline_heatmap": baseline_heatmap_outputs,
                "baseline_vs_projection_bars": projection_bar_outputs,
            }
        )

    write_json(
        summary_path,
        {
            "run_root": str(run_root),
            "figure_count": len(config.axes) * 4,
            "axes": summary_rows,
        },
    )
    write_stage_status(
        run_root,
        "build_paper_axis_figures",
        "completed",
        {
            "axis_count": len(config.axes),
            "figure_count": len(config.axes) * 4,
            "summary": str(summary_path),
        },
    )
    print(f"[paper-axis-figures] completed run_id={run_id} root={run_root}")


if __name__ == "__main__":
    main()
