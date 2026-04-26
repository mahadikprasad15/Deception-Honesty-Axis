#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
from typing import Any

import numpy as np

from deception_honesty_axis.common import ensure_dir, make_run_id, read_json, slugify, write_json
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.role_axis_transfer import DATASET_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple activation-row transfer runs over the same source-target matrix."
    )
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        metavar=("LABEL", "RUN_DIR"),
        required=True,
        help="Comparison run label and completed transfer run directory. Pass at least two.",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default=None,
        help="Label to use as the baseline for delta views. Defaults to the first --run label.",
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"), help="Repo artifact root.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed comparison run id.")
    return parser.parse_args()


def _load_metric_rows(run_dir: Path) -> list[dict[str, str]]:
    metrics_path = run_dir / "results" / "pairwise_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing pairwise_metrics.csv in {run_dir / 'results'}")
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_run_payload(label: str, run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "meta" / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run_manifest.json in {run_dir / 'meta'}")
    return {
        "label": str(label),
        "run_dir": run_dir,
        "manifest": read_json(manifest_path),
        "rows": _load_metric_rows(run_dir),
    }


def _dataset_order(manifest: dict[str, Any], rows: list[dict[str, str]]) -> list[str]:
    manifest_datasets = manifest.get("datasets")
    if isinstance(manifest_datasets, list) and manifest_datasets:
        names = [str(row["name"]) for row in manifest_datasets if row.get("name")]
        if names:
            return names
    seen: set[str] = set()
    ordered: list[str] = []
    for row in rows:
        for key in ("source_dataset", "target_dataset"):
            dataset = str(row[key])
            if dataset in seen:
                continue
            seen.add(dataset)
            ordered.append(dataset)
    return ordered


def _dataset_behavior_map(manifest: dict[str, Any]) -> dict[str, str | None]:
    payload: dict[str, str | None] = {}
    for row in manifest.get("datasets", []):
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if name in (None, ""):
            continue
        behavior = row.get("behavior")
        payload[str(name)] = None if behavior in (None, "") else str(behavior)
    return payload


def _metric_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            _canonical_layer_spec(str(row["layer_spec"])),
            str(row["source_dataset"]),
            str(row["target_dataset"]),
        )
        if key in lookup:
            raise ValueError(f"Duplicate metric row for key={key!r}")
        lookup[key] = row
    return lookup


def _canonical_layer_spec(layer_spec: str) -> str:
    tokens = [token.strip() for token in str(layer_spec).split("__") if token.strip()]
    if tokens and all(token == tokens[0] for token in tokens):
        return tokens[0]
    return str(layer_spec)


def _save_heatmap(
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
    fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 1.4), max(7, len(row_labels) * 0.75)))
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


def _save_aggregate_plot(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    baseline_label: str,
) -> str:
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    categories = [str(row["summary_name"]) for row in rows]
    labels = [str(row["label"]) for row in rows]
    unique_categories = list(dict.fromkeys(categories))
    unique_labels = list(dict.fromkeys(labels))
    values = {
        (str(row["summary_name"]), str(row["label"])): (
            np.nan if row["mean_auroc"] is None else float(row["mean_auroc"])
        )
        for row in rows
    }

    x_positions = np.arange(len(unique_categories), dtype=np.float32)
    width = 0.8 / max(1, len(unique_labels))
    fig, ax = plt.subplots(figsize=(max(10, len(unique_categories) * 2.2), 6.5))
    for index, label in enumerate(unique_labels):
        offsets = x_positions - 0.4 + width * 0.5 + index * width
        label_values = [values[(category, label)] for category in unique_categories]
        bars = ax.bar(offsets, label_values, width=width, label=label)
        for bar, value in zip(bars, label_values, strict=False):
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(0.98, value + 0.015),
                f"{value:.02f}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=90,
            )
    ax.set_xticks(x_positions, [category.replace("_", " ").title() for category in unique_categories], rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean AUROC")
    ax.set_title(f"Transfer Summary Comparison (baseline: {baseline_label})", fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Run")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output_path)


def _save_pairwise_grouped_bar_plot(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    labels: list[str],
    offdiag_only: bool,
) -> str:
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    filtered_rows = [row for row in rows if (not offdiag_only or not bool(row["is_diagonal"]))]
    pair_labels = [f"{row['source_label']}→{row['target_label']}" for row in filtered_rows]

    x_positions = np.arange(len(filtered_rows), dtype=np.float32)
    width = 0.8 / max(1, len(labels))
    fig_width = max(14, len(filtered_rows) * 0.65)
    fig, ax = plt.subplots(figsize=(fig_width, 6.8))

    for index, label in enumerate(labels):
        offsets = x_positions - 0.4 + width * 0.5 + index * width
        values = [float(row[f"auroc__{label}"]) for row in filtered_rows]
        bars = ax.bar(offsets, values, width=width, label=label)
        for bar, value in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(0.98, value + 0.012),
                f"{value:.02f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    title_prefix = "Off-Diagonal " if offdiag_only else ""
    ax.set_xticks(x_positions, pair_labels, rotation=65, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_xlabel("Source→Target Dataset Pair")
    ax.set_title(f"{title_prefix}Pairwise AUROC Comparison", fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Run")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output_path)


def _summary_rows(
    aligned_rows: list[dict[str, Any]],
    labels: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    predicates = {
        "overall": lambda row: True,
        "diagonal": lambda row: bool(row["is_diagonal"]),
        "offdiag": lambda row: not bool(row["is_diagonal"]),
        "within_behavior": lambda row: bool(row["is_within_behavior"]),
        "within_behavior_offdiag": lambda row: bool(row["is_within_behavior"]) and not bool(row["is_diagonal"]),
        "cross_behavior": lambda row: bool(row["is_cross_behavior"]),
    }
    for label in labels:
        for summary_name, predicate in predicates.items():
            values = [float(row[f"auroc__{label}"]) for row in aligned_rows if predicate(row)]
            rows.append(
                {
                    "label": label,
                    "summary_name": summary_name,
                    "count": len(values),
                    "mean_auroc": float(sum(values) / len(values)) if values else None,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    if len(args.run) < 2:
        raise ValueError("Pass at least two --run LABEL RUN_DIR pairs")

    run_payloads = [_load_run_payload(label, Path(run_dir).resolve()) for label, run_dir in args.run]
    labels = [payload["label"] for payload in run_payloads]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Run labels must be unique, got {labels!r}")
    baseline_label = args.baseline_label or labels[0]
    if baseline_label not in labels:
        raise ValueError(f"baseline_label={baseline_label!r} is not among run labels {labels!r}")

    baseline_payload = next(payload for payload in run_payloads if payload["label"] == baseline_label)
    model_name = str(baseline_payload["manifest"].get("model_name") or "unknown-model")
    behavior_name = str(baseline_payload["manifest"].get("behavior_name") or "unknown-behavior")
    dataset_set_name = str(baseline_payload["manifest"].get("dataset_set_name") or "unknown-dataset-set")
    dataset_order = _dataset_order(baseline_payload["manifest"], baseline_payload["rows"])
    dataset_behavior_map = _dataset_behavior_map(baseline_payload["manifest"])

    lookups = {payload["label"]: _metric_lookup(payload["rows"]) for payload in run_payloads}
    expected_keys = set(lookups[baseline_label].keys())
    for label, lookup in lookups.items():
        if set(lookup.keys()) != expected_keys:
            missing = sorted(expected_keys - set(lookup.keys()))
            extra = sorted(set(lookup.keys()) - expected_keys)
            raise ValueError(
                f"Run {label!r} does not match baseline matrix coverage; missing={missing[:5]!r}, extra={extra[:5]!r}"
            )

    aligned_rows: list[dict[str, Any]] = []
    for layer_spec, source_dataset, target_dataset in sorted(expected_keys):
        source_behavior = dataset_behavior_map.get(source_dataset)
        target_behavior = dataset_behavior_map.get(target_dataset)
        row: dict[str, Any] = {
            "layer_spec": layer_spec,
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "source_label": DATASET_LABELS.get(source_dataset, source_dataset),
            "target_label": DATASET_LABELS.get(target_dataset, target_dataset),
            "source_behavior": source_behavior,
            "target_behavior": target_behavior,
            "is_diagonal": source_dataset == target_dataset,
            "is_within_behavior": source_behavior is not None and source_behavior == target_behavior,
            "is_cross_behavior": (
                source_behavior is not None and target_behavior is not None and source_behavior != target_behavior
            ),
        }
        for label, lookup in lookups.items():
            row_payload = lookup[(layer_spec, source_dataset, target_dataset)]
            row[f"method__{label}"] = str(row_payload["method"])
            row[f"auroc__{label}"] = float(row_payload["auroc"])
            row[f"auprc__{label}"] = float(row_payload["auprc"])
        for left_label, right_label in itertools.combinations(labels, 2):
            row[f"delta__{right_label}__minus__{left_label}"] = float(row[f"auroc__{right_label}"]) - float(
                row[f"auroc__{left_label}"]
            )
        aligned_rows.append(row)

    run_id = args.run_id or make_run_id()
    comparison_slug = slugify("-vs-".join(labels))
    run_root = (
        args.artifact_root
        / "runs"
        / "activation-row-transfer-comparison"
        / slugify(model_name)
        / slugify(behavior_name)
        / slugify(dataset_set_name)
        / comparison_slug
        / run_id
    )
    for relative in ("inputs", "results", "logs", "checkpoints", "meta"):
        ensure_dir(run_root / relative)

    write_analysis_manifest(
        run_root,
        {
            "run_id": run_id,
            "baseline_label": baseline_label,
            "runs": [
                {
                    "label": payload["label"],
                    "run_dir": str(payload["run_dir"]),
                    "method_names": sorted({str(row["method"]) for row in payload["rows"]}),
                }
                for payload in run_payloads
            ],
            "model_name": model_name,
            "behavior_name": behavior_name,
            "dataset_set_name": dataset_set_name,
            "dataset_order": dataset_order,
            "dataset_behavior_map": dataset_behavior_map,
        },
    )
    write_stage_status(run_root, "compare_activation_row_transfer_runs", "running", {"run_id": run_id})
    write_json(run_root / "checkpoints" / "progress.json", {"state": "running", "aligned_row_count": len(aligned_rows)})

    aligned_path = run_root / "results" / "aligned_metrics.csv"
    with aligned_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(aligned_rows[0].keys()) if aligned_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aligned_rows)

    aggregate_rows = _summary_rows(aligned_rows, labels)
    aggregate_path = run_root / "results" / "aggregate_summary.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "summary_name", "count", "mean_auroc"])
        writer.writeheader()
        writer.writerows(aggregate_rows)

    plot_paths: list[str] = []
    row_labels = [DATASET_LABELS.get(dataset, dataset) for dataset in dataset_order]
    col_labels = row_labels
    plots_dir = ensure_dir(run_root / "results" / "plots")

    for left_label, right_label in itertools.combinations(labels, 2):
        delta_matrix = np.zeros((len(dataset_order), len(dataset_order)), dtype=np.float32)
        for row in aligned_rows:
            row_index = dataset_order.index(str(row["source_dataset"]))
            col_index = dataset_order.index(str(row["target_dataset"]))
            delta_matrix[row_index, col_index] = float(row[f"delta__{right_label}__minus__{left_label}"])
        plot_paths.append(
            _save_heatmap(
                plots_dir / f"{slugify(right_label)}__minus__{slugify(left_label)}__auroc_heatmap.png",
                delta_matrix,
                row_labels=row_labels,
                col_labels=col_labels,
                title=f"{right_label} - {left_label} AUROC",
                colorbar_label="AUROC Delta",
                vmin=-0.25,
                vmax=0.25,
                center=0.0,
                cmap="RdBu_r",
            )
        )

    aggregate_plot_path = _save_aggregate_plot(
        plots_dir / "aggregate_summary_comparison.png",
        aggregate_rows,
        baseline_label=baseline_label,
    )
    plot_paths.append(aggregate_plot_path)
    plot_paths.append(
        _save_pairwise_grouped_bar_plot(
            plots_dir / "pairwise_auroc_comparison.png",
            aligned_rows,
            labels=labels,
            offdiag_only=False,
        )
    )
    plot_paths.append(
        _save_pairwise_grouped_bar_plot(
            plots_dir / "pairwise_offdiag_auroc_comparison.png",
            aligned_rows,
            labels=labels,
            offdiag_only=True,
        )
    )

    write_json(
        run_root / "results" / "results.json",
        {
            "aligned_row_count": len(aligned_rows),
            "baseline_label": baseline_label,
            "labels": labels,
            "dataset_order": dataset_order,
            "plot_paths": plot_paths,
            "aggregate_summary_path": str(aggregate_path),
            "aligned_metrics_path": str(aligned_path),
        },
    )
    write_json(
        run_root / "checkpoints" / "progress.json",
        {"state": "completed", "aligned_row_count": len(aligned_rows), "plot_paths": plot_paths},
    )
    write_stage_status(
        run_root,
        "compare_activation_row_transfer_runs",
        "completed",
        {"aligned_row_count": len(aligned_rows), "plot_paths": plot_paths},
    )

    print(
        f"[activation-row-transfer-comparison] compared {len(labels)} runs over {len(aligned_rows)} aligned rows; "
        f"wrote outputs to {run_root}"
    )


if __name__ == "__main__":
    main()
