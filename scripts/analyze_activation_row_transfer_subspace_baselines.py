#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

from deception_honesty_axis.common import ensure_dir, make_run_id, read_json, slugify, write_json
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.role_axis_transfer import DATASET_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare subspace transfer baselines against a completed raw activation transfer run."
    )
    parser.add_argument("--baseline-run-dir", type=Path, required=True, help="Completed raw transfer run directory.")
    parser.add_argument(
        "--subspace-run-dir",
        type=Path,
        required=True,
        help="Completed activation-row-transfer-subspace-baselines run directory.",
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"), help="Repo artifact root.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed analysis run id.")
    parser.add_argument(
        "--push-to-hub-repo-id",
        default=None,
        help="Optional HF dataset repo to upload this analysis run into.",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Optional destination path in the HF artifacts repo. Defaults to the run path under artifacts/.",
    )
    return parser.parse_args()


def load_metric_rows(run_dir: Path) -> list[dict[str, Any]]:
    metrics_path = run_dir / "results" / "pairwise_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing pairwise_metrics.csv in {run_dir / 'results'}")
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("selected_pc_count") not in (None, ""):
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


def dataset_order(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    manifest_datasets = manifest.get("datasets")
    if isinstance(manifest_datasets, list) and manifest_datasets:
        names = [str(row["name"]) for row in manifest_datasets if isinstance(row, dict) and row.get("name")]
        if names:
            return names
    seen: set[str] = set()
    ordered: list[str] = []
    for row in rows:
        for key in ("source_dataset", "target_dataset"):
            name = str(row[key])
            if name not in seen:
                seen.add(name)
                ordered.append(name)
    return ordered


def baseline_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["layer_spec"]), str(row["source_dataset"]), str(row["target_dataset"]))
        if key in lookup:
            raise ValueError(f"Duplicate raw baseline row for {key!r}")
        lookup[key] = row
    return lookup


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

        limit = max(abs(vmin), abs(vmax))
        image.set_norm(TwoSlopeNorm(vmin=-limit, vcenter=center, vmax=limit))
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
                color="white" if abs(value - (center if center is not None else 0.5)) > 0.1 else "black",
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


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_delta_rows(raw_rows: list[dict[str, Any]], subspace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_by_pair = baseline_lookup(raw_rows)
    delta_rows: list[dict[str, Any]] = []
    for row in subspace_rows:
        key = (str(row["layer_spec"]), str(row["source_dataset"]), str(row["target_dataset"]))
        raw = raw_by_pair.get(key)
        if raw is None:
            continue
        delta_rows.append(
            {
                "method": row["method"],
                "selected_pc_count": int(row["selected_pc_count"]),
                "layer_spec": row["layer_spec"],
                "source_dataset": row["source_dataset"],
                "target_dataset": row["target_dataset"],
                "is_diagonal": row["source_dataset"] == row["target_dataset"],
                "raw_auroc": float(raw["auroc"]),
                "subspace_auroc": float(row["auroc"]),
                "delta_vs_raw": float(row["auroc"]) - float(raw["auroc"]),
                "raw_auprc": float(raw["auprc"]),
                "subspace_auprc": float(row["auprc"]),
                "auprc_delta_vs_raw": float(row["auprc"]) - float(raw["auprc"]),
            }
        )
    return delta_rows


def build_summary_rows(delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: dict[tuple[str, int], dict[str, Any]] = {}
    for row in delta_rows:
        key = (str(row["method"]), int(row["selected_pc_count"]))
        bucket = summary.setdefault(
            key,
            {
                "method": key[0],
                "selected_pc_count": key[1],
                "row_count": 0,
                "mean_delta_sum": 0.0,
                "win_count": 0,
                "diag_count": 0,
                "diag_delta_sum": 0.0,
                "offdiag_count": 0,
                "offdiag_delta_sum": 0.0,
                "offdiag_win_count": 0,
            },
        )
        delta = float(row["delta_vs_raw"])
        bucket["row_count"] += 1
        bucket["mean_delta_sum"] += delta
        bucket["win_count"] += int(delta > 0.0)
        if row["is_diagonal"]:
            bucket["diag_count"] += 1
            bucket["diag_delta_sum"] += delta
        else:
            bucket["offdiag_count"] += 1
            bucket["offdiag_delta_sum"] += delta
            bucket["offdiag_win_count"] += int(delta > 0.0)
    rows: list[dict[str, Any]] = []
    for key in sorted(summary):
        bucket = summary[key]
        rows.append(
            {
                "method": bucket["method"],
                "selected_pc_count": bucket["selected_pc_count"],
                "row_count": bucket["row_count"],
                "mean_delta_vs_raw": bucket["mean_delta_sum"] / bucket["row_count"] if bucket["row_count"] else None,
                "win_rate_vs_raw": bucket["win_count"] / bucket["row_count"] if bucket["row_count"] else None,
                "diag_count": bucket["diag_count"],
                "diag_mean_delta_vs_raw": (
                    bucket["diag_delta_sum"] / bucket["diag_count"] if bucket["diag_count"] else None
                ),
                "offdiag_count": bucket["offdiag_count"],
                "offdiag_mean_delta_vs_raw": (
                    bucket["offdiag_delta_sum"] / bucket["offdiag_count"] if bucket["offdiag_count"] else None
                ),
                "offdiag_win_rate_vs_raw": (
                    bucket["offdiag_win_count"] / bucket["offdiag_count"] if bucket["offdiag_count"] else None
                ),
            }
        )
    return rows


def save_delta_heatmaps(
    output_dir: Path,
    delta_rows: list[dict[str, Any]],
    *,
    dataset_names: list[str],
) -> list[str]:
    labels = [DATASET_LABELS.get(name, name) for name in dataset_names]
    dataset_index = {name: index for index, name in enumerate(dataset_names)}
    output_paths: list[str] = []
    groups = sorted({(str(row["method"]), int(row["selected_pc_count"]), str(row["layer_spec"])) for row in delta_rows})
    for method, k, layer_spec in groups:
        rows = [
            row
            for row in delta_rows
            if str(row["method"]) == method
            and int(row["selected_pc_count"]) == k
            and str(row["layer_spec"]) == layer_spec
        ]
        matrix = np.full((len(dataset_names), len(dataset_names)), np.nan, dtype=np.float32)
        for row in rows:
            matrix[dataset_index[str(row["source_dataset"])]][dataset_index[str(row["target_dataset"])]] = float(
                row["delta_vs_raw"]
            )
        limit = max(0.1, float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 0.1)
        output_paths.append(
            save_heatmap(
                output_dir / f"{method}__k-{k:03d}__delta_vs_raw_heatmap.png",
                matrix,
                row_labels=labels,
                col_labels=labels,
                title=f"{method.replace('_', ' ').title()} Delta vs Raw @ {layer_spec} (k={k})",
                colorbar_label="AUROC delta",
                vmin=-limit,
                vmax=limit,
                cmap="coolwarm",
                center=0.0,
            )
        )
    return output_paths


def main() -> None:
    args = parse_args()
    baseline_run_dir = args.baseline_run_dir.resolve()
    subspace_run_dir = args.subspace_run_dir.resolve()
    raw_rows = load_metric_rows(baseline_run_dir)
    subspace_rows = load_metric_rows(subspace_run_dir)
    baseline_manifest = read_json(baseline_run_dir / "meta" / "run_manifest.json")
    subspace_manifest = read_json(subspace_run_dir / "meta" / "run_manifest.json")
    dataset_names = dataset_order(subspace_manifest, subspace_rows)

    run_id = args.run_id or make_run_id()
    model_slug = slugify(str(subspace_manifest.get("model_name") or "unknown-model"))
    behavior_slug = slugify(str(subspace_manifest.get("behavior_name") or "unknown-behavior"))
    dataset_set_slug = slugify(str(subspace_manifest.get("dataset_set_name") or "unknown-dataset-set"))
    run_root = (
        args.artifact_root.resolve()
        / "runs"
        / "activation-row-transfer-subspace-baselines-analysis"
        / model_slug
        / behavior_slug
        / dataset_set_slug
        / run_id
    )
    for relative in ("inputs", "results", "logs", "checkpoints", "meta"):
        ensure_dir(run_root / relative)

    write_analysis_manifest(
        run_root,
        {
            "run_id": run_id,
            "baseline_run_dir": str(baseline_run_dir),
            "subspace_run_dir": str(subspace_run_dir),
            "baseline_manifest": baseline_manifest,
            "subspace_manifest": subspace_manifest,
            "datasets": dataset_names,
        },
    )
    write_stage_status(run_root, "analyze_activation_row_transfer_subspace_baselines", "running", {})
    delta_rows = build_delta_rows(raw_rows, subspace_rows)
    summary_rows = build_summary_rows(delta_rows)
    delta_path = run_root / "results" / "pairwise_delta_vs_raw.csv"
    summary_path = run_root / "results" / "summary_by_method_k_vs_raw.csv"
    write_rows(delta_path, delta_rows)
    write_rows(summary_path, summary_rows)
    heatmap_paths = save_delta_heatmaps(run_root / "results" / "plots", delta_rows, dataset_names=dataset_names)
    write_json(
        run_root / "results" / "results.json",
        {
            "pair_count": len(delta_rows),
            "summary_count": len(summary_rows),
            "datasets": dataset_names,
            "pairwise_delta_vs_raw_path": str(delta_path),
            "summary_by_method_k_vs_raw_path": str(summary_path),
            "heatmaps": heatmap_paths,
        },
    )
    write_json(
        run_root / "checkpoints" / "progress.json",
        {
            "state": "completed",
            "pair_count": len(delta_rows),
            "summary_count": len(summary_rows),
            "heatmaps": heatmap_paths,
        },
    )
    write_stage_status(
        run_root,
        "analyze_activation_row_transfer_subspace_baselines",
        "completed",
        {
            "pair_count": len(delta_rows),
            "summary_count": len(summary_rows),
            "heatmaps": heatmap_paths,
        },
    )

    if args.push_to_hub_repo_id:
        from huggingface_hub import upload_folder

        default_path_in_repo = str(run_root.relative_to(args.artifact_root.resolve()))
        path_in_repo = args.path_in_repo or default_path_in_repo
        upload_folder(
            repo_id=args.push_to_hub_repo_id,
            repo_type="dataset",
            folder_path=str(run_root),
            path_in_repo=path_in_repo,
            token=os.environ.get("HF_TOKEN"),
        )
        print(
            "[activation-row-transfer-subspace-baselines-analysis] uploaded to "
            f"hf://datasets/{args.push_to_hub_repo_id}/{path_in_repo}"
        )

    print(
        "[activation-row-transfer-subspace-baselines-analysis] completed "
        f"{len(delta_rows)} aligned pairs; wrote results to {run_root}"
    )


if __name__ == "__main__":
    main()
