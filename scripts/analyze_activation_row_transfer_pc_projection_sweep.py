#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deception_honesty_axis.common import ensure_dir, make_run_id, read_json, write_json
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.role_axis_transfer import DATASET_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a cumulative projected-transfer sweep against a raw transfer baseline."
    )
    parser.add_argument("--baseline-run-dir", type=Path, required=True, help="Completed raw transfer run directory.")
    parser.add_argument(
        "--pc-sweep-run-dir",
        type=Path,
        required=True,
        help="Completed activation-row-transfer-pc-projection-sweep run directory.",
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
        if "selected_pc_count" in row and row["selected_pc_count"] not in ("", None):
            row["selected_pc_count"] = int(row["selected_pc_count"])
        for key in ("auroc", "auprc", "balanced_accuracy", "f1", "accuracy", "count"):
            if row.get(key) not in (None, ""):
                row[key] = float(row[key])
    return rows


def dataset_order(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
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


def dataset_behavior_map(manifest: dict[str, Any]) -> dict[str, str | None]:
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


def baseline_metric_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["layer_spec"]), str(row["source_dataset"]), str(row["target_dataset"]))
        if key in lookup:
            raise ValueError(f"Duplicate baseline metric row for key={key!r}")
        lookup[key] = row
    return lookup


def sweep_metric_lookup(rows: list[dict[str, Any]]) -> dict[int, dict[tuple[str, str, str], dict[str, Any]]]:
    by_k: dict[int, dict[tuple[str, str, str], dict[str, Any]]] = {}
    for row in rows:
        k = int(row["selected_pc_count"])
        lookup = by_k.setdefault(k, {})
        key = (str(row["layer_spec"]), str(row["source_dataset"]), str(row["target_dataset"]))
        if key in lookup:
            raise ValueError(f"Duplicate sweep metric row for k={k}, key={key!r}")
        lookup[key] = row
    return by_k


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


def save_summary_vs_k_plot(output_path: Path, summary_df: pd.DataFrame) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        summary_df["selected_pc_count"],
        summary_df["offdiag_mean_auroc"],
        marker="o",
        linewidth=2.5,
        label="Mean Off-Diagonal AUROC",
    )
    ax.plot(
        summary_df["selected_pc_count"],
        summary_df["diagonal_mean_auroc"],
        marker="o",
        linewidth=2.2,
        label="Mean Diagonal AUROC",
    )
    ax.plot(
        summary_df["selected_pc_count"],
        summary_df["offdiag_p10_auroc"],
        marker="o",
        linewidth=2.0,
        label="10th Percentile Off-Diagonal AUROC",
    )
    ax.plot(
        summary_df["selected_pc_count"],
        summary_df["offdiag_win_rate_vs_raw"],
        marker="o",
        linewidth=2.0,
        label="Off-Diagonal Win Rate vs Raw",
    )
    ax.axhline(float(summary_df["raw_offdiag_mean_auroc"].iloc[0]), linestyle="--", linewidth=1.2, color="#495057")
    ax.axhline(float(summary_df["raw_diagonal_mean_auroc"].iloc[0]), linestyle=":", linewidth=1.2, color="#6c757d")
    ax.set_xlabel("Cumulative PC Count")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Transfer Sweep Summary vs k", fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output_path)


def save_pareto_plot(output_path: Path, summary_df: pd.DataFrame) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6.5))
    x_values = summary_df["diag_drop_vs_raw"]
    y_values = summary_df["offdiag_mean_delta_vs_raw"]
    scatter = ax.scatter(
        x_values,
        y_values,
        c=summary_df["selected_pc_count"],
        cmap="viridis",
        s=110,
        edgecolors="black",
        linewidths=0.6,
    )
    for _, row in summary_df.iterrows():
        ax.annotate(
            f"k={int(row['selected_pc_count'])}",
            (float(row["diag_drop_vs_raw"]), float(row["offdiag_mean_delta_vs_raw"])),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=9,
        )
    ax.axhline(0.0, color="#adb5bd", linewidth=1.0)
    ax.axvline(0.0, color="#adb5bd", linewidth=1.0)
    ax.set_xlabel("Diagonal AUROC Drop vs Raw")
    ax.set_ylabel("Off-Diagonal AUROC Gain vs Raw")
    ax.set_title("Projected Transfer Pareto Frontier", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.2)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Cumulative PC Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output_path)


def save_pair_delta_vs_k_heatmap(output_path: Path, pair_delta_df: pd.DataFrame) -> str:
    pair_pivot = pair_delta_df.pivot(index="pair_label", columns="selected_pc_count", values="delta_vs_raw")
    pair_pivot = pair_pivot.loc[pair_pivot.mean(axis=1).sort_values(ascending=False).index]
    limit = max(0.25, float(np.nanmax(np.abs(pair_pivot.to_numpy(dtype=np.float32)))))
    return save_heatmap(
        output_path,
        pair_pivot.to_numpy(dtype=np.float32),
        row_labels=pair_pivot.index.tolist(),
        col_labels=[str(column) for column in pair_pivot.columns.tolist()],
        title="Off-Diagonal Delta vs Raw by Pair and k",
        colorbar_label="AUROC Delta",
        vmin=-limit,
        vmax=limit,
        cmap="RdBu_r",
        center=0.0,
    )


def save_source_target_delta_plot(
    output_path: Path,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharex=True, sharey=True)
    for ax, frame, title in (
        (axes[0], source_df, "Source Off-Diagonal Delta vs k"),
        (axes[1], target_df, "Target Off-Diagonal Delta vs k"),
    ):
        for dataset_name, dataset_rows in frame.groupby("dataset_name", sort=False):
            ax.plot(
                dataset_rows["selected_pc_count"],
                dataset_rows["offdiag_mean_delta_vs_raw"],
                marker="o",
                linewidth=2.0,
                label=str(dataset_rows["dataset_label"].iloc[0]),
            )
        ax.axhline(0.0, color="#adb5bd", linewidth=1.0)
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Cumulative PC Count")
    axes[0].set_ylabel("Mean Off-Diagonal AUROC Delta vs Raw")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=min(3, len(handles)), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output_path)


def copy_baseline_heatmap(baseline_run_dir: Path, output_path: Path) -> str | None:
    candidates = sorted((baseline_run_dir / "results" / "plots").glob("*__auroc_heatmap.png"))
    if not candidates:
        return None
    ensure_dir(output_path.parent)
    shutil.copy2(candidates[0], output_path)
    return str(output_path)


def main() -> None:
    args = parse_args()
    baseline_run_dir = args.baseline_run_dir.resolve()
    pc_sweep_run_dir = args.pc_sweep_run_dir.resolve()

    baseline_manifest = read_json(baseline_run_dir / "meta" / "run_manifest.json")
    baseline_rows = load_metric_rows(baseline_run_dir)
    baseline_results = read_json(baseline_run_dir / "results" / "results.json")
    sweep_manifest = read_json(pc_sweep_run_dir / "meta" / "run_manifest.json")
    sweep_rows = load_metric_rows(pc_sweep_run_dir)
    sweep_results = read_json(pc_sweep_run_dir / "results" / "results.json")

    model_name = str(baseline_manifest.get("model_name") or "unknown-model")
    behavior_name = str(baseline_manifest.get("behavior_name") or "unknown-behavior")
    dataset_set_name = str(baseline_manifest.get("dataset_set_name") or "unknown-dataset-set")
    ordered_datasets = dataset_order(baseline_manifest, baseline_rows)
    dataset_behaviors = dataset_behavior_map(baseline_manifest)
    dataset_labels = [DATASET_LABELS.get(name, name) for name in ordered_datasets]

    baseline_lookup = baseline_metric_lookup(baseline_rows)
    expected_keys = set(baseline_lookup.keys())
    sweep_lookup_by_k = sweep_metric_lookup(sweep_rows)
    k_values = sorted(sweep_lookup_by_k.keys())
    for k, lookup in sweep_lookup_by_k.items():
        if set(lookup.keys()) != expected_keys:
            missing = sorted(expected_keys - set(lookup.keys()))
            extra = sorted(set(lookup.keys()) - expected_keys)
            raise ValueError(
                f"Sweep coverage mismatch for k={k}: missing={missing[:5]!r}, extra={extra[:5]!r}"
            )

    sweep_label = str(sweep_results.get("axis_slug") or sweep_manifest.get("axis_slug") or pc_sweep_run_dir.name)
    run_id = args.run_id or make_run_id()
    comparison_slug = normalize_slug(f"raw-vs-{sweep_label}")
    run_root = (
        args.artifact_root
        / "runs"
        / "activation-row-transfer-pc-projection-sweep-analysis"
        / normalize_slug(model_name)
        / normalize_slug(behavior_name)
        / normalize_slug(dataset_set_name)
        / comparison_slug
        / run_id
    )
    for relative in ("inputs", "results", "logs", "checkpoints", "meta"):
        ensure_dir(run_root / relative)

    write_analysis_manifest(
        run_root,
        {
            "run_id": run_id,
            "baseline_run_dir": str(baseline_run_dir),
            "pc_sweep_run_dir": str(pc_sweep_run_dir),
            "baseline_label": "raw",
            "sweep_label": sweep_label,
            "model_name": model_name,
            "behavior_name": behavior_name,
            "dataset_set_name": dataset_set_name,
            "dataset_order": ordered_datasets,
            "dataset_behavior_map": dataset_behaviors,
            "k_values": k_values,
        },
    )
    write_json(
        run_root / "inputs" / "config.json",
        {
            "baseline_run_dir": str(baseline_run_dir),
            "pc_sweep_run_dir": str(pc_sweep_run_dir),
            "k_values": k_values,
        },
    )
    write_stage_status(
        run_root,
        "analyze_activation_row_transfer_pc_projection_sweep",
        "running",
        {"k_values": k_values},
    )

    aligned_rows: list[dict[str, Any]] = []
    source_delta_rows: list[dict[str, Any]] = []
    target_delta_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    raw_diag_values = [
        float(row["auroc"])
        for row in baseline_rows
        if str(row["source_dataset"]) == str(row["target_dataset"])
    ]
    raw_offdiag_values = [
        float(row["auroc"])
        for row in baseline_rows
        if str(row["source_dataset"]) != str(row["target_dataset"])
    ]
    raw_diag_mean = float(sum(raw_diag_values) / len(raw_diag_values)) if raw_diag_values else float("nan")
    raw_offdiag_mean = float(sum(raw_offdiag_values) / len(raw_offdiag_values)) if raw_offdiag_values else float("nan")

    for k in k_values:
        sweep_lookup = sweep_lookup_by_k[k]
        per_source: dict[str, list[float]] = {}
        per_target: dict[str, list[float]] = {}
        diag_values: list[float] = []
        offdiag_values: list[float] = []
        offdiag_deltas: list[float] = []
        offdiag_wins = 0

        for layer_spec, source_dataset, target_dataset in sorted(expected_keys):
            baseline_row = baseline_lookup[(layer_spec, source_dataset, target_dataset)]
            sweep_row = sweep_lookup[(layer_spec, source_dataset, target_dataset)]
            baseline_auroc = float(baseline_row["auroc"])
            sweep_auroc = float(sweep_row["auroc"])
            delta_vs_raw = sweep_auroc - baseline_auroc
            source_behavior = dataset_behaviors.get(source_dataset)
            target_behavior = dataset_behaviors.get(target_dataset)
            is_diagonal = source_dataset == target_dataset
            pair_label = f"{DATASET_LABELS.get(source_dataset, source_dataset)}→{DATASET_LABELS.get(target_dataset, target_dataset)}"
            aligned_rows.append(
                {
                    "selected_pc_count": k,
                    "layer_spec": layer_spec,
                    "source_dataset": source_dataset,
                    "target_dataset": target_dataset,
                    "source_label": DATASET_LABELS.get(source_dataset, source_dataset),
                    "target_label": DATASET_LABELS.get(target_dataset, target_dataset),
                    "source_behavior": source_behavior,
                    "target_behavior": target_behavior,
                    "is_diagonal": is_diagonal,
                    "is_within_behavior": source_behavior is not None and source_behavior == target_behavior,
                    "is_cross_behavior": (
                        source_behavior is not None
                        and target_behavior is not None
                        and source_behavior != target_behavior
                    ),
                    "pair_label": pair_label,
                    "baseline_auroc": baseline_auroc,
                    "projected_auroc": sweep_auroc,
                    "delta_vs_raw": delta_vs_raw,
                }
            )
            if is_diagonal:
                diag_values.append(sweep_auroc)
            else:
                offdiag_values.append(sweep_auroc)
                offdiag_deltas.append(delta_vs_raw)
                if delta_vs_raw > 0.0:
                    offdiag_wins += 1
                per_source.setdefault(source_dataset, []).append(delta_vs_raw)
                per_target.setdefault(target_dataset, []).append(delta_vs_raw)

        summary_rows.append(
            {
                "selected_pc_count": k,
                "raw_diagonal_mean_auroc": raw_diag_mean,
                "raw_offdiag_mean_auroc": raw_offdiag_mean,
                "overall_mean_auroc": float(
                    np.mean([row["projected_auroc"] for row in aligned_rows if int(row["selected_pc_count"]) == k])
                ),
                "overall_mean_delta_vs_raw": float(
                    np.mean([row["delta_vs_raw"] for row in aligned_rows if int(row["selected_pc_count"]) == k])
                ),
                "diagonal_mean_auroc": float(np.mean(diag_values)) if diag_values else None,
                "diagonal_mean_delta_vs_raw": float(np.mean(diag_values) - raw_diag_mean) if diag_values else None,
                "offdiag_mean_auroc": float(np.mean(offdiag_values)) if offdiag_values else None,
                "offdiag_mean_delta_vs_raw": float(np.mean(offdiag_deltas)) if offdiag_deltas else None,
                "offdiag_p10_auroc": float(np.percentile(offdiag_values, 10)) if offdiag_values else None,
                "offdiag_p10_delta_vs_raw": float(np.percentile(offdiag_deltas, 10)) if offdiag_deltas else None,
                "offdiag_win_rate_vs_raw": float(offdiag_wins / len(offdiag_values)) if offdiag_values else None,
                "diag_drop_vs_raw": float(raw_diag_mean - np.mean(diag_values)) if diag_values else None,
            }
        )

        for dataset_name, deltas in sorted(per_source.items()):
            source_delta_rows.append(
                {
                    "selected_pc_count": k,
                    "dataset_name": dataset_name,
                    "dataset_label": DATASET_LABELS.get(dataset_name, dataset_name),
                    "offdiag_mean_delta_vs_raw": float(np.mean(deltas)),
                }
            )
        for dataset_name, deltas in sorted(per_target.items()):
            target_delta_rows.append(
                {
                    "selected_pc_count": k,
                    "dataset_name": dataset_name,
                    "dataset_label": DATASET_LABELS.get(dataset_name, dataset_name),
                    "offdiag_mean_delta_vs_raw": float(np.mean(deltas)),
                }
            )

    results_dir = run_root / "results"
    plots_dir = ensure_dir(results_dir / "plots")

    aligned_df = pd.DataFrame(aligned_rows)
    aligned_df.to_csv(results_dir / "pair_delta_by_k.csv", index=False)

    source_delta_df = pd.DataFrame(source_delta_rows)
    source_delta_df.to_csv(results_dir / "source_delta_by_k.csv", index=False)

    target_delta_df = pd.DataFrame(target_delta_rows)
    target_delta_df.to_csv(results_dir / "target_delta_by_k.csv", index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values("selected_pc_count").reset_index(drop=True)
    summary_df.to_csv(results_dir / "summary_by_k.csv", index=False)

    baseline_plot_copy = copy_baseline_heatmap(baseline_run_dir, plots_dir / "baseline__auroc_heatmap.png")

    delta_limit = max(0.25, float(np.nanmax(np.abs(aligned_df["delta_vs_raw"].to_numpy(dtype=np.float32)))))
    delta_heatmaps: list[str] = []
    dataset_index = {dataset_name: index for index, dataset_name in enumerate(ordered_datasets)}
    for k in k_values:
        matrix = np.full((len(ordered_datasets), len(ordered_datasets)), np.nan, dtype=np.float32)
        k_rows = aligned_df[aligned_df["selected_pc_count"] == k]
        for _, row in k_rows.iterrows():
            row_index = dataset_index[str(row["source_dataset"])]
            col_index = dataset_index[str(row["target_dataset"])]
            matrix[row_index, col_index] = float(row["delta_vs_raw"])
        delta_heatmaps.append(
            save_heatmap(
                plots_dir / f"k-{int(k):03d}__delta_vs_raw_heatmap.png",
                matrix,
                row_labels=dataset_labels,
                col_labels=dataset_labels,
                title=f"Projected - Raw AUROC @ k={int(k)}",
                colorbar_label="AUROC Delta",
                vmin=-delta_limit,
                vmax=delta_limit,
                cmap="RdBu_r",
                center=0.0,
            )
        )

    plot_paths = [
        save_summary_vs_k_plot(plots_dir / "summary_vs_k.png", summary_df),
        save_pareto_plot(plots_dir / "pareto_offdiag_gain_vs_diag_drop.png", summary_df),
        save_pair_delta_vs_k_heatmap(plots_dir / "pair_delta_vs_k_heatmap.png", aligned_df[~aligned_df["is_diagonal"]].copy()),
        save_source_target_delta_plot(plots_dir / "source_target_delta_vs_k.png", source_delta_df, target_delta_df),
        *delta_heatmaps,
    ]
    if baseline_plot_copy:
        plot_paths.append(baseline_plot_copy)

    write_json(
        results_dir / "results.json",
        {
            "baseline_run_dir": str(baseline_run_dir),
            "pc_sweep_run_dir": str(pc_sweep_run_dir),
            "k_values": k_values,
            "plot_paths": plot_paths,
            "summary_by_k_path": str(results_dir / "summary_by_k.csv"),
            "pair_delta_by_k_path": str(results_dir / "pair_delta_by_k.csv"),
            "source_delta_by_k_path": str(results_dir / "source_delta_by_k.csv"),
            "target_delta_by_k_path": str(results_dir / "target_delta_by_k.csv"),
        },
    )
    write_json(
        run_root / "checkpoints" / "progress.json",
        {
            "state": "completed",
            "k_values": k_values,
            "plot_paths": plot_paths,
        },
    )
    write_stage_status(
        run_root,
        "analyze_activation_row_transfer_pc_projection_sweep",
        "completed",
        {
            "k_values": k_values,
            "plot_paths": plot_paths,
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
        )
        print(
            "[activation-row-transfer-pc-projection-sweep-analysis] uploaded to "
            f"hf://datasets/{args.push_to_hub_repo_id}/{path_in_repo}"
        )

    print(
        "[activation-row-transfer-pc-projection-sweep-analysis] completed "
        f"{len(k_values)} k values; wrote outputs to {run_root}"
    )


def normalize_slug(text: str) -> str:
    from deception_honesty_axis.common import slugify

    return slugify(text)


if __name__ == "__main__":
    main()
