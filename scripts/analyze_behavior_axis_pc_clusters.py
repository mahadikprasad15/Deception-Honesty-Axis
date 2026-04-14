#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from deception_honesty_axis.common import ensure_dir, make_run_id, read_json, slugify, write_json
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.role_axis_transfer import DATASET_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster PCs from a behavior-axis PC sweep run using full dataset profiles and summary features."
    )
    parser.add_argument("--pc-sweep-run-dir", type=Path, required=True, help="Run dir produced by evaluate_behavior_axis_pc_sweep.py.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id.")
    parser.add_argument("--profile-clusters", type=int, default=6, help="Number of clusters for full signed-profile clustering.")
    parser.add_argument("--summary-clusters", type=int, default=6, help="Number of clusters for summary-feature clustering.")
    parser.add_argument("--push-to-hub-repo-id", default=None, help="Optional HF dataset repo to upload this run folder into.")
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Optional destination path in the HF artifacts repo. Defaults to the run path under artifacts/.",
    )
    return parser.parse_args()


def _dataset_label(dataset_name: str) -> str:
    return DATASET_LABELS.get(dataset_name, dataset_name)


def _safe_cluster_count(requested: int, item_count: int) -> int:
    if item_count < 2:
        return 1
    return max(2, min(int(requested), item_count))


def _cluster_labels(matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    cluster_count = _safe_cluster_count(n_clusters, int(matrix.shape[0]))
    if cluster_count <= 1:
        return np.zeros(matrix.shape[0], dtype=int)
    model = AgglomerativeClustering(n_clusters=cluster_count, linkage="ward")
    return model.fit_predict(matrix)


def _order_pc_frame(frame: pd.DataFrame, cluster_column: str, primary_columns: list[str]) -> pd.DataFrame:
    sort_columns = [cluster_column] + primary_columns + ["pc_index"]
    ascending = [True] + [False] * len(primary_columns) + [True]
    return frame.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def _profile_matrix(metric_rows: pd.DataFrame, dataset_order: list[str]) -> pd.DataFrame:
    matrix = metric_rows.pivot(index="pc_label", columns="target_dataset", values="auroc")
    matrix = matrix.loc[:, dataset_order]
    matrix.columns = [_dataset_label(name) for name in matrix.columns]
    return matrix


def _summary_feature_frame(metric_rows: pd.DataFrame, dataset_behavior_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pc_label, pc_rows in metric_rows.groupby("pc_label", sort=False):
        pc_index = int(pc_rows["pc_index"].iloc[0])
        auroc = pd.to_numeric(pc_rows["auroc"], errors="coerce")
        signed = auroc - 0.5
        deception = signed[pc_rows["target_behavior"] == "deception"]
        sycophancy = signed[pc_rows["target_behavior"] == "sycophancy"]
        rows.append(
            {
                "pc_label": pc_label,
                "pc_index": pc_index,
                "mean_all_signed": float(signed.mean()),
                "mean_deception_signed": float(deception.mean()) if not deception.empty else np.nan,
                "mean_sycophancy_signed": float(sycophancy.mean()) if not sycophancy.empty else np.nan,
                "gap_deception_minus_sycophancy": (
                    float(deception.mean() - sycophancy.mean())
                    if not deception.empty and not sycophancy.empty
                    else np.nan
                ),
                "std_all_signed": float(signed.std(ddof=0)),
                "std_deception_signed": float(deception.std(ddof=0)) if not deception.empty else np.nan,
                "std_sycophancy_signed": float(sycophancy.std(ddof=0)) if not sycophancy.empty else np.nan,
                "max_abs_signed": float(np.abs(signed).max()),
            }
        )
    frame = pd.DataFrame(rows)
    feature_columns = [
        "mean_all_signed",
        "mean_deception_signed",
        "mean_sycophancy_signed",
        "gap_deception_minus_sycophancy",
        "std_all_signed",
        "std_deception_signed",
        "std_sycophancy_signed",
        "max_abs_signed",
    ]
    frame[feature_columns] = frame[feature_columns].fillna(0.0)
    return frame


def _save_heatmap(
    output_path: Path,
    matrix: pd.DataFrame,
    *,
    title: str,
    center: float | None,
    cmap: str,
    cbar_label: str,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(max(10, matrix.shape[1] * 0.45), max(6, matrix.shape[0] * 0.32)))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        center=center,
        annot=True,
        fmt=".02f",
        linewidths=0.5,
        cbar_kws={"label": cbar_label},
    )
    ax.set_title(title)
    ax.set_xlabel(matrix.columns.name or "")
    ax.set_ylabel(matrix.index.name or "")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_cluster_scatter(
    output_path: Path,
    frame: pd.DataFrame,
    *,
    cluster_column: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    sns.scatterplot(
        data=frame,
        x="mean_deception_signed",
        y="mean_sycophancy_signed",
        hue=cluster_column,
        palette="tab10",
        s=120,
        edgecolor="black",
        linewidth=0.6,
        ax=ax,
    )
    bounds = [
        float(frame["mean_deception_signed"].min()),
        float(frame["mean_deception_signed"].max()),
        float(frame["mean_sycophancy_signed"].min()),
        float(frame["mean_sycophancy_signed"].max()),
        -0.05,
        0.05,
    ]
    min_bound = min(bounds) - 0.03
    max_bound = max(bounds) + 0.03
    ax.plot([min_bound, max_bound], [min_bound, max_bound], linestyle="--", color="#6c757d", linewidth=1.5)
    ax.axhline(0.0, color="#adb5bd", linewidth=1.0)
    ax.axvline(0.0, color="#adb5bd", linewidth=1.0)
    ax.set_xlim(min_bound, max_bound)
    ax.set_ylim(min_bound, max_bound)
    ax.set_xlabel("Mean Deception Signed AUROC (AUROC - 0.5)")
    ax.set_ylabel("Mean Sycophancy Signed AUROC (AUROC - 0.5)")
    ax.set_title(title)
    top_labels = set(frame.sort_values("max_abs_signed", ascending=False).head(min(12, len(frame)))["pc_label"].tolist())
    for _, row in frame.iterrows():
        if row["pc_label"] in top_labels:
            ax.annotate(
                row["pc_label"],
                (row["mean_deception_signed"], row["mean_sycophancy_signed"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=9,
            )
    legend = ax.legend(title="Cluster", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    if legend is not None:
        legend.get_frame().set_alpha(0.95)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_embedding_scatter(
    output_path: Path,
    embedding: np.ndarray,
    frame: pd.DataFrame,
    *,
    cluster_column: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = frame.copy()
    plot_df["embed_x"] = embedding[:, 0]
    plot_df["embed_y"] = embedding[:, 1]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    sns.scatterplot(
        data=plot_df,
        x="embed_x",
        y="embed_y",
        hue=cluster_column,
        palette="tab10",
        s=120,
        edgecolor="black",
        linewidth=0.6,
        ax=ax,
    )
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    ax.set_title(title)
    top_labels = set(plot_df.sort_values("max_abs_signed", ascending=False).head(min(12, len(plot_df)))["pc_label"].tolist())
    for _, row in plot_df.iterrows():
        if row["pc_label"] in top_labels:
            ax.annotate(
                row["pc_label"],
                (row["embed_x"], row["embed_y"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=9,
            )
    legend = ax.legend(title="Cluster", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    if legend is not None:
        legend.get_frame().set_alpha(0.95)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pc_sweep_run_dir = args.pc_sweep_run_dir.resolve()
    results_dir = pc_sweep_run_dir / "results"
    metrics_path = results_dir / "pairwise_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing pairwise_metrics.csv in {results_dir}")

    run_results = read_json(results_dir / "results.json")
    manifest = read_json(pc_sweep_run_dir / "meta" / "run_manifest.json")
    axis_slug = str(run_results.get("axis_slug") or manifest.get("axis_slug") or pc_sweep_run_dir.name)
    datasets = [str(name) for name in run_results["datasets"]]
    dataset_behaviors = manifest.get("dataset_behaviors", {})

    artifact_root = next(parent for parent in pc_sweep_run_dir.parents if parent.name == "artifacts")
    run_id = args.run_id or make_run_id()
    output_root = (
        artifact_root
        / "runs"
        / "behavior-axis-pc-clusters"
        / pc_sweep_run_dir.parts[-5]
        / pc_sweep_run_dir.parts[-4]
        / pc_sweep_run_dir.parts[-3]
        / axis_slug
        / run_id
    )
    for relative in ("inputs", "results", "logs", "checkpoints", "meta"):
        ensure_dir(output_root / relative)

    write_analysis_manifest(
        output_root,
        {
            "pc_sweep_run_dir": str(pc_sweep_run_dir),
            "run_id": run_id,
            "axis_slug": axis_slug,
            "profile_clusters": args.profile_clusters,
            "summary_clusters": args.summary_clusters,
            "datasets": datasets,
            "dataset_behaviors": dataset_behaviors,
        },
    )
    write_json(
        output_root / "inputs" / "config.json",
        {
            "pc_sweep_run_dir": str(pc_sweep_run_dir),
            "profile_clusters": args.profile_clusters,
            "summary_clusters": args.summary_clusters,
        },
    )
    write_stage_status(
        output_root,
        "analyze_behavior_axis_pc_clusters",
        "running",
        {
            "pc_sweep_run_dir": str(pc_sweep_run_dir),
            "profile_clusters": args.profile_clusters,
            "summary_clusters": args.summary_clusters,
        },
    )

    metric_rows = pd.read_csv(metrics_path)
    metric_rows["pc_index"] = metric_rows["pc_index"].astype(int)
    metric_rows["auroc"] = pd.to_numeric(metric_rows["auroc"], errors="coerce")
    metric_rows["signed_auroc"] = metric_rows["auroc"] - 0.5

    available_layers = sorted(metric_rows["layer_spec"].astype(str).unique().tolist())
    plot_artifacts: dict[str, dict[str, str]] = {}

    for layer_spec in available_layers:
        layer_rows = metric_rows[metric_rows["layer_spec"].astype(str) == str(layer_spec)].copy()
        profile_df = _profile_matrix(layer_rows, datasets)
        signed_profile_df = profile_df - 0.5

        profile_cluster_labels = _cluster_labels(signed_profile_df.to_numpy(dtype=np.float32), args.profile_clusters)
        summary_frame = _summary_feature_frame(layer_rows, dataset_behaviors)
        summary_feature_columns = [
            "mean_all_signed",
            "mean_deception_signed",
            "mean_sycophancy_signed",
            "gap_deception_minus_sycophancy",
            "std_all_signed",
            "std_deception_signed",
            "std_sycophancy_signed",
            "max_abs_signed",
        ]
        scaled_summary = StandardScaler().fit_transform(summary_frame[summary_feature_columns].to_numpy(dtype=np.float32))
        scaled_summary_frame = summary_frame[["pc_label", "pc_index"]].copy()
        for column_index, column_name in enumerate(summary_feature_columns):
            scaled_summary_frame[column_name] = scaled_summary[:, column_index]
        summary_cluster_labels = _cluster_labels(scaled_summary, args.summary_clusters)

        summary_frame["profile_cluster"] = profile_cluster_labels
        summary_frame["summary_cluster"] = summary_cluster_labels
        scaled_summary_frame["profile_cluster"] = profile_cluster_labels
        scaled_summary_frame["summary_cluster"] = summary_cluster_labels

        ordered_profile = _order_pc_frame(summary_frame, "profile_cluster", ["max_abs_signed", "mean_all_signed"])
        ordered_summary = _order_pc_frame(summary_frame, "summary_cluster", ["max_abs_signed", "mean_all_signed"])
        ordered_summary_scaled = (
            ordered_summary[["pc_label", "pc_index", "profile_cluster", "summary_cluster"]]
            .merge(scaled_summary_frame, on=["pc_label", "pc_index", "profile_cluster", "summary_cluster"], how="left")
        )

        profile_order_labels = ordered_profile["pc_label"].tolist()
        summary_order_labels = ordered_summary["pc_label"].tolist()
        profile_heatmap = signed_profile_df.loc[profile_order_labels].copy()
        profile_heatmap.index.name = "PC"
        profile_heatmap.columns.name = "Dataset"

        summary_heatmap = (
            ordered_summary_scaled.set_index("pc_label")[summary_feature_columns].copy()
        )
        summary_heatmap.index.name = "PC"
        summary_heatmap.columns.name = "Summary Feature"

        plots_dir = output_root / "results" / "plots"
        ensure_dir(plots_dir)

        profile_heatmap_path = plots_dir / f"pc_profile_clustered_heatmap__{layer_spec}.png"
        _save_heatmap(
            profile_heatmap_path,
            profile_heatmap,
            title=f"PC Clusters From Full Dataset Profiles @ Layer {layer_spec}",
            center=0.0,
            cmap="RdBu_r",
            cbar_label="Signed AUROC (AUROC - 0.5)",
        )

        summary_heatmap_path = plots_dir / f"pc_summary_clustered_heatmap__{layer_spec}.png"
        _save_heatmap(
            summary_heatmap_path,
            summary_heatmap,
            title=f"PC Clusters From Behavior Summary Features @ Layer {layer_spec}",
            center=0.0,
            cmap="RdBu_r",
            cbar_label="Standardized Summary Feature",
        )

        profile_scatter_path = plots_dir / f"pc_profile_cluster_scatter__{layer_spec}.png"
        _save_cluster_scatter(
            profile_scatter_path,
            ordered_profile,
            cluster_column="profile_cluster",
            title=f"PC Clusters Colored by Full-Profile Cluster @ Layer {layer_spec}",
        )

        summary_scatter_path = plots_dir / f"pc_summary_cluster_scatter__{layer_spec}.png"
        _save_cluster_scatter(
            summary_scatter_path,
            ordered_summary,
            cluster_column="summary_cluster",
            title=f"PC Clusters Colored by Summary-Feature Cluster @ Layer {layer_spec}",
        )

        profile_embedding = PCA(n_components=2, random_state=0).fit_transform(
            signed_profile_df.loc[ordered_profile["pc_label"]].to_numpy(dtype=np.float32)
        )
        profile_embedding_path = plots_dir / f"pc_profile_embedding__{layer_spec}.png"
        _save_embedding_scatter(
            profile_embedding_path,
            profile_embedding,
            ordered_profile,
            cluster_column="profile_cluster",
            title=f"PC Embedding From Full Dataset Profiles @ Layer {layer_spec}",
        )

        summary_embedding = PCA(n_components=2, random_state=0).fit_transform(
            ordered_summary_scaled[summary_feature_columns].to_numpy(dtype=np.float32)
        )
        summary_embedding_path = plots_dir / f"pc_summary_embedding__{layer_spec}.png"
        _save_embedding_scatter(
            summary_embedding_path,
            summary_embedding,
            ordered_summary,
            cluster_column="summary_cluster",
            title=f"PC Embedding From Summary Features @ Layer {layer_spec}",
        )

        profile_assignments = ordered_profile.copy()
        profile_assignments.to_csv(output_root / "results" / f"pc_clusters__{layer_spec}.csv", index=False)
        profile_assignments.to_json(output_root / "results" / f"pc_clusters__{layer_spec}.json", orient="records", indent=2)

        plot_artifacts[layer_spec] = {
            "pc_profile_clustered_heatmap": str(profile_heatmap_path),
            "pc_summary_clustered_heatmap": str(summary_heatmap_path),
            "pc_profile_cluster_scatter": str(profile_scatter_path),
            "pc_summary_cluster_scatter": str(summary_scatter_path),
            "pc_profile_embedding": str(profile_embedding_path),
            "pc_summary_embedding": str(summary_embedding_path),
            "cluster_assignments_csv": str(output_root / "results" / f"pc_clusters__{layer_spec}.csv"),
        }

    write_json(
        output_root / "results" / "results.json",
        {
            "pc_sweep_run_dir": str(pc_sweep_run_dir),
            "axis_slug": axis_slug,
            "datasets": datasets,
            "profile_clusters": args.profile_clusters,
            "summary_clusters": args.summary_clusters,
            "plot_artifacts": plot_artifacts,
        },
    )
    write_stage_status(
        output_root,
        "analyze_behavior_axis_pc_clusters",
        "completed",
        {
            "pc_sweep_run_dir": str(pc_sweep_run_dir),
            "axis_slug": axis_slug,
            "plot_artifacts": plot_artifacts,
        },
    )

    if args.push_to_hub_repo_id:
        from huggingface_hub import upload_folder

        default_path_in_repo = str(output_root.relative_to(artifact_root))
        path_in_repo = args.path_in_repo or default_path_in_repo
        upload_folder(
            repo_id=args.push_to_hub_repo_id,
            repo_type="dataset",
            folder_path=str(output_root),
            path_in_repo=path_in_repo,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"[behavior-axis-pc-clusters] uploaded to hf://datasets/{args.push_to_hub_repo_id}/{path_in_repo}")

    print(f"[behavior-axis-pc-clusters] completed; wrote results to {output_root}")


if __name__ == "__main__":
    main()
