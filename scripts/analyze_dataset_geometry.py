#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from deception_honesty_axis.analysis import infer_activation_layer_numbers
from deception_honesty_axis.common import ensure_dir, load_jsonl, make_run_id, read_json, slugify, utc_now_iso, write_json
from deception_honesty_axis.config import ExperimentConfig, corpus_root as resolve_corpus_root, load_config
from deception_honesty_axis.indexes import load_index
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.records import load_activation_records
from deception_honesty_axis.role_axis_transfer import (
    DATASET_LABELS,
    build_role_axis_bundle,
    evaluate_zero_shot,
    load_completion_mean_split,
    load_manifest_rows,
    resolve_layer_specs,
    role_vector_for_spec,
    scores_for_layer,
)
from deception_honesty_axis.transfer_config import RoleAxisTransferConfig, load_transfer_config
from deception_honesty_axis.work_units import build_work_units, load_questions, select_first_n


PRIMARY_STAGE = "primary"
SECONDARY_STAGE = "secondary"
EXPLORATORY_STAGE = "exploratory"
SUPPORTED_STAGES = [PRIMARY_STAGE, SECONDARY_STAGE, EXPLORATORY_STAGE]
PAIRWISE_COORDS = [("contrast", "pc1"), ("contrast", "pc2"), ("pc1", "pc2")]
ACTIVE_METHOD_TO_COORD = {
    "contrast_zero_shot": ("contrast", None),
    "pc1_zero_shot": ("pc_scores", 0),
    "pc2_zero_shot": ("pc_scores", 1),
    "pc3_zero_shot": ("pc_scores", 2),
}
HONEST_COLOR = "#1f77b4"
DECEPTIVE_COLOR = "#d62728"


@dataclass(frozen=True)
class AnalysisContext:
    experiment_config: ExperimentConfig
    transfer_config: RoleAxisTransferConfig
    method: str
    layer_spec: str
    datasets: list[str]
    source_kind: str
    source_run_dir: Path | None
    selected_honest_roles: list[str]
    selected_deceptive_roles: list[str]
    selected_question_ids: list[str]
    analysis_label: str
    artifact_root: Path
    target_activations_root: Path
    eval_split: str
    source_split: str


@dataclass
class DatasetPayload:
    dataset: str
    dataset_label: str
    split_dir: Path
    sample_ids: list[str]
    labels: np.ndarray
    features: np.ndarray
    manifest_rows: dict[str, dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze dataset geometry for a role-axis probe using saved activations and optional fixed-track selections."
    )
    parser.add_argument("--search-run-dir", type=Path, default=None, help="Optional fixed-track search run directory.")
    parser.add_argument("--experiment-config", type=Path, default=None, help="Experiment config path when not using --search-run-dir.")
    parser.add_argument("--probe-config", type=Path, default=None, help="Probe config path when not using --search-run-dir.")
    parser.add_argument("--method", type=str, default=None, help="Probe method when not using --search-run-dir.")
    parser.add_argument("--layer-spec", type=str, default=None, help="Layer spec when not using --search-run-dir.")
    parser.add_argument("--dataset", action="append", default=None, help="Optional dataset filter. Can be passed multiple times.")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=[PRIMARY_STAGE, SECONDARY_STAGE],
        choices=SUPPORTED_STAGES,
        help="Analysis stages to run.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional analysis run id.")
    parser.add_argument("--resume", action="store_true", help="Skip already completed stages for this analysis run id.")
    parser.add_argument(
        "--force-stage",
        action="append",
        default=[],
        choices=SUPPORTED_STAGES,
        help="Force rerun of selected stages even if already completed.",
    )
    parser.add_argument("--no-pc2", action="store_true", help="Disable plots and metrics that depend on pc2.")
    parser.add_argument("--top-k-examples", type=int, default=10, help="Top deceptive examples per dataset to export.")
    parser.add_argument(
        "--attribution-margin-threshold",
        type=float,
        default=0.05,
        help="Minimum nearest-role margin for exported qualitative examples.",
    )
    parser.add_argument(
        "--umap-input",
        type=str,
        default="features",
        choices=["features", "scores"],
        help="Input representation when running the exploratory UMAP stage.",
    )
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--umap-random-state", type=int, default=42, help="UMAP random_state.")
    return parser.parse_args()


def load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed_stages": {}, "updated_at": None}
    return read_json(path)


def save_progress(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = utc_now_iso()
    write_json(path, payload)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def resolve_context(args: argparse.Namespace) -> AnalysisContext:
    if args.search_run_dir is not None:
        search_run_dir = args.search_run_dir.resolve()
        manifest_path = search_run_dir / "meta" / "run_manifest.json"
        selection_path = search_run_dir / "results" / "final_selection.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
        if not selection_path.exists():
            raise FileNotFoundError(f"Missing final selection: {selection_path}")

        manifest = read_json(manifest_path)
        selection = read_json(selection_path)
        selected_probe_path = search_run_dir / "selected" / "probe_config.json"
        probe_config_path = selected_probe_path if selected_probe_path.exists() else Path(manifest["probe_config_path"])
        experiment_config_path = Path(manifest["experiment_config_path"])

        experiment_config = load_config(experiment_config_path)
        transfer_config = load_transfer_config(probe_config_path)
        method = str(selection["method"])
        layer_spec = str(selection["layer_spec"])
        datasets = args.dataset or list(manifest.get("target_datasets") or transfer_config.target_datasets)
        analysis_label = slugify(
            f"{experiment_config.path.stem}__{search_run_dir.parent.name}__{search_run_dir.name}"
        )
        selected_honest_roles = [str(item) for item in selection["honest_roles"]]
        selected_deceptive_roles = [str(item) for item in selection["deceptive_roles"]]
        selected_question_ids = [str(item) for item in selection["question_ids"]]
        return AnalysisContext(
            experiment_config=experiment_config,
            transfer_config=transfer_config,
            method=method,
            layer_spec=layer_spec,
            datasets=datasets,
            source_kind="search",
            source_run_dir=search_run_dir,
            selected_honest_roles=selected_honest_roles,
            selected_deceptive_roles=selected_deceptive_roles,
            selected_question_ids=selected_question_ids,
            analysis_label=analysis_label,
            artifact_root=experiment_config.artifact_root,
            target_activations_root=transfer_config.target_activations_root,
            eval_split=transfer_config.eval_split,
            source_split=transfer_config.source_split,
        )

    if args.experiment_config is None or args.probe_config is None:
        raise ValueError("Either --search-run-dir or both --experiment-config and --probe-config are required.")

    experiment_config = load_config(args.experiment_config.resolve())
    transfer_config = load_transfer_config(args.probe_config.resolve())
    method = args.method
    layer_spec = args.layer_spec
    if method is None:
        if len(transfer_config.methods) != 1:
            raise ValueError("Specify --method when the probe config contains multiple methods.")
        method = transfer_config.methods[0]
    if layer_spec is None:
        if len(transfer_config.layer_specs) != 1:
            raise ValueError("Specify --layer-spec when the probe config contains multiple layer specs.")
        layer_spec = transfer_config.layer_specs[0]
    if method not in ACTIVE_METHOD_TO_COORD:
        raise ValueError(f"Unsupported method for geometry analysis: {method}")

    question_rows = select_first_n(
        load_questions(experiment_config.question_file_path),
        int(experiment_config.subset["question_count"]),
    )
    selected_question_ids = [row.question_id for row in question_rows]
    datasets = args.dataset or list(transfer_config.target_datasets)
    analysis_label = slugify(f"{experiment_config.path.stem}__{method}__layer-{layer_spec}")
    return AnalysisContext(
        experiment_config=experiment_config,
        transfer_config=transfer_config,
        method=method,
        layer_spec=str(layer_spec),
        datasets=datasets,
        source_kind="config",
        source_run_dir=None,
        selected_honest_roles=list(transfer_config.honest_roles),
        selected_deceptive_roles=list(transfer_config.deceptive_roles),
        selected_question_ids=selected_question_ids,
        analysis_label=analysis_label,
        artifact_root=experiment_config.artifact_root,
        target_activations_root=transfer_config.target_activations_root,
        eval_split=transfer_config.eval_split,
        source_split=transfer_config.source_split,
    )


def analysis_run_root(context: AnalysisContext, run_id: str) -> Path:
    root = (
        context.artifact_root
        / "analysis"
        / "dataset-geometry"
        / context.experiment_config.model_slug
        / context.experiment_config.dataset_slug
        / context.experiment_config.role_set_slug
        / context.analysis_label
        / run_id
    )
    for relative in ("inputs", "checkpoints", "results", "plots", "meta", "logs"):
        ensure_dir(root / relative)
    return root


def active_probe_scores(score_payload: dict[str, np.ndarray], method: str) -> np.ndarray:
    score_key, component_index = ACTIVE_METHOD_TO_COORD[method]
    if component_index is None:
        return score_payload[score_key]
    if score_payload[score_key].shape[1] <= component_index:
        raise ValueError(f"Method {method} requested component {component_index + 1}, but scores are unavailable.")
    return score_payload[score_key][:, component_index]


def question_ids_from_config(config: ExperimentConfig) -> list[str]:
    rows = select_first_n(load_questions(config.question_file_path), int(config.subset["question_count"]))
    return [row.question_id for row in rows]


def load_selected_role_question_records(context: AnalysisContext) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[int]]:
    corpus_path = resolve_corpus_root(context.experiment_config)
    work_units = build_work_units(context.experiment_config.repo_root, context.experiment_config)
    item_meta = {unit["item_id"]: unit for unit in work_units}
    allowed_roles = {
        context.transfer_config.anchor_role,
        *context.selected_honest_roles,
        *context.selected_deceptive_roles,
    }
    allowed_questions = set(context.selected_question_ids)
    target_ids = {
        unit["item_id"]
        for unit in work_units
        if unit["role"] in allowed_roles and unit["question_id"] in allowed_questions
    }
    activation_index = load_index(corpus_path, "activations")
    missing = sorted(target_ids - set(activation_index))
    if missing:
        preview = missing[:8]
        raise RuntimeError(
            f"Missing {len(missing)} activation records for selected roles/questions; first missing ids: {preview}"
        )
    selected_rows = [row for item_id, row in activation_index.items() if item_id in target_ids]
    records = load_activation_records(corpus_path, selected_rows)
    if not records:
        raise RuntimeError("No corpus activation records matched the selected roles/questions.")

    import torch

    role_question_sum: dict[tuple[str, str], torch.Tensor] = {}
    role_question_count: dict[tuple[str, str], int] = {}
    for record in records:
        meta = item_meta[record["item_id"]]
        key = (str(meta["role"]), str(meta["question_id"]))
        pooled = record["pooled_activations"].to(torch.float32)
        if key in role_question_sum:
            role_question_sum[key] += pooled
            role_question_count[key] += 1
        else:
            role_question_sum[key] = pooled.clone()
            role_question_count[key] = 1
    return role_question_sum, role_question_count, item_meta, infer_activation_layer_numbers(records)


def build_selected_bundle_and_vectors(
    context: AnalysisContext,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    role_question_sum, role_question_count, _, activation_layer_numbers = load_selected_role_question_records(context)
    example_tensor = next(iter(role_question_sum.values()))
    resolved_spec = resolve_layer_specs(int(example_tensor.shape[0]), [context.layer_spec], activation_layer_numbers)[0]
    needed_roles = [context.transfer_config.anchor_role, *context.selected_honest_roles, *context.selected_deceptive_roles]

    role_vectors: dict[str, Any] = {}
    for role_name in needed_roles:
        total_sum = None
        total_count = 0
        for question_id in context.selected_question_ids:
            key = (role_name, question_id)
            if key not in role_question_sum:
                continue
            if total_sum is None:
                total_sum = role_question_sum[key].clone()
            else:
                total_sum += role_question_sum[key]
            total_count += role_question_count[key]
        if total_sum is None or total_count == 0:
            raise RuntimeError(f"No activation coverage for role={role_name} over selected questions.")
        role_vectors[role_name] = total_sum / total_count

    bundle = build_role_axis_bundle(
        role_vectors=role_vectors,
        honest_roles=context.selected_honest_roles,
        deceptive_roles=context.selected_deceptive_roles,
        anchor_role=context.transfer_config.anchor_role,
        layer_specs=[resolved_spec.key],
        layer_numbers=activation_layer_numbers,
    )
    return bundle, role_vectors, resolved_spec


def prepare_datasets(context: AnalysisContext, resolved_spec) -> list[DatasetPayload]:  # noqa: ANN001
    payloads: list[DatasetPayload] = []
    for dataset_name in context.datasets:
        split_dir = context.target_activations_root / dataset_name / context.eval_split
        cached = load_completion_mean_split(split_dir, [resolved_spec])
        payloads.append(
            DatasetPayload(
                dataset=dataset_name,
                dataset_label=DATASET_LABELS.get(dataset_name, dataset_name),
                split_dir=split_dir,
                sample_ids=cached.sample_ids,
                labels=cached.labels,
                features=cached.features_by_spec[resolved_spec.key],
                manifest_rows=load_manifest_rows(split_dir),
            )
        )
    return payloads


def score_frame(features: np.ndarray, layer_bundle: dict[str, Any], include_pc2: bool) -> dict[str, np.ndarray]:
    payload = scores_for_layer(features, layer_bundle)
    frame = {
        "contrast": payload["contrast"].astype(np.float32),
        "pc1": payload["pc_scores"][:, 0].astype(np.float32),
    }
    if include_pc2 and payload["pc_scores"].shape[1] > 1:
        frame["pc2"] = payload["pc_scores"][:, 1].astype(np.float32)
    return frame


def role_coordinate_rows(
    context: AnalysisContext,
    layer_bundle: dict[str, Any],
    role_vectors: dict[str, Any],
    resolved_spec,  # noqa: ANN001
    include_pc2: bool,
) -> list[dict[str, Any]]:
    import torch

    contrast_axis = layer_bundle["contrast_axis"].detach().cpu().numpy().astype(np.float32)
    pca_mean = layer_bundle["pca_mean"].detach().cpu().numpy().astype(np.float32)
    pc_components = layer_bundle["pc_components"].detach().cpu().numpy().astype(np.float32)

    rows: list[dict[str, Any]] = []
    for role_name, role_tensor in role_vectors.items():
        vec = role_vector_for_spec(role_tensor, resolved_spec)
        role_vector = vec.detach().cpu().numpy().astype(np.float32)
        centered = role_vector - pca_mean
        pc_scores = centered @ pc_components.T
        role_kind = "anchor"
        if role_name in context.selected_honest_roles:
            role_kind = "honest"
        elif role_name in context.selected_deceptive_roles:
            role_kind = "deceptive"
        row = {
            "role": role_name,
            "role_kind": role_kind,
            "contrast": float(role_vector @ contrast_axis),
            "pc1": float(pc_scores[0]) if len(pc_scores) > 0 else None,
            "pc2": float(pc_scores[1]) if include_pc2 and len(pc_scores) > 1 else None,
        }
        rows.append(row)
    return rows


def euclidean_spread(points: np.ndarray, centroid: np.ndarray) -> float:
    if points.shape[0] == 0:
        return 0.0
    distances = np.linalg.norm(points - centroid[None, :], axis=1)
    return float(np.mean(distances))


def safe_correlation(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) < 2 or len(values_b) < 2:
        return None
    array_a = np.asarray(values_a, dtype=np.float64)
    array_b = np.asarray(values_b, dtype=np.float64)
    if np.allclose(array_a, array_a[0]) or np.allclose(array_b, array_b[0]):
        return None
    return float(np.corrcoef(array_a, array_b)[0, 1])


def row_metric_correlation(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float | None:
    pairs = [
        (safe_float(row.get(x_key)), safe_float(row.get(y_key)))
        for row in rows
    ]
    filtered = [(left, right) for left, right in pairs if left is not None and right is not None]
    if len(filtered) < 2:
        return None
    return safe_correlation(
        [float(left) for left, _ in filtered],
        [float(right) for _, right in filtered],
    )


def covariance_ellipse(ax, points: np.ndarray, color: str, label: str | None = None) -> None:  # noqa: ANN001
    from matplotlib.patches import Ellipse

    if points.shape[0] < 2:
        return
    covariance = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2.0 * np.sqrt(np.maximum(eigenvalues, 1e-8)) * 2.0
    centroid = points.mean(axis=0)
    ellipse = Ellipse(
        xy=(centroid[0], centroid[1]),
        width=float(width),
        height=float(height),
        angle=float(angle),
        facecolor="none",
        edgecolor=color,
        linewidth=1.6,
        alpha=0.8,
        label=label,
    )
    ax.add_patch(ellipse)


def extract_text_fields(row: dict[str, Any]) -> dict[str, Any]:
    keep_keys = [
        "prompt",
        "question",
        "question_text",
        "statement",
        "response",
        "completion",
        "completion_text",
        "text",
        "assistant_text",
        "user_text",
        "messages",
    ]
    return {key: row[key] for key in keep_keys if key in row}


def geometry_metrics_for_points(points: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    honest_points = points[labels == 0]
    deceptive_points = points[labels == 1]
    if honest_points.shape[0] == 0 or deceptive_points.shape[0] == 0:
        raise ValueError("Need both honest and deceptive samples to compute geometry metrics.")
    honest_centroid = honest_points.mean(axis=0)
    deceptive_centroid = deceptive_points.mean(axis=0)
    centroid_distance = float(np.linalg.norm(honest_centroid - deceptive_centroid))
    honest_spread = euclidean_spread(honest_points, honest_centroid)
    deceptive_spread = euclidean_spread(deceptive_points, deceptive_centroid)
    pooled_spread = float(
        (honest_spread * honest_points.shape[0] + deceptive_spread * deceptive_points.shape[0])
        / max(honest_points.shape[0] + deceptive_points.shape[0], 1)
    )
    return {
        "honest_centroid": honest_centroid.tolist(),
        "deceptive_centroid": deceptive_centroid.tolist(),
        "centroid_distance": centroid_distance,
        "honest_spread": honest_spread,
        "deceptive_spread": deceptive_spread,
        "pooled_spread": pooled_spread,
        "separation_ratio": float(centroid_distance / pooled_spread) if pooled_spread > 0 else None,
    }


def pairwise_coordinate_specs(include_pc2: bool, keys: set[str]) -> list[tuple[str, str]]:
    allowed = [("contrast", "pc1")]
    if include_pc2 and "pc2" in keys:
        allowed.extend([("contrast", "pc2"), ("pc1", "pc2")])
    return allowed


def save_pairwise_plot(
    path: Path,
    title: str,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    color_key: str,
    marker_key: str | None = None,
    role_rows: list[dict[str, Any]] | None = None,
    include_role_kinds: set[str] | None = None,
    dataset_centroid_rows: list[dict[str, Any]] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    grouped: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row[color_key]), str(row.get(marker_key)) if marker_key else None), []).append(row)

    color_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    color_lookup: dict[str, str] = {}
    marker_lookup = {"honest": "o", "deceptive": "x", None: "o"}
    for idx, color_value in enumerate(sorted({str(row[color_key]) for row in rows})):
        if color_key == "label_name":
            color_lookup[color_value] = HONEST_COLOR if color_value == "honest" else DECEPTIVE_COLOR
        else:
            color_lookup[color_value] = color_palette[idx % len(color_palette)]

    for (color_value, marker_value), members in grouped.items():
        x_values = [float(member[x_key]) for member in members]
        y_values = [float(member[y_key]) for member in members]
        marker = marker_lookup.get(marker_value, "o")
        alpha = 0.65 if marker_value == "honest" else 0.8
        ax.scatter(
            x_values,
            y_values,
            s=18,
            alpha=alpha,
            marker=marker,
            color=color_lookup[color_value],
            label=f"{color_value}" if marker_key is None else f"{color_value} / {marker_value}",
        )

    if color_key == "label_name":
        for label_name, color in [("honest", HONEST_COLOR), ("deceptive", DECEPTIVE_COLOR)]:
            label_points = np.asarray(
                [[float(row[x_key]), float(row[y_key])] for row in rows if row["label_name"] == label_name],
                dtype=np.float32,
            )
            if label_points.shape[0] > 0:
                covariance_ellipse(ax, label_points, color, None)

    if role_rows:
        for role_row in role_rows:
            if include_role_kinds and str(role_row["role_kind"]) not in include_role_kinds:
                continue
            x_value = safe_float(role_row.get(x_key))
            y_value = safe_float(role_row.get(y_key))
            if x_value is None or y_value is None:
                continue
            role_kind = str(role_row["role_kind"])
            edge_color = "#111111" if role_kind == "anchor" else ("#c92a2a" if role_kind == "deceptive" else "#1971c2")
            marker = "D" if role_kind == "anchor" else "^"
            ax.scatter([x_value], [y_value], s=95, marker=marker, color="white", edgecolor=edge_color, linewidth=1.7, zorder=5)
            ax.text(x_value, y_value, str(role_row["role"]), fontsize=8, color=edge_color, ha="left", va="bottom")

    if dataset_centroid_rows:
        for row in dataset_centroid_rows:
            x_value = safe_float(row.get(x_key))
            y_value = safe_float(row.get(y_key))
            if x_value is None or y_value is None:
                continue
            marker = "o" if row["label_name"] == "honest" else "X"
            color = color_lookup.get(str(row["dataset_label"]), "#343a40")
            ax.scatter([x_value], [y_value], s=120, marker=marker, color=color, edgecolor="#111111", linewidth=0.8, zorder=6)
            ax.text(x_value, y_value, f"{row['dataset_label']}:{row['label_name'][0].upper()}", fontsize=8, color=color, ha="left", va="bottom")

    ax.axhline(0.0, color="#adb5bd", linewidth=0.9)
    ax.axvline(0.0, color="#adb5bd", linewidth=0.9)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    ax.grid(True, color="#e9ecef", alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels, strict=False):
        dedup.setdefault(label, handle)
    if dedup:
        ax.legend(dedup.values(), dedup.keys(), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_primary_rows(
    context: AnalysisContext,
    datasets: list[DatasetPayload],
    layer_bundle: dict[str, Any],
    role_rows: list[dict[str, Any]],
    include_pc2: bool,
    top_k_examples: int,
    attribution_margin_threshold: float,
    run_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sample_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []

    role_lookup = {
        row["role"]: row
        for row in role_rows
        if row["role_kind"] == "deceptive"
    }
    role_attribution_dims = ["contrast", "pc1"] if any(row.get("pc1") is not None for row in role_rows) else ["contrast"]
    role_vectors = {
        role: np.asarray([safe_float(role_lookup[role].get(dim)) for dim in role_attribution_dims], dtype=np.float32)
        for role in role_lookup
    }

    primary_plot_dir = run_root / "plots" / PRIMARY_STAGE
    dataset_plot_dir = primary_plot_dir / "per_dataset"
    combined_rows_by_label: list[dict[str, Any]] = []
    combined_rows_by_dataset: list[dict[str, Any]] = []
    dataset_centroid_rows: list[dict[str, Any]] = []

    for payload in datasets:
        frame = score_frame(payload.features, layer_bundle, include_pc2=include_pc2)
        active_scores = active_probe_scores(
            {"contrast": frame["contrast"], "pc_scores": np.column_stack([frame["pc1"], frame.get("pc2", np.zeros_like(frame["pc1"]))])},
            context.method,
        )
        metrics, _, _ = evaluate_zero_shot(payload.labels, active_scores)
        coord_keys = [key for key in ("contrast", "pc1", "pc2") if key in frame]
        matrix = np.column_stack([frame[key] for key in coord_keys]).astype(np.float32)
        geometry = geometry_metrics_for_points(matrix, payload.labels)
        geometry_rows.append(
            {
                "stage": PRIMARY_STAGE,
                "space": "task_aligned",
                "dataset": payload.dataset,
                "dataset_label": payload.dataset_label,
                "method": context.method,
                "layer_spec": context.layer_spec,
                "sample_count": int(payload.labels.shape[0]),
                "honest_count": int(np.sum(payload.labels == 0)),
                "deceptive_count": int(np.sum(payload.labels == 1)),
                "auroc": float(metrics["auroc"]),
                "centroid_distance": geometry["centroid_distance"],
                "honest_spread": geometry["honest_spread"],
                "deceptive_spread": geometry["deceptive_spread"],
                "pooled_spread": geometry["pooled_spread"],
                "separation_ratio": geometry["separation_ratio"],
            }
        )

        honest_centroid = np.asarray(geometry["honest_centroid"], dtype=np.float32)
        deceptive_centroid = np.asarray(geometry["deceptive_centroid"], dtype=np.float32)
        centroid_by_label = {"honest": honest_centroid, "deceptive": deceptive_centroid}
        for label_name, centroid in centroid_by_label.items():
            centroid_row = {
                "dataset": payload.dataset,
                "dataset_label": payload.dataset_label,
                "label_name": label_name,
            }
            for index, key in enumerate(coord_keys):
                centroid_row[key] = float(centroid[index])
            dataset_centroid_rows.append(centroid_row)

        role_frequency: dict[str, dict[str, float]] = {}
        deceptive_examples: list[dict[str, Any]] = []
        for index, sample_id in enumerate(payload.sample_ids):
            label_value = int(payload.labels[index])
            coords = {key: float(frame[key][index]) for key in coord_keys}
            manifest_row = payload.manifest_rows.get(sample_id, {})
            attribution = {
                "nearest_deceptive_role": None,
                "second_nearest_deceptive_role": None,
                "nearest_distance": None,
                "second_nearest_distance": None,
                "nearest_margin": None,
            }
            if role_vectors:
                sample_vector = np.asarray([coords[dim] for dim in role_attribution_dims], dtype=np.float32)
                ranked = sorted(
                    (
                        (
                            role_name,
                            float(np.linalg.norm(sample_vector - role_vector)),
                        )
                        for role_name, role_vector in role_vectors.items()
                    ),
                    key=lambda item: item[1],
                )
                if ranked:
                    attribution["nearest_deceptive_role"] = ranked[0][0]
                    attribution["nearest_distance"] = ranked[0][1]
                if len(ranked) > 1:
                    attribution["second_nearest_deceptive_role"] = ranked[1][0]
                    attribution["second_nearest_distance"] = ranked[1][1]
                    attribution["nearest_margin"] = ranked[1][1] - ranked[0][1]

            row = {
                "dataset": payload.dataset,
                "dataset_label": payload.dataset_label,
                "sample_id": sample_id,
                "label": label_value,
                "label_name": "deceptive" if label_value == 1 else "honest",
                "method": context.method,
                "layer_spec": context.layer_spec,
                **coords,
                **attribution,
                **extract_text_fields(manifest_row),
            }
            sample_rows.append(row)
            combined_rows_by_label.append(row)
            combined_rows_by_dataset.append(row)
            if label_value == 1 and attribution["nearest_deceptive_role"] is not None:
                bucket = role_frequency.setdefault(
                    str(attribution["nearest_deceptive_role"]),
                    {"count": 0.0, "margin_sum": 0.0},
                )
                bucket["count"] += 1.0
                bucket["margin_sum"] += float(attribution["nearest_margin"] or 0.0)
                if safe_float(attribution["nearest_margin"]) is not None and float(attribution["nearest_margin"]) >= attribution_margin_threshold:
                    deceptive_examples.append(row)

        total_deceptive = max(1, int(np.sum(payload.labels == 1)))
        for role_name, bucket in sorted(role_frequency.items()):
            attribution_rows.append(
                {
                    "dataset": payload.dataset,
                    "dataset_label": payload.dataset_label,
                    "role": role_name,
                    "count": int(bucket["count"]),
                    "fraction_of_deceptive": float(bucket["count"] / total_deceptive),
                    "mean_margin": float(bucket["margin_sum"] / max(bucket["count"], 1.0)),
                }
            )
        for example in sorted(
            deceptive_examples,
            key=lambda item: float(item["nearest_margin"] or 0.0),
            reverse=True,
        )[:top_k_examples]:
            example_rows.append(example)

        dataset_rows = [row for row in sample_rows if row["dataset"] == payload.dataset]
        pairs = pairwise_coordinate_specs(include_pc2=include_pc2, keys=set(coord_keys))
        for x_key, y_key in pairs:
            save_pairwise_plot(
                dataset_plot_dir / slugify(payload.dataset_label) / f"{x_key}_vs_{y_key}.png",
                f"{payload.dataset_label}: {x_key} vs {y_key}",
                dataset_rows,
                x_key,
                y_key,
                color_key="label_name",
                role_rows=role_rows,
                include_role_kinds={"deceptive"},
            )

    pairs = pairwise_coordinate_specs(include_pc2=include_pc2, keys={key for row in sample_rows for key in row.keys()})
    for x_key, y_key in pairs:
        save_pairwise_plot(
            primary_plot_dir / f"all_samples_by_dataset__{x_key}_vs_{y_key}.png",
            f"All datasets by dataset: {x_key} vs {y_key}",
            combined_rows_by_dataset,
            x_key,
            y_key,
            color_key="dataset_label",
            marker_key="label_name",
            role_rows=role_rows,
            include_role_kinds={"deceptive"},
        )
        save_pairwise_plot(
            primary_plot_dir / f"all_samples_by_label__{x_key}_vs_{y_key}.png",
            f"All datasets by label: {x_key} vs {y_key}",
            combined_rows_by_label,
            x_key,
            y_key,
            color_key="label_name",
            role_rows=role_rows,
            include_role_kinds={"deceptive"},
        )
        save_pairwise_plot(
            primary_plot_dir / f"dataset_centroids_and_roles__{x_key}_vs_{y_key}.png",
            f"Dataset centroids and role points: {x_key} vs {y_key}",
            dataset_centroid_rows,
            x_key,
            y_key,
            color_key="dataset_label",
            role_rows=role_rows,
            include_role_kinds={"anchor", "honest", "deceptive"},
            dataset_centroid_rows=dataset_centroid_rows,
        )

    correlations = {
        "space": "task_aligned",
        "auroc_vs_centroid_distance": row_metric_correlation(geometry_rows, "auroc", "centroid_distance"),
        "auroc_vs_honest_spread": row_metric_correlation(geometry_rows, "auroc", "honest_spread"),
        "auroc_vs_deceptive_spread": row_metric_correlation(geometry_rows, "auroc", "deceptive_spread"),
        "auroc_vs_pooled_spread": row_metric_correlation(geometry_rows, "auroc", "pooled_spread"),
        "auroc_vs_separation_ratio": row_metric_correlation(geometry_rows, "auroc", "separation_ratio"),
    }
    return sample_rows, geometry_rows, attribution_rows, {
        "correlations": correlations,
        "role_examples": example_rows,
        "dataset_centroids": dataset_centroid_rows,
    }


def build_secondary_outputs(
    context: AnalysisContext,
    datasets: list[DatasetPayload],
    layer_bundle: dict[str, Any],
    include_pc2: bool,
    role_rows: list[dict[str, Any]],
    run_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    secondary_plot_dir = run_root / "plots" / SECONDARY_STAGE
    dataset_plot_dir = secondary_plot_dir / "per_dataset"
    matrix = np.concatenate([payload.features for payload in datasets], axis=0)
    component_count = min(3, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=component_count, svd_solver="full")
    transformed = pca.fit_transform(matrix).astype(np.float32)

    sample_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    start = 0
    for payload in datasets:
        stop = start + payload.features.shape[0]
        coords = transformed[start:stop]
        start = stop
        coord_keys = ["pca1", "pca2"] + (["pca3"] if coords.shape[1] > 2 else [])
        geometry = geometry_metrics_for_points(coords[:, : min(coords.shape[1], 3)], payload.labels)
        primary_scores = score_frame(payload.features, layer_bundle, include_pc2=include_pc2)
        active_metrics, _, _ = evaluate_zero_shot(payload.labels, active_probe_scores(
            {"contrast": primary_scores["contrast"], "pc_scores": np.column_stack([primary_scores["pc1"], primary_scores.get("pc2", np.zeros_like(primary_scores["pc1"]))])},
            context.method,
        ))
        geometry_rows.append(
            {
                "stage": SECONDARY_STAGE,
                "space": "global_pca",
                "dataset": payload.dataset,
                "dataset_label": payload.dataset_label,
                "method": context.method,
                "layer_spec": context.layer_spec,
                "sample_count": int(payload.labels.shape[0]),
                "honest_count": int(np.sum(payload.labels == 0)),
                "deceptive_count": int(np.sum(payload.labels == 1)),
                "auroc": float(active_metrics["auroc"]),
                "centroid_distance": geometry["centroid_distance"],
                "honest_spread": geometry["honest_spread"],
                "deceptive_spread": geometry["deceptive_spread"],
                "pooled_spread": geometry["pooled_spread"],
                "separation_ratio": geometry["separation_ratio"],
            }
        )

        dataset_rows: list[dict[str, Any]] = []
        for index, sample_id in enumerate(payload.sample_ids):
            row = {
                "dataset": payload.dataset,
                "dataset_label": payload.dataset_label,
                "sample_id": sample_id,
                "label": int(payload.labels[index]),
                "label_name": "deceptive" if int(payload.labels[index]) == 1 else "honest",
                "pca1": float(coords[index, 0]) if coords.shape[1] > 0 else None,
                "pca2": float(coords[index, 1]) if coords.shape[1] > 1 else None,
                "pca3": float(coords[index, 2]) if coords.shape[1] > 2 else None,
            }
            sample_rows.append(row)
            dataset_rows.append(row)

        secondary_pairs = [("pca1", "pca2")]
        if include_pc2 and coords.shape[1] > 2:
            secondary_pairs.extend([("pca1", "pca3"), ("pca2", "pca3")])
        for x_key, y_key in secondary_pairs:
            save_pairwise_plot(
                dataset_plot_dir / slugify(payload.dataset_label) / f"{x_key}_vs_{y_key}.png",
                f"{payload.dataset_label}: {x_key} vs {y_key}",
                dataset_rows,
                x_key,
                y_key,
                color_key="label_name",
            )

    secondary_pairs = [("pca1", "pca2")] + ([("pca1", "pca3"), ("pca2", "pca3")] if include_pc2 and component_count > 2 else [])
    for x_key, y_key in secondary_pairs:
        save_pairwise_plot(
            secondary_plot_dir / f"all_samples_by_dataset__{x_key}_vs_{y_key}.png",
            f"Global PCA by dataset: {x_key} vs {y_key}",
            sample_rows,
            x_key,
            y_key,
            color_key="dataset_label",
            marker_key="label_name",
        )
        save_pairwise_plot(
            secondary_plot_dir / f"all_samples_by_label__{x_key}_vs_{y_key}.png",
            f"Global PCA by label: {x_key} vs {y_key}",
            sample_rows,
            x_key,
            y_key,
            color_key="label_name",
        )

    correlations = {
        "space": "global_pca",
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "auroc_vs_centroid_distance": row_metric_correlation(geometry_rows, "auroc", "centroid_distance"),
        "auroc_vs_honest_spread": row_metric_correlation(geometry_rows, "auroc", "honest_spread"),
        "auroc_vs_deceptive_spread": row_metric_correlation(geometry_rows, "auroc", "deceptive_spread"),
        "auroc_vs_pooled_spread": row_metric_correlation(geometry_rows, "auroc", "pooled_spread"),
        "auroc_vs_separation_ratio": row_metric_correlation(geometry_rows, "auroc", "separation_ratio"),
    }
    return sample_rows, geometry_rows, correlations


def build_exploratory_outputs(
    context: AnalysisContext,
    datasets: list[DatasetPayload],
    layer_bundle: dict[str, Any],
    include_pc2: bool,
    role_rows: list[dict[str, Any]],
    run_root: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    try:
        import umap
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("UMAP stage requested, but the `umap-learn` package is not installed.") from exc

    rows: list[dict[str, Any]] = []
    matrices: list[np.ndarray] = []
    for payload in datasets:
        frame = score_frame(payload.features, layer_bundle, include_pc2=include_pc2)
        if args.umap_input == "scores":
            coord_keys = [key for key in ("contrast", "pc1", "pc2") if key in frame]
            matrix = np.column_stack([frame[key] for key in coord_keys]).astype(np.float32)
        else:
            matrix = payload.features.astype(np.float32)
        matrices.append(matrix)
    combined = np.concatenate(matrices, axis=0)
    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        random_state=args.umap_random_state,
    )
    embedding = reducer.fit_transform(combined).astype(np.float32)
    start = 0
    for payload, matrix in zip(datasets, matrices, strict=False):
        stop = start + matrix.shape[0]
        coords = embedding[start:stop]
        start = stop
        for index, sample_id in enumerate(payload.sample_ids):
            rows.append(
                {
                    "dataset": payload.dataset,
                    "dataset_label": payload.dataset_label,
                    "sample_id": sample_id,
                    "label": int(payload.labels[index]),
                    "label_name": "deceptive" if int(payload.labels[index]) == 1 else "honest",
                    "umap1": float(coords[index, 0]),
                    "umap2": float(coords[index, 1]),
                }
            )

    exploratory_plot_dir = run_root / "plots" / EXPLORATORY_STAGE
    save_pairwise_plot(
        exploratory_plot_dir / "all_samples_by_dataset__umap1_vs_umap2.png",
        f"UMAP by dataset ({args.umap_input})",
        rows,
        "umap1",
        "umap2",
        color_key="dataset_label",
        marker_key="label_name",
    )
    save_pairwise_plot(
        exploratory_plot_dir / "all_samples_by_label__umap1_vs_umap2.png",
        f"UMAP by label ({args.umap_input})",
        rows,
        "umap1",
        "umap2",
        color_key="label_name",
    )
    return rows


def stage_is_complete(progress: dict[str, Any], stage: str) -> bool:
    return bool(progress.get("completed_stages", {}).get(stage))


def mark_stage_complete(progress_path: Path, progress: dict[str, Any], stage: str) -> None:
    progress.setdefault("completed_stages", {})[stage] = {
        "completed": True,
        "completed_at": utc_now_iso(),
    }
    save_progress(progress_path, progress)


def main() -> None:
    args = parse_args()
    context = resolve_context(args)
    run_id = args.run_id or make_run_id()
    run_root = analysis_run_root(context, run_id)
    progress_path = run_root / "checkpoints" / "progress.json"
    progress = load_progress(progress_path)
    force_stages = set(args.force_stage)
    include_pc2 = not args.no_pc2

    write_analysis_manifest(
        run_root,
        {
            "experiment_config_path": str(context.experiment_config.path),
            "probe_config_path": str(context.transfer_config.path),
            "source_kind": context.source_kind,
            "source_run_dir": str(context.source_run_dir) if context.source_run_dir else None,
            "method": context.method,
            "layer_spec": context.layer_spec,
            "datasets": context.datasets,
            "stages": args.stages,
            "selected_honest_roles": context.selected_honest_roles,
            "selected_deceptive_roles": context.selected_deceptive_roles,
            "selected_question_count": len(context.selected_question_ids),
            "target_activations_root": str(context.target_activations_root),
            "run_id": run_id,
        },
    )
    write_json(run_root / "inputs" / "experiment_config.json", context.experiment_config.raw)
    write_json(run_root / "inputs" / "probe_config.json", context.transfer_config.raw)
    if context.source_run_dir is not None:
        write_json(
            run_root / "inputs" / "source_run_reference.json",
            {
                "source_run_dir": str(context.source_run_dir),
                "final_selection": read_json(context.source_run_dir / "results" / "final_selection.json"),
                "run_manifest": read_json(context.source_run_dir / "meta" / "run_manifest.json"),
            },
        )

    write_stage_status(
        run_root,
        "analyze_dataset_geometry",
        "running",
        {
            "stages": args.stages,
            "method": context.method,
            "layer_spec": context.layer_spec,
            "dataset_count": len(context.datasets),
        },
    )
    save_progress(progress_path, progress)

    bundle, role_vectors, resolved_spec = build_selected_bundle_and_vectors(context)
    layer_bundle = bundle["layers"][resolved_spec.key]
    role_rows = role_coordinate_rows(context, layer_bundle, role_vectors, resolved_spec, include_pc2=include_pc2)
    datasets = prepare_datasets(context, resolved_spec)

    write_csv(
        run_root / "results" / "role_coordinates.csv",
        ["role", "role_kind", "contrast", "pc1", "pc2"],
        role_rows,
    )
    write_json(
        run_root / "results" / "selection_summary.json",
        {
            "selected_honest_roles": context.selected_honest_roles,
            "selected_deceptive_roles": context.selected_deceptive_roles,
            "selected_question_ids": context.selected_question_ids,
            "selected_question_count": len(context.selected_question_ids),
        },
    )

    correlations_payload: dict[str, Any] = {}
    for stage in args.stages:
        if args.resume and stage not in force_stages and stage_is_complete(progress, stage):
            print(f"[dataset-geometry] skipping completed stage={stage}")
            continue

        if stage == PRIMARY_STAGE:
            sample_rows, geometry_rows, attribution_rows, extras = build_primary_rows(
                context=context,
                datasets=datasets,
                layer_bundle=layer_bundle,
                role_rows=role_rows,
                include_pc2=include_pc2,
                top_k_examples=args.top_k_examples,
                attribution_margin_threshold=args.attribution_margin_threshold,
                run_root=run_root,
            )
            write_jsonl(run_root / "results" / "per_sample_scores.jsonl", sample_rows)
            if geometry_rows:
                write_csv(run_root / "results" / "per_dataset_geometry.csv", list(geometry_rows[0].keys()), geometry_rows)
            if attribution_rows:
                write_csv(
                    run_root / "results" / "per_dataset_role_attribution.csv",
                    list(attribution_rows[0].keys()),
                    attribution_rows,
                )
            write_jsonl(run_root / "results" / "nearest_role_examples.jsonl", extras["role_examples"])
            write_json(run_root / "results" / "dataset_centroids_primary.json", {"rows": extras["dataset_centroids"]})
            correlations_payload[PRIMARY_STAGE] = extras["correlations"]

        elif stage == SECONDARY_STAGE:
            sample_rows, geometry_rows, correlations = build_secondary_outputs(
                context=context,
                datasets=datasets,
                layer_bundle=layer_bundle,
                include_pc2=include_pc2,
                role_rows=role_rows,
                run_root=run_root,
            )
            write_jsonl(run_root / "results" / "per_sample_global_pca.jsonl", sample_rows)
            if geometry_rows:
                write_csv(
                    run_root / "results" / "per_dataset_geometry_pca.csv",
                    list(geometry_rows[0].keys()),
                    geometry_rows,
                )
            correlations_payload[SECONDARY_STAGE] = correlations

        elif stage == EXPLORATORY_STAGE:
            rows = build_exploratory_outputs(
                context=context,
                datasets=datasets,
                layer_bundle=layer_bundle,
                include_pc2=include_pc2,
                role_rows=role_rows,
                run_root=run_root,
                args=args,
            )
            write_jsonl(run_root / "results" / "per_sample_umap.jsonl", rows)
            correlations_payload[EXPLORATORY_STAGE] = {"umap_input": args.umap_input}

        mark_stage_complete(progress_path, progress, stage)
        write_stage_status(
            run_root,
            f"analyze_dataset_geometry_{stage}",
            "completed",
            {"stage": stage},
        )
        print(f"[dataset-geometry] completed stage={stage}")

    write_json(run_root / "results" / "correlations.json", correlations_payload)
    write_stage_status(
        run_root,
        "analyze_dataset_geometry",
        "completed",
        {
            "stages": args.stages,
            "method": context.method,
            "layer_spec": context.layer_spec,
            "dataset_count": len(context.datasets),
            "results_dir": str(run_root / "results"),
            "plots_dir": str(run_root / "plots"),
        },
    )
    print(f"[dataset-geometry] wrote analysis artifacts to {run_root}")


if __name__ == "__main__":
    main()
