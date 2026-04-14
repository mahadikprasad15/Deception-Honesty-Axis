#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from deception_honesty_axis.activation_row_targets import (
    ensure_pooling_compatibility,
    ensure_single_layer_compatibility,
    grouped_indices,
    load_activation_rows,
    validate_and_stack_activation_rows,
)
from deception_honesty_axis.common import make_run_id, read_json, slugify, utc_now_iso, write_json
from deception_honesty_axis.analysis import infer_activation_layer_numbers
from deception_honesty_axis.config import corpus_root as resolve_corpus_root, load_config
from deception_honesty_axis.indexes import load_index
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.records import load_activation_records
from deception_honesty_axis.role_axis_transfer import (
    CachedSplitFeatures,
    DATASET_LABELS,
    ZERO_SHOT_METHODS,
    build_role_axis_bundle,
    evaluate_zero_shot,
    load_completion_mean_split,
    resolve_layer_specs,
    scores_for_layer,
)
from deception_honesty_axis.transfer_config import load_transfer_config
from deception_honesty_axis.work_units import build_work_units, load_questions, select_first_n


SUPPORTED_FIXED_METHODS = set(ZERO_SHOT_METHODS)
ZERO_SHOT_METHOD_COMPONENT_INDEX = {
    "contrast_zero_shot": None,
    "pc1_zero_shot": 0,
    "pc2_zero_shot": 1,
    "pc3_zero_shot": 2,
}
ACTIVATION_TARGET_LABELS = {
    "all": "All",
    "factual": "Factual",
    "evaluative": "Evaluative",
    "aesthetic": "Aesthetic",
}
DEFAULT_DATASET_ORDER = [
    "Deception-AILiar-completion",
    "Deception-ConvincingGame-completion",
    "Deception-HarmPressureChoice-completion",
    "Deception-InstructedDeception-completion",
    "Deception-Mask-completion",
    "Deception-Roleplaying-completion",
    "Deception-ClaimsDefinitional-completion",
    "Deception-ClaimsEvidential-completion",
    "Deception-ClaimsFictional-completion",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed-probe, fixed-layer greedy pruning search over role/question subsets."
    )
    parser.add_argument("--experiment-config", type=Path, required=True, help="Path to the experiment config.")
    parser.add_argument("--probe-config", type=Path, required=True, help="Path to the transfer config.")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=sorted(SUPPORTED_FIXED_METHODS),
        help="Fixed zero-shot method to use for every dataset in the run.",
    )
    parser.add_argument("--layer-spec", type=str, default="14", help="Fixed layer spec to use for every dataset.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id.")
    parser.add_argument("--min-honest", type=int, default=4, help="Minimum honest roles to keep.")
    parser.add_argument("--min-deceptive", type=int, default=4, help="Minimum deceptive roles to keep.")
    parser.add_argument("--min-questions", type=int, default=10, help="Minimum questions to keep.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress after every N candidate states evaluated within a step.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from an existing checkpoint for this run id.")
    parser.add_argument(
        "--target-dataset",
        action="append",
        default=None,
        help="Optional override to evaluate only selected target datasets. Can be passed multiple times.",
    )
    parser.add_argument(
        "--activation-dataset-repo-id",
        default=None,
        help="Optional HF Dataset repo with cached 1D activation rows. If set, uses activation-row targets instead of completion splits.",
    )
    parser.add_argument(
        "--activation-dataset-dir",
        type=Path,
        default=None,
        help="Optional local datasets.save_to_disk directory for cached activation rows.",
    )
    parser.add_argument(
        "--activation-jsonl",
        type=Path,
        default=None,
        help="Optional local records.jsonl for cached activation rows.",
    )
    parser.add_argument(
        "--activation-split",
        default="train",
        help="HF split to load when using --activation-dataset-repo-id.",
    )
    parser.add_argument(
        "--activation-group-field",
        default="dataset_source",
        help="Activation-row field used to define evaluation groups. 'all' is always added automatically.",
    )
    parser.add_argument(
        "--target-slug",
        default=None,
        help="Optional suffix used to distinguish non-default evaluation targets in the run path.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="mean_delta_first",
        choices=["mean_delta_first", "threshold_first"],
        help="Objective ordering for greedy selection.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Threshold used by the threshold_first objective.",
    )
    return parser.parse_args()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def objective_slug(objective: str, threshold: float) -> str:
    if objective == "threshold_first":
        return f"threshold-first__thr-{threshold:.2f}".replace(".", "p")
    return "mean-delta-first"


def has_activation_row_source(args: argparse.Namespace) -> bool:
    return bool(args.activation_dataset_repo_id or args.activation_dataset_dir or args.activation_jsonl)


def evaluation_target_slug(args: argparse.Namespace) -> str:
    if args.target_slug:
        return slugify(args.target_slug)
    if not has_activation_row_source(args):
        return "transfer"
    source_token = args.activation_dataset_repo_id
    if not source_token and args.activation_dataset_dir:
        source_token = args.activation_dataset_dir.resolve().name
    if not source_token and args.activation_jsonl:
        source_token = args.activation_jsonl.resolve().stem
    source_token = source_token or "activation-rows"
    return slugify(f"activation-rows-{args.activation_group_field}-{source_token}")


def fixed_track_run_root(
    experiment_config,
    method: str,
    layer_spec: str,
    run_id: str,
    objective: str,
    threshold: float,
    target_slug: str,
) -> Path:  # noqa: ANN001
    track_slug = f"{method}__layer-{layer_spec}__objective-{objective_slug(objective, threshold)}"
    if target_slug != "transfer":
        track_slug = f"{track_slug}__target-{target_slug}"
    root = (
        experiment_config.artifact_root
        / "runs"
        / "fixed-track-search"
        / experiment_config.model_slug
        / experiment_config.dataset_slug
        / experiment_config.role_set_slug
        / track_slug
        / run_id
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def axis_expected_pooling(experiment_config) -> str | None:  # noqa: ANN001
    return str(experiment_config.analysis.get("pooling") or "")


def target_dataset_label(dataset_name: str) -> str:
    return ACTIVATION_TARGET_LABELS.get(dataset_name, DATASET_LABELS.get(dataset_name, dataset_name))


def score_tuple(score: dict[str, float | int], objective: str) -> tuple[float | int, ...]:
    if objective == "threshold_first":
        return (
            int(score["count_gt_threshold"]),
            int(score["count_ge_threshold"]),
            float(score["clipped_threshold_sum"]),
            int(score["improved_count"]),
            int(score["nonnegative_count"]),
            float(score["mean_auroc"]),
            float(score["mean_delta"]),
        )
    return (
        int(score["improved_count"]),
        int(score["nonnegative_count"]),
        float(score["mean_delta"]),
        float(score["sum_delta"]),
        float(score["mean_auroc"]),
    )


def save_selected_questions(path: Path, selected_ids: list[str], question_rows: list[Any]) -> None:
    selected_set = set(selected_ids)
    ordered_rows = [row for row in question_rows if row.question_id in selected_set]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in ordered_rows:
            handle.write(
                json.dumps(
                    {
                        "question_id": row.question_id,
                        "question": row.question,
                        **{
                            key: value
                            for key, value in row.metadata.items()
                            if key not in {"question_id", "question"}
                        },
                    },
                    ensure_ascii=True,
                )
            )
            handle.write("\n")


def save_selected_role_manifest(
    path: Path,
    role_manifest: dict[str, Any],
    honest_roles: list[str],
    deceptive_roles: list[str],
) -> None:
    source_by_role = {entry["name"]: entry["source"] for entry in role_manifest["roles"]}
    selected_roles = ["default", *deceptive_roles, *honest_roles]
    payload = {
        "name": f"selected-{path.parent.parent.name}",
        "anchor_role": "default",
        "pca_roles": [*deceptive_roles, *honest_roles],
        "roles": [
            {
                "name": role_name,
                "source": source_by_role[role_name],
                "anchor_only": role_name == "default",
            }
            for role_name in selected_roles
        ],
    }
    write_json(path, payload)


def save_selected_probe_snapshot(
    path: Path,
    transfer_config,  # noqa: ANN001
    experiment_config,  # noqa: ANN001
    honest_roles: list[str],
    deceptive_roles: list[str],
    method: str,
    layer_spec: str,
    objective: str,
    threshold: float,
    role_manifest_path: Path,
    question_file_path: Path,
    *,
    target_manifest: dict[str, Any],
    target_slug: str,
) -> None:
    payload = json.loads(json.dumps(transfer_config.raw))
    payload["role_axis"]["experiment_config"] = str(experiment_config.path)
    payload["role_axis"]["honest_roles"] = honest_roles
    payload["role_axis"]["deceptive_roles"] = deceptive_roles
    payload["role_axis"]["layer_specs"] = [layer_spec]
    payload["evaluation"]["methods"] = [method]
    payload["selection"] = {"objective": objective, "threshold": threshold}
    payload["selected_inputs"] = {
        "role_manifest_file": str(role_manifest_path),
        "question_file": str(question_file_path),
    }
    payload["selected_evaluation_target"] = {
        "target_slug": target_slug,
        **target_manifest,
    }
    write_json(path, payload)


def plot_baseline_vs_final(path: Path, baseline_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    dataset_labels = [row["dataset_label"] for row in baseline_rows]
    baseline_values = [float(row["auroc"]) for row in baseline_rows]
    final_by_dataset = {row["dataset"]: row for row in final_rows}
    final_values = [float(final_by_dataset[row["dataset"]]["auroc"]) for row in baseline_rows]
    x_positions = np.arange(len(dataset_labels), dtype=np.float32)
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars_a = ax.bar(x_positions - width / 2, baseline_values, width=width, label="baseline", color="#adb5bd")
    bars_b = ax.bar(x_positions + width / 2, final_values, width=width, label="final", color="#0b7285")
    for bars in (bars_a, bars_b):
        for bar in bars:
            value = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(0.985, value + 0.012),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=90,
            )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title("Baseline vs Final Fixed-Track AUROC")
    ax.set_xticks(x_positions, dataset_labels, rotation=30, ha="right")
    ax.grid(True, axis="y", color="#dee2e6", alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_deltas(path: Path, final_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    dataset_labels = [row["dataset_label"] for row in final_rows]
    deltas = [float(row["delta"]) for row in final_rows]
    colors = ["#2b8a3e" if value >= 0 else "#c92a2a" for value in deltas]
    x_positions = np.arange(len(dataset_labels), dtype=np.float32)

    fig, ax = plt.subplots(figsize=(12, 5.0))
    bars = ax.bar(x_positions, deltas, color=colors)
    for bar, value in zip(bars, deltas, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.002 if value >= 0 else -0.002),
            f"{value:.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
            rotation=90,
        )
    ax.axhline(0.0, color="#495057", linewidth=1.0)
    ax.set_ylabel("AUROC delta vs baseline")
    ax.set_title("Fixed-Track Delta vs Baseline")
    ax.set_xticks(x_positions, dataset_labels, rotation=30, ha="right")
    ax.grid(True, axis="y", color="#e9ecef", alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(path: Path, history_rows: list[dict[str, Any]], objective: str, threshold: float) -> None:
    import matplotlib.pyplot as plt

    steps = [int(row["step"]) for row in history_rows]
    mean_auroc = [float(row["mean_auroc"]) for row in history_rows]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if objective == "threshold_first":
        gt_threshold = [int(row["count_gt_threshold"]) for row in history_rows]
        clipped_sum = [float(row["clipped_threshold_sum"]) for row in history_rows]
        ax.plot(steps, gt_threshold, marker="o", color="#2b8a3e", label=f"datasets > {threshold:.2f}")
        ax.plot(steps, clipped_sum, marker="o", color="#c92a2a", label=f"clipped sum @ {threshold:.2f}")
    else:
        improved = [int(row["improved_count"]) for row in history_rows]
        mean_delta = [float(row["mean_delta"]) for row in history_rows]
        ax.plot(steps, improved, marker="o", color="#2b8a3e", label="datasets improved")
        ax.plot(steps, mean_delta, marker="o", color="#c92a2a", label="mean delta")
    ax.plot(steps, mean_auroc, marker="o", color="#0b7285", label="mean AUROC")
    ax.set_xlabel("Greedy step")
    ax.set_title("Fixed-Track Search Trajectory")
    ax.grid(True, color="#e9ecef", alpha=0.85)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_search_targets(
    args: argparse.Namespace,
    transfer_config,  # noqa: ANN001
    experiment_config,  # noqa: ANN001
    resolved_spec,  # noqa: ANN001
) -> tuple[dict[str, CachedSplitFeatures], list[str], dict[str, Any]]:
    if not has_activation_row_source(args):
        target_datasets = args.target_dataset or transfer_config.target_datasets
        split_cache: dict[str, CachedSplitFeatures] = {}
        for dataset_name in target_datasets:
            split_dir = transfer_config.target_activations_root / dataset_name / transfer_config.eval_split
            print(f"[fixed-track-search] loading target split {dataset_name} from {split_dir}")
            split_cache[dataset_name] = load_completion_mean_split(split_dir, [resolved_spec])
        return split_cache, target_datasets, {
            "backend": "completion_split",
            "target_datasets": target_datasets,
            "target_activations_root": str(transfer_config.target_activations_root),
            "eval_split": transfer_config.eval_split,
        }

    rows, source_manifest = load_activation_rows(
        activation_dataset_repo_id=args.activation_dataset_repo_id,
        activation_dataset_dir=args.activation_dataset_dir,
        activation_jsonl=args.activation_jsonl,
        split=args.activation_split,
    )
    stacked = validate_and_stack_activation_rows(
        rows,
        group_field=args.activation_group_field,
        source_manifest=source_manifest,
    )
    ensure_pooling_compatibility(
        activation_pooling=stacked.activation_pooling,
        expected_pooling=axis_expected_pooling(experiment_config),
        context="Activation-row search target",
    )
    ensure_single_layer_compatibility(
        activation_layer_number=stacked.activation_layer_number,
        expected_layer_numbers=resolved_spec.layer_numbers,
        context="Activation-row search target",
    )
    group_index = grouped_indices(stacked.group_values)
    target_datasets = args.target_dataset or list(group_index.keys())
    missing_datasets = [dataset_name for dataset_name in target_datasets if dataset_name not in group_index]
    if missing_datasets:
        raise ValueError(
            f"Requested activation-row groups are unavailable: {missing_datasets}; "
            f"available={list(group_index.keys())}"
        )

    split_cache = {}
    for dataset_name in target_datasets:
        indices = group_index[dataset_name]
        split_cache[dataset_name] = CachedSplitFeatures(
            dataset=dataset_name,
            split="all",
            sample_ids=[stacked.sample_ids[index] for index in indices.tolist()],
            labels=stacked.labels[indices],
            features_by_spec={resolved_spec.key: stacked.features[indices]},
        )
    return split_cache, target_datasets, {
        "backend": "activation_rows",
        "group_field": args.activation_group_field,
        "target_datasets": target_datasets,
        "activation_source": source_manifest,
        "activation_pooling": stacked.activation_pooling,
        "activation_layer_number": stacked.activation_layer_number,
    }


def main() -> None:
    args = parse_args()
    experiment_config = load_config(args.experiment_config.resolve())
    transfer_config = load_transfer_config(args.probe_config.resolve())
    if args.method not in set(transfer_config.methods):
        raise ValueError(f"Requested method {args.method} is not present in {transfer_config.path}")
    if args.layer_spec not in set(transfer_config.layer_specs):
        raise ValueError(f"Requested layer {args.layer_spec} is not present in {transfer_config.path}")

    run_id = args.run_id or make_run_id()
    target_slug = evaluation_target_slug(args)
    run_root = fixed_track_run_root(
        experiment_config,
        args.method,
        args.layer_spec,
        run_id,
        args.objective,
        args.threshold,
        target_slug,
    )
    checkpoints_dir = run_root / "checkpoints"
    results_dir = run_root / "results"
    plots_dir = run_root / "plots"
    selected_dir = run_root / "selected"
    for directory in (checkpoints_dir, results_dir, plots_dir, selected_dir, run_root / "meta"):
        directory.mkdir(parents=True, exist_ok=True)

    completed_selection_path = results_dir / "final_selection.json"
    if completed_selection_path.exists():
        status_path = run_root / "meta" / "run_fixed_track_search_status.json"
        status_payload = read_json(status_path) if status_path.exists() else {}
        if str(status_payload.get("status") or status_payload.get("state")) == "completed":
            print(f"[fixed-track-search] existing completed run found at {run_root}; nothing to do")
            return

    corpus_root = resolve_corpus_root(experiment_config)
    role_manifest = read_json(experiment_config.role_manifest_path)
    question_rows = select_first_n(load_questions(experiment_config.question_file_path), int(experiment_config.subset["question_count"]))
    question_map = {row.question_id: row for row in question_rows}
    question_order = [row.question_id for row in question_rows]

    work_units = build_work_units(experiment_config.repo_root, experiment_config)
    target_ids = {unit["item_id"] for unit in work_units}
    item_meta = {unit["item_id"]: unit for unit in work_units}
    activation_index = load_index(corpus_root, "activations")
    missing_activations = sorted(target_ids - set(activation_index))
    if missing_activations:
        preview = missing_activations[:8]
        raise RuntimeError(
            f"Missing {len(missing_activations)} activation records for this experiment; first missing ids: {preview}"
        )

    selected_rows = [row for item_id, row in activation_index.items() if item_id in target_ids]
    records = load_activation_records(corpus_root, selected_rows)
    if not records:
        raise RuntimeError("No activation records loaded for the configured experiment.")

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

    example_tensor = next(iter(role_question_sum.values()))
    activation_layer_numbers = infer_activation_layer_numbers(records)
    resolved_spec = resolve_layer_specs(int(example_tensor.shape[0]), [args.layer_spec], activation_layer_numbers)[0]
    split_cache, target_datasets, target_manifest = load_search_targets(
        args,
        transfer_config,
        experiment_config,
        resolved_spec,
    )

    honest_pool = list(transfer_config.honest_roles)
    deceptive_pool = list(transfer_config.deceptive_roles)
    required_roles = {transfer_config.anchor_role, *honest_pool, *deceptive_pool}
    missing_roles = sorted(role for role in required_roles if not any(key[0] == role for key in role_question_sum))
    if missing_roles:
        raise RuntimeError(f"Missing role/question activation coverage for roles: {missing_roles}")

    def role_vectors_for_state(honest_roles: tuple[str, ...], deceptive_roles: tuple[str, ...], question_ids: tuple[str, ...]) -> dict[str, Any]:
        needed_roles = [transfer_config.anchor_role, *honest_roles, *deceptive_roles]
        vectors: dict[str, Any] = {}
        for role_name in needed_roles:
            total_sum = None
            total_count = 0
            for question_id in question_ids:
                key = (role_name, question_id)
                if key not in role_question_sum:
                    continue
                if total_sum is None:
                    total_sum = role_question_sum[key].clone()
                else:
                    total_sum += role_question_sum[key]
                total_count += role_question_count[key]
            if total_sum is None or total_count == 0:
                raise RuntimeError(f"No activation coverage for role={role_name} over selected questions")
            vectors[role_name] = total_sum / total_count
        return vectors

    state_cache: dict[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], dict[str, Any]] = {}

    def evaluate_state(honest_roles: tuple[str, ...], deceptive_roles: tuple[str, ...], question_ids: tuple[str, ...]) -> dict[str, Any]:
        cache_key = (honest_roles, deceptive_roles, question_ids)
        cached = state_cache.get(cache_key)
        if cached is not None:
            return cached

        role_vectors = role_vectors_for_state(honest_roles, deceptive_roles, question_ids)
        bundle = build_role_axis_bundle(
            role_vectors=role_vectors,
            honest_roles=list(honest_roles),
            deceptive_roles=list(deceptive_roles),
            anchor_role=transfer_config.anchor_role,
            layer_specs=[resolved_spec.key],
            layer_numbers=activation_layer_numbers,
        )
        layer_bundle = bundle["layers"][resolved_spec.key]
        per_dataset: list[dict[str, Any]] = []
        for dataset_name in target_datasets:
            split_payload = split_cache[dataset_name]
            features = split_payload.features_by_spec[resolved_spec.key]
            score_payload = scores_for_layer(features, layer_bundle)
            component_index = ZERO_SHOT_METHOD_COMPONENT_INDEX[args.method]
            if component_index is not None and score_payload["pc_scores"].shape[1] <= component_index:
                raise RuntimeError(
                    f"Method {args.method} requires PC{component_index + 1}, "
                    f"but only {score_payload['pc_scores'].shape[1]} PC dimensions are available"
                )
            honest_like_scores = (
                score_payload["contrast"]
                if component_index is None
                else score_payload["pc_scores"][:, component_index]
            )
            metrics, deceptive_scores, probabilities = evaluate_zero_shot(split_payload.labels, honest_like_scores)
            per_dataset.append(
                {
                    "dataset": dataset_name,
                    "dataset_label": target_dataset_label(dataset_name),
                    "layer_spec": resolved_spec.key,
                    "method": args.method,
                    "auroc": float(metrics["auroc"]),
                    "auprc": float(metrics["auprc"]),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "f1": float(metrics["f1"]),
                    "accuracy": float(metrics["accuracy"]),
                    "count": int(metrics["count"]),
                    "score_mean": float(np.mean(deceptive_scores)),
                    "probability_mean": float(np.mean(probabilities)),
                }
            )
        result = {
            "honest_roles": list(honest_roles),
            "deceptive_roles": list(deceptive_roles),
            "question_ids": list(question_ids),
            "question_count": len(question_ids),
            "per_dataset": per_dataset,
            "mean_auroc": float(np.mean([row["auroc"] for row in per_dataset])),
        }
        state_cache[cache_key] = result
        return result

    baseline_state = (
        tuple(honest_pool),
        tuple(deceptive_pool),
        tuple(question_order),
    )
    baseline_result = evaluate_state(*baseline_state)
    baseline_by_dataset = {row["dataset"]: float(row["auroc"]) for row in baseline_result["per_dataset"]}

    def score_result(result: dict[str, Any]) -> dict[str, Any]:
        aurocs = [float(row["auroc"]) for row in result["per_dataset"]]
        deltas = [float(row["auroc"]) - baseline_by_dataset[row["dataset"]] for row in result["per_dataset"]]
        return {
            "count_gt_threshold": int(sum(value > args.threshold for value in aurocs)),
            "count_ge_threshold": int(sum(value >= args.threshold for value in aurocs)),
            "clipped_threshold_sum": float(np.sum([min(value, args.threshold) for value in aurocs])),
            "improved_count": int(sum(delta > 1e-6 for delta in deltas)),
            "nonnegative_count": int(sum(delta >= -1e-6 for delta in deltas)),
            "mean_delta": float(np.mean(deltas)),
            "sum_delta": float(np.sum(deltas)),
            "mean_auroc": float(result["mean_auroc"]),
        }

    history_rows: list[dict[str, Any]] = []
    checkpoint_path = checkpoints_dir / "search_state.json"
    progress_path = checkpoints_dir / "progress.json"

    current_honest = list(honest_pool)
    current_deceptive = list(deceptive_pool)
    current_questions = list(question_order)
    if args.resume and checkpoint_path.exists():
        checkpoint = read_json(checkpoint_path)
        current_honest = list(checkpoint["current_honest_roles"])
        current_deceptive = list(checkpoint["current_deceptive_roles"])
        current_questions = list(checkpoint["current_question_ids"])
        history_rows = list(checkpoint.get("history_rows", []))
        print(
            f"[fixed-track-search] resuming run_id={run_id} from step={len(history_rows) - 1} "
            f"with {len(current_honest)} honest, {len(current_deceptive)} deceptive, {len(current_questions)} questions"
        )

    current_result = evaluate_state(tuple(current_honest), tuple(current_deceptive), tuple(current_questions))
    current_score = score_result(current_result)

    if not history_rows:
        history_rows = [
            {
                "step": 0,
                "move_type": "start",
                "item": "",
                **current_score,
                "honest_count": len(current_honest),
                "deceptive_count": len(current_deceptive),
                "question_count": len(current_questions),
                "honest_roles": ",".join(current_honest),
                "deceptive_roles": ",".join(current_deceptive),
                "question_ids": ",".join(current_questions),
                "updated_at": utc_now_iso(),
            }
        ]

    write_analysis_manifest(
        run_root,
        {
            "experiment_config_path": str(experiment_config.path),
            "probe_config_path": str(transfer_config.path),
            "corpus_root": str(corpus_root),
            "method": args.method,
            "layer_spec": resolved_spec.key,
            "objective": args.objective,
            "threshold": args.threshold,
            "target_slug": target_slug,
            "target_manifest": target_manifest,
            "target_datasets": target_datasets,
            "honest_pool": honest_pool,
            "deceptive_pool": deceptive_pool,
            "question_count": len(question_order),
            "min_honest": args.min_honest,
            "min_deceptive": args.min_deceptive,
            "min_questions": args.min_questions,
        },
    )
    write_stage_status(
        run_root,
        "run_fixed_track_search",
        "running",
        {
            "step": len(history_rows) - 1,
            "method": args.method,
            "layer_spec": resolved_spec.key,
            "objective": args.objective,
            "threshold": args.threshold,
            "honest_count": len(current_honest),
            "deceptive_count": len(current_deceptive),
            "question_count": len(current_questions),
            "current_score": current_score,
        },
    )

    baseline_rows = []
    for row in baseline_result["per_dataset"]:
        baseline_rows.append(
            {
                "dataset": row["dataset"],
                "dataset_label": row["dataset_label"],
                "method": args.method,
                "layer_spec": resolved_spec.key,
                "auroc": f"{float(row['auroc']):.6f}",
                "auprc": f"{float(row['auprc']):.6f}",
                "balanced_accuracy": f"{float(row['balanced_accuracy']):.6f}",
                "f1": f"{float(row['f1']):.6f}",
                "accuracy": f"{float(row['accuracy']):.6f}",
                "count": int(row["count"]),
            }
        )
    write_csv(
        results_dir / "baseline_per_dataset.csv",
        list(baseline_rows[0].keys()),
        baseline_rows,
    )
    print(
        "[fixed-track-search] baseline "
        f"method={args.method} layer={resolved_spec.key} "
        f"objective={args.objective} threshold={args.threshold:.2f} "
        f"honest={len(honest_pool)} deceptive={len(deceptive_pool)} questions={len(question_order)} "
        f"mean_auroc={baseline_result['mean_auroc']:.3f}"
    )

    def save_checkpoint() -> None:
        write_json(
            checkpoint_path,
            {
                "method": args.method,
                "layer_spec": resolved_spec.key,
                "objective": args.objective,
                "threshold": args.threshold,
                "current_honest_roles": current_honest,
                "current_deceptive_roles": current_deceptive,
                "current_question_ids": current_questions,
                "history_rows": history_rows,
                "updated_at": utc_now_iso(),
            },
        )
        write_json(
            progress_path,
            {
                "state": "running",
                "step": len(history_rows) - 1,
                "objective": args.objective,
                "threshold": args.threshold,
                "current_score": current_score,
                "honest_count": len(current_honest),
                "deceptive_count": len(current_deceptive),
                "question_count": len(current_questions),
                "updated_at": utc_now_iso(),
            },
        )

    save_checkpoint()

    step_number = len(history_rows)
    while True:
        candidate_rows: list[dict[str, Any]] = []
        candidates: list[tuple[str, str, list[str], list[str], list[str], dict[str, Any], dict[str, Any]]] = []
        if len(current_questions) > args.min_questions:
            for question_id in current_questions:
                next_questions = [value for value in current_questions if value != question_id]
                result = evaluate_state(tuple(current_honest), tuple(current_deceptive), tuple(next_questions))
                score = score_result(result)
                candidates.append(("remove_question", question_id, list(current_honest), list(current_deceptive), next_questions, result, score))
        if len(current_honest) > args.min_honest:
            for role_name in current_honest:
                next_honest = [value for value in current_honest if value != role_name]
                result = evaluate_state(tuple(next_honest), tuple(current_deceptive), tuple(current_questions))
                score = score_result(result)
                candidates.append(("remove_honest", role_name, next_honest, list(current_deceptive), list(current_questions), result, score))
        if len(current_deceptive) > args.min_deceptive:
            for role_name in current_deceptive:
                next_deceptive = [value for value in current_deceptive if value != role_name]
                result = evaluate_state(tuple(current_honest), tuple(next_deceptive), tuple(current_questions))
                score = score_result(result)
                candidates.append(("remove_deceptive", role_name, list(current_honest), next_deceptive, list(current_questions), result, score))

        if not candidates:
            print("[fixed-track-search] no further removable candidates remain")
            break

        print(
            f"[fixed-track-search] step={step_number} evaluating {len(candidates)} candidates "
            f"(current score={score_tuple(current_score, args.objective)})"
        )
        step_started = time.monotonic()
        best_candidate = None
        for idx, candidate in enumerate(candidates, start=1):
            move_type, item, honest_roles, deceptive_roles, question_ids, result, score = candidate
            if best_candidate is None or score_tuple(score, args.objective) > score_tuple(best_candidate[6], args.objective):
                best_candidate = candidate
            candidate_rows.append(
                {
                    "step": step_number,
                    "move_type": move_type,
                    "item": item,
                    **score,
                    "honest_count": len(honest_roles),
                    "deceptive_count": len(deceptive_roles),
                    "question_count": len(question_ids),
                    "honest_roles": ",".join(honest_roles),
                    "deceptive_roles": ",".join(deceptive_roles),
                    "question_ids": ",".join(question_ids),
                }
            )
            if idx == 1 or (args.progress_every > 0 and idx % args.progress_every == 0):
                print(
                    f"[fixed-track-search] step={step_number} candidates={idx}/{len(candidates)} "
                    f"best={score_tuple(best_candidate[6], args.objective)}"
                )

        write_csv(
            results_dir / "candidate_moves" / f"step_{step_number:03d}.csv",
            list(candidate_rows[0].keys()),
            candidate_rows,
        )

        assert best_candidate is not None
        best_score = best_candidate[6]
        if score_tuple(best_score, args.objective) <= score_tuple(current_score, args.objective):
            elapsed = time.monotonic() - step_started
            print(
                f"[fixed-track-search] no improving removal found at step={step_number} "
                f"after {elapsed:.1f}s; stopping"
            )
            break

        move_type, item, next_honest, next_deceptive, next_questions, current_result, current_score = best_candidate
        current_honest = list(next_honest)
        current_deceptive = list(next_deceptive)
        current_questions = list(next_questions)
        history_rows.append(
            {
                "step": step_number,
                "move_type": move_type,
                "item": item,
                **current_score,
                "honest_count": len(current_honest),
                "deceptive_count": len(current_deceptive),
                "question_count": len(current_questions),
                "honest_roles": ",".join(current_honest),
                "deceptive_roles": ",".join(current_deceptive),
                "question_ids": ",".join(current_questions),
                "updated_at": utc_now_iso(),
            }
        )
        elapsed = time.monotonic() - step_started
        print(
            f"[fixed-track-search] step={step_number} chose {move_type}={item} "
            f"score={score_tuple(current_score, args.objective)} elapsed={elapsed:.1f}s"
        )
        save_checkpoint()
        step_number += 1

    final_rows: list[dict[str, Any]] = []
    for row in current_result["per_dataset"]:
        baseline_auroc = baseline_by_dataset[row["dataset"]]
        final_rows.append(
            {
                "dataset": row["dataset"],
                "dataset_label": row["dataset_label"],
                "method": args.method,
                "layer_spec": resolved_spec.key,
                "baseline_auroc": f"{baseline_auroc:.6f}",
                "auroc": f"{float(row['auroc']):.6f}",
                "delta": f"{float(row['auroc']) - baseline_auroc:.6f}",
                "auprc": f"{float(row['auprc']):.6f}",
                "balanced_accuracy": f"{float(row['balanced_accuracy']):.6f}",
                "f1": f"{float(row['f1']):.6f}",
                "accuracy": f"{float(row['accuracy']):.6f}",
                "count": int(row["count"]),
            }
        )
    write_csv(results_dir / "final_per_dataset.csv", list(final_rows[0].keys()), final_rows)
    write_csv(results_dir / "step_history.csv", list(history_rows[0].keys()), history_rows)
    write_jsonl(results_dir / "step_history.jsonl", history_rows)
    write_json(
        results_dir / "final_selection.json",
        {
            "method": args.method,
            "layer_spec": resolved_spec.key,
            "objective": args.objective,
            "threshold": args.threshold,
            "target_slug": target_slug,
            "evaluation_target": target_manifest,
            "honest_roles": current_honest,
            "deceptive_roles": current_deceptive,
            "question_ids": current_questions,
            "question_count": len(current_questions),
            "score": current_score,
            "completed_at": utc_now_iso(),
        },
    )
    write_json(
        results_dir / "search_summary.json",
        {
            "baseline_score": score_result(baseline_result),
            "final_score": current_score,
            "objective": args.objective,
            "threshold": args.threshold,
            "steps_completed": len(history_rows) - 1,
        },
    )
    write_json(
        results_dir / "final_question_texts.json",
        {
            "questions": [
                {"question_id": question_id, "question": question_map[question_id].question}
                for question_id in current_questions
            ]
        },
    )
    write_json(
        results_dir / "final_role_sets.json",
        {
            "honest_roles": current_honest,
            "deceptive_roles": current_deceptive,
        },
    )

    selected_questions_path = selected_dir / "questions.jsonl"
    selected_manifest_path = selected_dir / "role_manifest.json"
    selected_probe_path = selected_dir / "probe_config.json"
    save_selected_questions(selected_questions_path, current_questions, question_rows)
    save_selected_role_manifest(selected_manifest_path, role_manifest, current_honest, current_deceptive)
    save_selected_probe_snapshot(
        selected_probe_path,
        transfer_config,
        experiment_config,
        current_honest,
        current_deceptive,
        args.method,
        resolved_spec.key,
        args.objective,
        args.threshold,
        selected_manifest_path,
        selected_questions_path,
        target_manifest=target_manifest,
        target_slug=target_slug,
    )

    final_plot_rows = [
        {
            "dataset": row["dataset"],
            "dataset_label": row["dataset_label"],
            "auroc": float(row["auroc"]),
            "delta": float(row["delta"]),
        }
        for row in final_rows
    ]
    plot_baseline_vs_final(
        plots_dir / "baseline_vs_final_auroc.png",
        [
            {"dataset": row["dataset"], "dataset_label": row["dataset_label"], "auroc": float(row["auroc"])}
            for row in baseline_rows
        ],
        final_plot_rows,
    )
    plot_deltas(plots_dir / "delta_vs_baseline.png", final_plot_rows)
    plot_trajectory(plots_dir / "search_trajectory.png", history_rows, args.objective, args.threshold)

    write_json(
        progress_path,
        {
            "state": "completed",
            "step": len(history_rows) - 1,
            "current_score": current_score,
            "honest_count": len(current_honest),
            "deceptive_count": len(current_deceptive),
            "question_count": len(current_questions),
            "updated_at": utc_now_iso(),
        },
    )
    write_stage_status(
        run_root,
        "run_fixed_track_search",
        "completed",
        {
            "step": len(history_rows) - 1,
            "method": args.method,
            "layer_spec": resolved_spec.key,
            "objective": args.objective,
            "threshold": args.threshold,
            "final_score": current_score,
            "final_selection_path": str(completed_selection_path),
        },
    )
    print(
        f"[fixed-track-search] completed run_id={run_id} "
        f"steps={len(history_rows) - 1} final_score={score_tuple(current_score, args.objective)}"
    )
    print(
        f"[fixed-track-search] final honest={current_honest} deceptive={current_deceptive} "
        f"questions={len(current_questions)}"
    )


if __name__ == "__main__":
    main()
