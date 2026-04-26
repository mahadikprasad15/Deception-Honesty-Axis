#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from deception_honesty_axis.activation_row_transfer import load_activation_row_dataset
from deception_honesty_axis.activation_row_transfer_config import load_activation_row_transfer_config
from deception_honesty_axis.common import ensure_dir, make_run_id, slugify, write_json
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.role_axis_transfer import load_role_axis_bundle, scores_for_layer
from deception_honesty_axis.sycophancy_activations import normalize_activation_pooling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-sample 2D scatters using PC1 projections from two role-axis bundles."
    )
    parser.add_argument("--config", type=Path, required=True, help="Activation-row transfer config JSON.")
    parser.add_argument("--x-axis-bundle-run-dir", type=Path, required=True, help="Run dir containing results/axis_bundle.pt.")
    parser.add_argument("--y-axis-bundle-run-dir", type=Path, required=True, help="Run dir containing results/axis_bundle.pt.")
    parser.add_argument("--x-axis-label", default=None, help="Optional label override for the x axis.")
    parser.add_argument("--y-axis-label", default=None, help="Optional label override for the y axis.")
    parser.add_argument(
        "--split-mode",
        choices=["combined", "train", "eval", "both"],
        default="combined",
        help="Which dataset splits to plot. 'combined' concatenates train+eval unless they are identical.",
    )
    parser.add_argument("--run-id", default=None, help="Optional fixed run id.")
    return parser.parse_args()


def _safe_equal_manifest(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left == right


def _same_samples(sample_ids_a: list[str], sample_ids_b: list[str], labels_a: np.ndarray, labels_b: np.ndarray) -> bool:
    return sample_ids_a == sample_ids_b and np.array_equal(labels_a, labels_b)


def axis_expected_pooling(axis_bundle_dir: Path) -> str | None:
    manifest_path = axis_bundle_dir / "meta" / "run_manifest.json"
    if not manifest_path.exists():
        return None
    payload = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    pooling = payload.get("expected_pooling")
    if pooling in (None, ""):
        return None
    return normalize_activation_pooling(str(pooling))


def infer_axis_slug(axis_bundle_dir: Path) -> str:
    manifest_path = axis_bundle_dir / "meta" / "run_manifest.json"
    if manifest_path.exists():
        payload = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("bundle_name", "bundle_slug", "axis_name", "variant_name"):
            value = payload.get(key)
            if value not in (None, ""):
                return slugify(str(value))
    return slugify(axis_bundle_dir.name)


def resolve_axis_layer_bundle(axis_bundle: dict[str, Any], expected_layer_number: int | None) -> tuple[str, str, dict[str, Any]]:
    layer_specs = list(axis_bundle["layers"].keys())
    if expected_layer_number is None:
        if len(layer_specs) != 1:
            raise ValueError(
                "Expected exactly one layer in axis bundle when config expected_layer_number is omitted; "
                f"found {layer_specs!r}"
            )
        layer_spec = layer_specs[0]
    else:
        matches: list[str] = []
        for layer_spec in layer_specs:
            layer_payload = axis_bundle["layers"][layer_spec]
            layer_numbers = [int(value) for value in (layer_payload.get("layer_numbers") or [])]
            if expected_layer_number in layer_numbers:
                matches.append(layer_spec)
                continue
            try:
                parsed_layer_spec = int(str(layer_spec))
            except (TypeError, ValueError):
                parsed_layer_spec = None
            if parsed_layer_spec == int(expected_layer_number):
                matches.append(layer_spec)
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one axis bundle layer for layer {expected_layer_number}; found {matches!r}"
            )
        layer_spec = matches[0]
    layer_bundle = axis_bundle["layers"][layer_spec]
    layer_numbers = [int(value) for value in layer_bundle.get("layer_numbers", [])]
    layer_label = f"L{layer_numbers[0]}" if len(layer_numbers) == 1 else str(layer_spec)
    return str(layer_spec), layer_label, layer_bundle


def _projection_rows_for_split(
    *,
    dataset_name: str,
    split_name: str,
    sample_ids: list[str],
    labels: np.ndarray,
    x_scores: np.ndarray,
    y_scores: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "dataset": dataset_name,
            "split": split_name,
            "sample_id": sample_id,
            "label": int(label),
            "class_name": "deceptive" if int(label) == 1 else "honest",
            "x_pc1": float(x_value),
            "y_pc1": float(y_value),
        }
        for sample_id, label, x_value, y_value in zip(sample_ids, labels, x_scores, y_scores, strict=False)
    ]


def _save_scatter(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    show_split_style: bool,
) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(path.parent)
    label_order = [0, 1]
    label_meta = {
        0: {"name": "honest", "color": "#1f77b4"},
        1: {"name": "deceptive", "color": "#d62728"},
    }
    split_markers = {
        "train": "o",
        "eval": "^",
        "all": "o",
        "combined": "o",
    }

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    ax.set_facecolor("#fbfbfd")
    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.25)

    for label in label_order:
        label_rows = [row for row in rows if int(row["label"]) == label]
        if not label_rows:
            continue
        if show_split_style:
            for split_name in sorted({str(row["split"]) for row in label_rows}):
                split_rows = [row for row in label_rows if str(row["split"]) == split_name]
                ax.scatter(
                    [row["x_pc1"] for row in split_rows],
                    [row["y_pc1"] for row in split_rows],
                    s=28,
                    alpha=0.65,
                    c=label_meta[label]["color"],
                    marker=split_markers.get(split_name, "o"),
                    edgecolors="none",
                    label=f"{label_meta[label]['name']} | {split_name}",
                )
        else:
            ax.scatter(
                [row["x_pc1"] for row in label_rows],
                [row["y_pc1"] for row in label_rows],
                s=28,
                alpha=0.65,
                c=label_meta[label]["color"],
                marker="o",
                edgecolors="none",
                label=label_meta[label]["name"],
            )

    ax.axhline(0.0, color="#888888", linewidth=0.8, alpha=0.5)
    ax.axvline(0.0, color="#888888", linewidth=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_activation_row_transfer_config(args.config.resolve())

    x_axis_bundle_dir = args.x_axis_bundle_run_dir.resolve()
    y_axis_bundle_dir = args.y_axis_bundle_run_dir.resolve()
    x_axis_bundle = load_role_axis_bundle(x_axis_bundle_dir / "results" / "axis_bundle.pt")
    y_axis_bundle = load_role_axis_bundle(y_axis_bundle_dir / "results" / "axis_bundle.pt")

    expected_layer_number = config.expected_layer_number
    x_layer_spec, x_layer_label, x_layer_bundle = resolve_axis_layer_bundle(x_axis_bundle, expected_layer_number)
    y_layer_spec, y_layer_label, y_layer_bundle = resolve_axis_layer_bundle(y_axis_bundle, expected_layer_number)

    x_axis_slug = infer_axis_slug(x_axis_bundle_dir)
    y_axis_slug = infer_axis_slug(y_axis_bundle_dir)
    x_axis_label = args.x_axis_label or f"{x_axis_slug} PC1"
    y_axis_label = args.y_axis_label or f"{y_axis_slug} PC1"

    x_expected_pooling = axis_expected_pooling(x_axis_bundle_dir)
    y_expected_pooling = axis_expected_pooling(y_axis_bundle_dir)
    if x_expected_pooling and config.expected_pooling and x_expected_pooling != normalize_activation_pooling(config.expected_pooling):
        raise ValueError(
            f"X-axis bundle expects pooling {x_expected_pooling!r}, config expects {config.expected_pooling!r}"
        )
    if y_expected_pooling and config.expected_pooling and y_expected_pooling != normalize_activation_pooling(config.expected_pooling):
        raise ValueError(
            f"Y-axis bundle expects pooling {y_expected_pooling!r}, config expects {config.expected_pooling!r}"
        )

    run_id = args.run_id or make_run_id()
    run_root = (
        config.artifact_root
        / "runs"
        / "activation-row-dual-axis-pc1-scatter"
        / slugify(config.model_name)
        / slugify(config.dataset_set_name)
        / slugify(f"{x_axis_slug}__{y_axis_slug}")
        / run_id
    )
    for relative in ("inputs", "results", "logs", "checkpoints", "meta"):
        ensure_dir(run_root / relative)

    write_analysis_manifest(
        run_root,
        {
            "config_path": str(config.path),
            "run_id": run_id,
            "experiment_name": "activation-row-dual-axis-pc1-scatter",
            "model_name": config.model_name,
            "behavior_name": config.behavior_name,
            "dataset_set_name": config.dataset_set_name,
            "expected_pooling": config.expected_pooling,
            "expected_layer_number": config.expected_layer_number,
            "split_mode": args.split_mode,
            "x_axis_bundle_run_dir": str(x_axis_bundle_dir),
            "y_axis_bundle_run_dir": str(y_axis_bundle_dir),
            "x_axis_label": x_axis_label,
            "y_axis_label": y_axis_label,
            "x_layer_spec": x_layer_spec,
            "y_layer_spec": y_layer_spec,
        },
    )
    write_stage_status(
        run_root,
        "plot_activation_row_dual_axis_pc1_scatter",
        "running",
        {"dataset_count": len(config.datasets), "split_mode": args.split_mode},
    )

    all_projection_rows: list[dict[str, Any]] = []
    dataset_summaries: list[dict[str, Any]] = []
    plot_paths: list[str] = []

    for dataset_config in config.datasets:
        loaded = load_activation_row_dataset(
            dataset_config,
            expected_pooling=config.expected_pooling,
            expected_layer_number=config.expected_layer_number,
        )

        train_x = scores_for_layer(loaded.train.features, x_layer_bundle)["pc_scores"][:, 0].astype(np.float32, copy=False)
        train_y = scores_for_layer(loaded.train.features, y_layer_bundle)["pc_scores"][:, 0].astype(np.float32, copy=False)
        eval_x = scores_for_layer(loaded.eval.features, x_layer_bundle)["pc_scores"][:, 0].astype(np.float32, copy=False)
        eval_y = scores_for_layer(loaded.eval.features, y_layer_bundle)["pc_scores"][:, 0].astype(np.float32, copy=False)

        train_rows = _projection_rows_for_split(
            dataset_name=loaded.name,
            split_name="train",
            sample_ids=loaded.train.sample_ids,
            labels=loaded.train.labels,
            x_scores=train_x,
            y_scores=train_y,
        )
        eval_rows = _projection_rows_for_split(
            dataset_name=loaded.name,
            split_name="eval",
            sample_ids=loaded.eval.sample_ids,
            labels=loaded.eval.labels,
            x_scores=eval_x,
            y_scores=eval_y,
        )

        if args.split_mode == "train":
            active_rows = train_rows
            title_suffix = "train"
        elif args.split_mode == "eval":
            active_rows = eval_rows
            title_suffix = "eval"
        elif args.split_mode == "both":
            for split_name, split_rows in (("train", train_rows), ("eval", eval_rows)):
                plot_path = run_root / "results" / "plots" / f"{slugify(loaded.name)}__{split_name}.png"
                _save_scatter(
                    plot_path,
                    split_rows,
                    title=f"{loaded.name} | {split_name}",
                    x_label=x_axis_label,
                    y_label=y_axis_label,
                    show_split_style=False,
                )
                plot_paths.append(str(plot_path))
                all_projection_rows.extend(split_rows)
            dataset_summaries.append(
                {
                    "dataset": loaded.name,
                    "mode": "both",
                    "train_count": len(train_rows),
                    "eval_count": len(eval_rows),
                }
            )
            continue
        else:
            same_source = _safe_equal_manifest(loaded.train.source_manifest, loaded.eval.source_manifest)
            same_records = _same_samples(
                loaded.train.sample_ids,
                loaded.eval.sample_ids,
                loaded.train.labels,
                loaded.eval.labels,
            )
            if same_source and same_records:
                active_rows = train_rows
                title_suffix = "all"
            else:
                active_rows = train_rows + eval_rows
                title_suffix = "combined"

        plot_path = run_root / "results" / "plots" / f"{slugify(loaded.name)}__{slugify(title_suffix)}.png"
        _save_scatter(
            plot_path,
            active_rows,
            title=f"{loaded.name} | {title_suffix}",
            x_label=x_axis_label,
            y_label=y_axis_label,
            show_split_style=(args.split_mode == "combined" and title_suffix == "combined"),
        )
        plot_paths.append(str(plot_path))
        all_projection_rows.extend(active_rows)
        dataset_summaries.append(
            {
                "dataset": loaded.name,
                "mode": args.split_mode,
                "row_count": len(active_rows),
                "title_suffix": title_suffix,
                "label_counts": {
                    str(label): sum(1 for row in active_rows if int(row["label"]) == label)
                    for label in (0, 1)
                },
            }
        )

    import csv

    projections_jsonl = run_root / "results" / "per_sample_projections.jsonl"
    projections_csv = run_root / "results" / "per_sample_projections.csv"
    from deception_honesty_axis.common import append_jsonl

    append_jsonl(projections_jsonl, all_projection_rows)
    with projections_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "split", "sample_id", "label", "class_name", "x_pc1", "y_pc1"],
        )
        writer.writeheader()
        writer.writerows(all_projection_rows)

    write_json(
        run_root / "results" / "summary.json",
        {
            "dataset_summaries": dataset_summaries,
            "plot_paths": plot_paths,
            "per_sample_projections_jsonl": str(projections_jsonl),
            "per_sample_projections_csv": str(projections_csv),
        },
    )
    write_stage_status(
        run_root,
        "plot_activation_row_dual_axis_pc1_scatter",
        "completed",
        {
            "dataset_count": len(dataset_summaries),
            "plot_count": len(plot_paths),
            "per_sample_projections_jsonl": str(projections_jsonl),
            "per_sample_projections_csv": str(projections_csv),
        },
    )
    print(f"[activation-row-dual-axis-pc1-scatter] completed run_id={run_id} root={run_root}")


if __name__ == "__main__":
    main()
