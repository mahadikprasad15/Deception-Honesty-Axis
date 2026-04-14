#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from deception_honesty_axis.common import ensure_dir, make_run_id, read_json, write_json
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.role_axis_transfer import METHOD_COLORS, METHOD_LABELS


DEFAULT_METHODS = ["contrast_zero_shot", "pc1_zero_shot", "pc2_zero_shot", "pc3_zero_shot"]
DATASET_LABEL_OVERRIDES = {
    "sycophancy_dataset_train": "Sycophancy Dataset",
    "sycophancy_dataset-train": "Sycophancy Dataset",
    "open_ended_sycophancy_train": "Open-Ended Sycophancy",
    "open-ended_sycophancy_train": "Open-Ended Sycophancy",
    "open-ended-sycophancy-train": "Open-Ended Sycophancy",
    "oeq_validation_human_balanced_200": "OEQ Validation",
    "oeq-validation_human-balanced-200": "OEQ Validation",
    "oeq-validation-human-balanced-200": "OEQ Validation",
    "oeq_indirectness_human_balanced_200": "OEQ Indirectness",
    "oeq-indirectness_human-balanced-200": "OEQ Indirectness",
    "oeq-indirectness-human-balanced-200": "OEQ Indirectness",
    "oeq_framing_human_balanced_200": "OEQ Framing",
    "oeq-framing_human-balanced-200": "OEQ Framing",
    "oeq-framing-human-balanced-200": "OEQ Framing",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create grouped AUROC bar charts across completed sycophancy activation probe runs."
    )
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Completed sycophancy-activation-probe run directories.",
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"), help="Repo artifact root.")
    parser.add_argument("--run-id", default=None, help="Optional fixed comparison run id.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Zero-shot methods to include.",
    )
    parser.add_argument(
        "--layer-specs",
        nargs="+",
        default=None,
        help="Optional layer specs to include. Defaults to all compatible layer specs present in the selected rows.",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include target_dataset=all rows in addition to dataset-specific rows.",
    )
    parser.add_argument(
        "--exclude-datasets",
        nargs="+",
        default=None,
        help="Optional target_dataset names to exclude entirely.",
    )
    return parser.parse_args()


def _infer_run_label(run_dir: Path) -> str:
    manifest_path = run_dir / "meta" / "run_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        target_name = str(manifest.get("target_name") or "").strip()
        if target_name:
            return target_name
    return run_dir.name


def _display_label(dataset_name: str) -> str:
    if dataset_name in DATASET_LABEL_OVERRIDES:
        return DATASET_LABEL_OVERRIDES[dataset_name]
    text = dataset_name.replace("_", " ").replace("-", " ").strip()
    text = " ".join(part for part in text.split() if part)
    title_cased = text.title()
    return title_cased.replace("Oeq", "OEQ")


def _load_metric_rows(run_dir: Path) -> list[dict[str, str]]:
    metrics_path = run_dir / "results" / "pairwise_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing pairwise_metrics.csv in {run_dir / 'results'}")
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _select_rows(run_dir: Path, methods: list[str], include_all: bool) -> list[dict[str, str]]:
    rows = _load_metric_rows(run_dir)
    filtered = [row for row in rows if str(row.get("method", "")) in methods]
    dataset_specific = [row for row in filtered if str(row.get("target_dataset", "")) != "all"]
    if dataset_specific and not include_all:
        return dataset_specific
    return filtered


def _collect_records(run_dirs: list[Path], methods: list[str], include_all: bool, excluded_datasets: set[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for run_dir in run_dirs:
        run_label = _infer_run_label(run_dir.resolve())
        for row in _select_rows(run_dir.resolve(), methods, include_all):
            target_dataset = str(row.get("target_dataset", "")).strip()
            if target_dataset in excluded_datasets:
                continue
            method = str(row.get("method", "")).strip()
            layer_spec = str(row.get("layer_spec", "")).strip()
            key = (target_dataset, method, layer_spec)
            if key in seen_keys:
                raise ValueError(
                    "Duplicate dataset/method/layer coverage across selected runs; "
                    f"duplicate key={key!r}. Pass only one run per dataset."
                )
            seen_keys.add(key)
            records.append(
                {
                    "dataset": target_dataset,
                    "dataset_label": _display_label(target_dataset),
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "layer_spec": layer_spec,
                    "auroc": float(row["auroc"]),
                    "source_run_dir": str(run_dir.resolve()),
                    "source_run_label": run_label,
                }
            )
    return records


def _dataset_order(records: list[dict[str, object]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for record in records:
        dataset = str(record["dataset"])
        if dataset in seen:
            continue
        seen.add(dataset)
        ordered.append(dataset)
    return ordered


def _plot_grouped_bars(output_path: Path, records: list[dict[str, object]], methods: list[str], datasets: list[str], layer_spec: str) -> str:
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    values = {
        (str(record["dataset"]), str(record["method"])): float(record["auroc"])
        for record in records
        if str(record["layer_spec"]) == layer_spec
    }
    x_positions = np.arange(len(datasets), dtype=np.float32)
    width = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(max(16, len(datasets) * 1.9), 7.0))
    for index, method in enumerate(methods):
        offsets = x_positions - 0.4 + width * 0.5 + index * width
        method_values = [values[(dataset, method)] for dataset in datasets]
        bars = ax.bar(
            offsets,
            method_values,
            width=width,
            color=METHOD_COLORS.get(method, "#495057"),
            label=METHOD_LABELS.get(method, method),
        )
        for bar, value in zip(bars, method_values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(0.98, value + 0.015),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=90,
            )

    ax.set_xticks(x_positions, [_display_label(dataset) for dataset in datasets], rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_xlabel("Dataset")
    ax.set_title(f"Sycophancy Zero-Shot AUROC by Dataset @ Layer {layer_spec}")
    ax.grid(True, axis="y", color="#dee2e6", alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=max(1, min(4, len(methods))))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def main() -> None:
    args = parse_args()
    run_dirs = [path.resolve() for path in args.run_dirs]
    excluded_datasets = {str(value) for value in (args.exclude_datasets or [])}
    records = _collect_records(
        run_dirs,
        [str(method) for method in args.methods],
        bool(args.include_all),
        excluded_datasets,
    )
    if not records:
        raise ValueError("No comparison rows found for the selected run dirs and methods")

    layer_specs = (
        [str(value) for value in args.layer_specs]
        if args.layer_specs
        else sorted({str(record["layer_spec"]) for record in records}, key=lambda value: (len(value), value))
    )
    datasets = _dataset_order(records)

    run_id = args.run_id or make_run_id()
    run_root = ensure_dir(args.artifact_root / "runs" / "sycophancy-activation-probe-comparison" / run_id)
    for relative in ("inputs", "results", "logs", "checkpoints", "meta"):
        ensure_dir(run_root / relative)

    write_analysis_manifest(
        run_root,
        {
            "run_id": run_id,
            "run_dirs": [str(path) for path in run_dirs],
            "methods": [str(method) for method in args.methods],
            "layer_specs": layer_specs,
            "include_all": bool(args.include_all),
            "exclude_datasets": sorted(excluded_datasets),
        },
    )
    write_stage_status(run_root, "compare_sycophancy_probe_runs", "running", {"run_id": run_id})
    write_json(run_root / "inputs" / "selected_run_dirs.json", [str(path) for path in run_dirs])
    write_json(
        run_root / "checkpoints" / "progress.json",
        {"state": "running", "run_count": len(run_dirs), "record_count": len(records)},
    )

    long_rows_path = run_root / "results" / "combined_metrics.csv"
    with long_rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "dataset_label",
                "method",
                "method_label",
                "layer_spec",
                "auroc",
                "source_run_dir",
                "source_run_label",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    plot_paths = []
    for layer_spec in layer_specs:
        layer_records = [record for record in records if str(record["layer_spec"]) == layer_spec]
        expected_keys = {(dataset, method) for dataset in datasets for method in args.methods}
        actual_keys = {(str(record["dataset"]), str(record["method"])) for record in layer_records}
        missing = expected_keys - actual_keys
        if missing:
            preview = sorted(missing)[:8]
            raise ValueError(f"Missing dataset/method coverage for layer {layer_spec}: {preview}")
        plot_paths.append(
            _plot_grouped_bars(
                run_root / "results" / "plots" / f"sycophancy_probe_comparison__auroc__{layer_spec}.png",
                layer_records,
                [str(method) for method in args.methods],
                datasets,
                layer_spec,
            )
        )

    write_json(
        run_root / "results" / "summary.json",
        {"record_count": len(records), "datasets": datasets, "methods": [str(method) for method in args.methods], "plots": plot_paths},
    )
    write_json(
        run_root / "checkpoints" / "progress.json",
        {"state": "completed", "run_count": len(run_dirs), "record_count": len(records), "plots": plot_paths},
    )
    write_stage_status(
        run_root,
        "compare_sycophancy_probe_runs",
        "completed",
        {"run_id": run_id, "record_count": len(records), "plots": plot_paths},
    )
    print(
        "[sycophancy-probe-compare] wrote "
        f"{len(records)} metric rows and {len(plot_paths)} plot(s) under {run_root}"
    )


if __name__ == "__main__":
    main()
