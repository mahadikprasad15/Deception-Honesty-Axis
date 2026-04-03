from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from deception_honesty_axis.common import ensure_dir, make_run_id, read_json
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.role_axis_transfer import DATASET_LABELS, ZERO_SHOT_SOURCE


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
DEFAULT_METHODS = ["contrast_zero_shot", "pc1_zero_shot", "pc2_zero_shot", "pc3_zero_shot"]
DEFAULT_LAYER_SPECS = ["7", "14", "21", "28"]
DEFAULT_VARIANT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


@dataclass(frozen=True)
class ComparisonVariant:
    label: str
    transfer_run_dir: Path


@dataclass(frozen=True)
class ZeroShotComparisonManifest:
    path: Path
    repo_root: Path
    artifact_root: Path
    model_slug: str
    variants: list[ComparisonVariant]
    datasets: list[str]
    methods: list[str]
    layer_specs: list[str]


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate repo root from {start}")


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def _infer_model_slug(variant_dirs: list[Path]) -> str:
    for run_dir in variant_dirs:
        parts = run_dir.resolve().parts
        for idx, part in enumerate(parts):
            if part == "role-axis-transfer" and idx + 1 < len(parts):
                return parts[idx + 1]
    return "comparison"


def load_comparison_manifest(path: str | Path) -> ZeroShotComparisonManifest:
    manifest_path = Path(path).resolve()
    raw = read_json(manifest_path)
    repo_root = _find_repo_root(manifest_path.parent)
    artifact_root = _resolve_path(repo_root, str(raw.get("artifacts", {}).get("root", "artifacts")))

    variant_entries = raw.get("variants", [])
    if not variant_entries:
        raise ValueError("Comparison manifest must define at least one variant")
    variants = [
        ComparisonVariant(
            label=str(entry["label"]),
            transfer_run_dir=_resolve_path(repo_root, str(entry["transfer_run_dir"])),
        )
        for entry in variant_entries
    ]

    datasets = [str(item) for item in raw.get("dataset_order", DEFAULT_DATASET_ORDER)]
    methods = [str(item) for item in raw.get("methods", DEFAULT_METHODS)]
    layer_specs = [str(item) for item in raw.get("layer_specs", DEFAULT_LAYER_SPECS)]
    model_slug = str(raw.get("model_slug") or _infer_model_slug([variant.transfer_run_dir for variant in variants]))
    return ZeroShotComparisonManifest(
        path=manifest_path,
        repo_root=repo_root,
        artifact_root=artifact_root,
        model_slug=model_slug,
        variants=variants,
        datasets=datasets,
        methods=methods,
        layer_specs=layer_specs,
    )


def load_zero_shot_records(
    variants: list[ComparisonVariant],
    datasets: list[str],
    methods: list[str],
    layer_specs: list[str],
) -> list[dict[str, Any]]:
    expected_keys = {
        (variant.label, method, layer_spec, dataset)
        for variant in variants
        for method in methods
        for layer_spec in layer_specs
        for dataset in datasets
    }
    records: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()

    for variant in variants:
        metrics_path = variant.transfer_run_dir / "results" / "pairwise_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing pairwise_metrics.csv in {variant.transfer_run_dir / 'results'}")
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            method = str(row.get("method", ""))
            layer_spec = str(row.get("layer_spec", ""))
            dataset = str(row.get("target_dataset", ""))
            source_dataset = str(row.get("source_dataset", ""))
            if method not in methods or layer_spec not in layer_specs or dataset not in datasets:
                continue
            if source_dataset != ZERO_SHOT_SOURCE:
                continue
            key = (variant.label, method, layer_spec, dataset)
            if key in seen_keys:
                raise ValueError(f"Duplicate zero-shot metric row for {key} in {metrics_path}")
            seen_keys.add(key)
            value = row.get("auroc")
            if value in (None, ""):
                raise ValueError(f"Missing AUROC for {key} in {metrics_path}")
            records.append(
                {
                    "variant": variant.label,
                    "method": method,
                    "layer_spec": layer_spec,
                    "target_dataset": dataset,
                    "dataset_label": DATASET_LABELS.get(dataset, dataset),
                    "auroc": float(value),
                    "source_run_dir": str(variant.transfer_run_dir),
                }
            )

    missing = expected_keys - seen_keys
    if missing:
        preview = sorted(missing)[:8]
        raise ValueError(
            "Missing zero-shot coverage for some variant/method/layer/dataset combinations; "
            f"first missing keys: {preview}"
        )
    return sorted(
        records,
        key=lambda row: (
            methods.index(str(row["method"])),
            layer_specs.index(str(row["layer_spec"])),
            datasets.index(str(row["target_dataset"])),
            [variant.label for variant in variants].index(str(row["variant"])),
        ),
    )


def write_long_csv(path: Path, records: list[dict[str, Any]]) -> str:
    fieldnames = [
        "variant",
        "method",
        "layer_spec",
        "target_dataset",
        "dataset_label",
        "auroc",
        "source_run_dir",
    ]
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return str(path)


def write_wide_csv(
    path: Path,
    records: list[dict[str, Any]],
    variants: list[ComparisonVariant],
    datasets: list[str],
    methods: list[str],
    layer_specs: list[str],
) -> str:
    index = {
        (str(row["variant"]), str(row["method"]), str(row["layer_spec"]), str(row["target_dataset"])): row
        for row in records
    }
    fieldnames = ["target_dataset", "dataset_label"]
    for method in methods:
        for layer_spec in layer_specs:
            for variant in variants:
                fieldnames.append(f"{variant.label}__{method}__{layer_spec}")
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in datasets:
            row: dict[str, Any] = {
                "target_dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset),
            }
            for method in methods:
                for layer_spec in layer_specs:
                    for variant in variants:
                        record = index[(variant.label, method, layer_spec, dataset)]
                        row[f"{variant.label}__{method}__{layer_spec}"] = f"{float(record['auroc']):.6f}"
            writer.writerow(row)
    return str(path)


def plot_grouped_bars(
    output_dir: Path,
    records: list[dict[str, Any]],
    variants: list[ComparisonVariant],
    datasets: list[str],
    methods: list[str],
    layer_specs: list[str],
) -> list[str]:
    import matplotlib.pyplot as plt

    ensure_dir(output_dir)
    index = {
        (str(row["variant"]), str(row["method"]), str(row["layer_spec"]), str(row["target_dataset"])): float(row["auroc"])
        for row in records
    }
    variant_colors = {
        variant.label: DEFAULT_VARIANT_COLORS[idx % len(DEFAULT_VARIANT_COLORS)]
        for idx, variant in enumerate(variants)
    }
    dataset_labels = [DATASET_LABELS.get(dataset, dataset) for dataset in datasets]
    x_positions = np.arange(len(datasets), dtype=np.float32)
    width = 0.8 / max(1, len(variants))
    output_paths: list[str] = []

    for method in methods:
        for layer_spec in layer_specs:
            fig, ax = plt.subplots(figsize=(max(16, len(datasets) * 1.8), 6.5))
            for idx, variant in enumerate(variants):
                offsets = x_positions - 0.4 + width * 0.5 + idx * width
                values = [index[(variant.label, method, layer_spec, dataset)] for dataset in datasets]
                bars = ax.bar(
                    offsets,
                    values,
                    width=width,
                    color=variant_colors[variant.label],
                    label=variant.label,
                )
                for bar, value in zip(bars, values, strict=False):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        min(0.98, value + 0.015),
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        rotation=90,
                    )

            ax.set_xticks(x_positions, dataset_labels, rotation=35, ha="right")
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("AUROC")
            ax.set_title(f"Zero-Shot AUROC: {method} @ {layer_spec}")
            ax.grid(True, axis="y", color="#dee2e6", alpha=0.9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False, ncol=max(1, min(3, len(variants))))
            fig.tight_layout()
            path = output_dir / f"{method}__{layer_spec}__grouped_bar_auroc.png"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            output_paths.append(str(path))

    return output_paths


def plot_overview(
    output_dir: Path,
    records: list[dict[str, Any]],
    variants: list[ComparisonVariant],
    datasets: list[str],
    methods: list[str],
    layer_specs: list[str],
) -> str:
    import matplotlib.pyplot as plt

    ensure_dir(output_dir)
    index = {
        (str(row["variant"]), str(row["method"]), str(row["layer_spec"]), str(row["target_dataset"])): float(row["auroc"])
        for row in records
    }
    variant_colors = {
        variant.label: DEFAULT_VARIANT_COLORS[idx % len(DEFAULT_VARIANT_COLORS)]
        for idx, variant in enumerate(variants)
    }
    dataset_labels = [DATASET_LABELS.get(dataset, dataset) for dataset in datasets]
    x_positions = np.arange(len(datasets), dtype=np.float32)
    width = 0.8 / max(1, len(variants))

    ncols = 2
    nrows = int(np.ceil((len(methods) * len(layer_specs)) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, max(4.8 * nrows, 10)), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(-1)

    for ax in axes_array:
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", color="#e9ecef")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    panel_specs = [(method, layer_spec) for method in methods for layer_spec in layer_specs]
    for ax, (method, layer_spec) in zip(axes_array, panel_specs, strict=False):
        for idx, variant in enumerate(variants):
            offsets = x_positions - 0.4 + width * 0.5 + idx * width
            values = [index[(variant.label, method, layer_spec, dataset)] for dataset in datasets]
            ax.bar(
                offsets,
                values,
                width=width,
                color=variant_colors[variant.label],
                label=variant.label,
            )
        ax.set_title(f"{method} @ {layer_spec}")
        ax.set_xticks(x_positions, dataset_labels, rotation=35, ha="right")
        ax.set_ylabel("AUROC")

    for ax in axes_array[len(panel_specs):]:
        ax.axis("off")

    handles, labels = axes_array[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, min(3, len(variants))), frameon=False)
    fig.suptitle("Role-Axis Zero-Shot AUROC Comparison Across Datasets", fontsize=18, fontweight="bold")
    path = output_dir / "all_variants_zero_shot_overview.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run_zero_shot_transfer_comparison(
    manifest: ZeroShotComparisonManifest,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    compare_run_id = run_id or make_run_id()
    run_root = (
        manifest.artifact_root
        / "runs"
        / "role-axis-transfer-comparisons"
        / manifest.model_slug
        / compare_run_id
    )
    ensure_dir(run_root)
    write_analysis_manifest(
        run_root,
        {
            "comparison_manifest_path": str(manifest.path),
            "variants": [
                {"label": variant.label, "transfer_run_dir": str(variant.transfer_run_dir)}
                for variant in manifest.variants
            ],
            "datasets": manifest.datasets,
            "methods": manifest.methods,
            "layer_specs": manifest.layer_specs,
        },
    )
    write_stage_status(
        run_root,
        "compare_role_axis_zero_shot_runs",
        "running",
        {
            "variant_count": len(manifest.variants),
            "dataset_count": len(manifest.datasets),
            "methods": manifest.methods,
            "layer_specs": manifest.layer_specs,
        },
    )

    records = load_zero_shot_records(
        manifest.variants,
        manifest.datasets,
        manifest.methods,
        manifest.layer_specs,
    )
    results_dir = run_root / "results"
    plots_dir = results_dir / "plots"
    long_csv = write_long_csv(results_dir / "zero_shot_auroc_long.csv", records)
    wide_csv = write_wide_csv(
        results_dir / "zero_shot_auroc_wide.csv",
        records,
        manifest.variants,
        manifest.datasets,
        manifest.methods,
        manifest.layer_specs,
    )
    plot_paths = plot_grouped_bars(
        plots_dir,
        records,
        manifest.variants,
        manifest.datasets,
        manifest.methods,
        manifest.layer_specs,
    )
    overview_path = plot_overview(
        plots_dir,
        records,
        manifest.variants,
        manifest.datasets,
        manifest.methods,
        manifest.layer_specs,
    )
    payload = {
        "record_count": len(records),
        "variant_count": len(manifest.variants),
        "dataset_count": len(manifest.datasets),
        "long_csv": long_csv,
        "wide_csv": wide_csv,
        "plots": plot_paths,
        "overview_plot": overview_path,
    }
    write_stage_status(run_root, "compare_role_axis_zero_shot_runs", "completed", payload)
    return {"run_root": str(run_root), **payload}
