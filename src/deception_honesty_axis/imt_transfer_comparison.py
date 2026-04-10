from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deception_honesty_axis.common import ensure_dir, make_run_id, read_json
from deception_honesty_axis.imt_plotting import IMT_AXIS_ORDER, IMT_DATASET_ORDER, save_axis_variant_heatmap, save_four_axis_grouped_bars
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status


@dataclass(frozen=True)
class IMTComparisonVariant:
    label: str
    imt_run_dir: Path


@dataclass(frozen=True)
class IMTComparisonManifest:
    path: Path
    repo_root: Path
    artifact_root: Path
    model_slug: str
    variants: list[IMTComparisonVariant]
    datasets: list[str]
    axis_variant: str


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
            if part == "imt-recovery" and idx + 1 < len(parts):
                return parts[idx + 1]
    return "imt-comparison"


def load_imt_comparison_manifest(path: str | Path) -> IMTComparisonManifest:
    manifest_path = Path(path).resolve()
    raw = read_json(manifest_path)
    repo_root = _find_repo_root(manifest_path.parent)
    artifact_root = _resolve_path(repo_root, str(raw.get("artifacts", {}).get("root", "artifacts")))
    variants = [
        IMTComparisonVariant(
            label=str(entry["label"]),
            imt_run_dir=_resolve_path(repo_root, str(entry["imt_run_dir"])),
        )
        for entry in raw["variants"]
    ]
    datasets = [str(item) for item in raw.get("dataset_order", IMT_DATASET_ORDER)]
    axis_variant = str(raw.get("axis_variant", "raw"))
    model_slug = str(raw.get("model_slug") or _infer_model_slug([variant.imt_run_dir for variant in variants]))
    return IMTComparisonManifest(
        path=manifest_path,
        repo_root=repo_root,
        artifact_root=artifact_root,
        model_slug=model_slug,
        variants=variants,
        datasets=datasets,
        axis_variant=axis_variant,
    )


def load_comparison_records(
    variants: list[IMTComparisonVariant],
    *,
    datasets: list[str],
    axis_variant: str,
) -> list[dict[str, Any]]:
    expected = {
        (variant.label, axis_name, dataset)
        for variant in variants
        for axis_name in IMT_AXIS_ORDER
        for dataset in datasets
    }
    seen: set[tuple[str, str, str]] = set()
    records: list[dict[str, Any]] = []
    for variant in variants:
        metrics_path = variant.imt_run_dir / "results" / "eval" / "pairwise_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing IMT metrics at {metrics_path}")
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            dataset = str(row["target_dataset"])
            axis_name = str(row["axis_name"])
            row_axis_variant = str(row["axis_variant"])
            if dataset not in datasets or axis_name not in IMT_AXIS_ORDER or row_axis_variant != axis_variant:
                continue
            key = (variant.label, axis_name, dataset)
            seen.add(key)
            records.append(
                {
                    "variant": variant.label,
                    "axis_name": axis_name,
                    "target_dataset": dataset,
                    "value": float(row["auroc"]),
                    "imt_run_dir": str(variant.imt_run_dir),
                }
            )
    missing = expected - seen
    if missing:
        raise ValueError(f"Missing comparison coverage for IMT runs; first missing keys: {sorted(missing)[:8]}")
    return records


def run_imt_transfer_comparison(manifest: IMTComparisonManifest, *, run_id: str | None = None) -> dict[str, Any]:
    comparison_run_id = run_id or make_run_id()
    run_root = (
        manifest.artifact_root
        / "runs"
        / "imt-transfer-comparison"
        / manifest.model_slug
        / f"axis-variant={manifest.axis_variant}"
        / comparison_run_id
    )
    ensure_dir(run_root)
    write_analysis_manifest(
        run_root,
        {
            "comparison_manifest_path": str(manifest.path),
            "axis_variant": manifest.axis_variant,
            "datasets": manifest.datasets,
            "variants": [{"label": variant.label, "imt_run_dir": str(variant.imt_run_dir)} for variant in manifest.variants],
        },
    )
    write_stage_status(
        run_root,
        "run_imt_transfer_comparison",
        "running",
        {"axis_variant": manifest.axis_variant, "variant_count": len(manifest.variants), "dataset_count": len(manifest.datasets)},
    )
    records = load_comparison_records(manifest.variants, datasets=manifest.datasets, axis_variant=manifest.axis_variant)
    results_dir = ensure_dir(run_root / "results")
    long_path = results_dir / "comparison_long.csv"
    wide_path = results_dir / "comparison_wide.csv"

    with long_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "axis_name", "target_dataset", "value", "imt_run_dir"])
        writer.writeheader()
        writer.writerows(records)

    with wide_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["target_dataset"]
        for axis_name in IMT_AXIS_ORDER:
            for variant in manifest.variants:
                fieldnames.append(f"{axis_name}__{variant.label}")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        index = {(row["variant"], row["axis_name"], row["target_dataset"]): row["value"] for row in records}
        for dataset in manifest.datasets:
            row = {"target_dataset": dataset}
            for axis_name in IMT_AXIS_ORDER:
                for variant in manifest.variants:
                    row[f"{axis_name}__{variant.label}"] = index[(variant.label, axis_name, dataset)]
            writer.writerow(row)

    plots_dir = ensure_dir(results_dir / "plots")
    grouped_bar = save_four_axis_grouped_bars(
        plots_dir / "imt_comparison_grouped_bars__auroc.png",
        [
            {
                "axis_name": row["axis_name"],
                "variant": row["variant"],
                "target_dataset": row["target_dataset"],
                "value": row["value"],
            }
            for row in records
        ],
        variant_order=[variant.label for variant in manifest.variants],
        dataset_order=manifest.datasets,
        title=f"Recovered IMT Zero-Shot AUROC by Dataset\nAxis Variant: {manifest.axis_variant}",
    )
    heatmap = save_axis_variant_heatmap(
        plots_dir / "imt_comparison_heatmap__auroc.png",
        [
            {
                "axis_name": row["axis_name"],
                "variant": row["variant"],
                "target_dataset": row["target_dataset"],
                "value": row["value"],
            }
            for row in records
        ],
        variant_order=[variant.label for variant in manifest.variants],
        dataset_order=manifest.datasets,
        title=f"Recovered IMT Zero-Shot AUROC Heatmap\nAxis Variant: {manifest.axis_variant}",
    )
    write_stage_status(
        run_root,
        "run_imt_transfer_comparison",
        "completed",
        {
            "record_count": len(records),
            "plots": [grouped_bar, heatmap],
            "results_dir": str(results_dir),
        },
    )
    return {
        "run_root": str(run_root),
        "record_count": len(records),
        "plots": [grouped_bar, heatmap],
        "long_csv": str(long_path),
        "wide_csv": str(wide_path),
    }
