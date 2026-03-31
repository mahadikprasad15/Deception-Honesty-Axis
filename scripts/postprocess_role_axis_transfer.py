#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from deception_honesty_axis.common import read_json
from deception_honesty_axis.metadata import write_stage_status
from deception_honesty_axis.role_axis_transfer import save_metric_heatmaps, write_fit_summary_csv, write_summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate role-axis transfer summaries and heatmaps from saved results only."
    )
    parser.add_argument(
        "--transfer-run-dir",
        type=Path,
        required=True,
        help="Existing role-axis transfer run directory containing results/pairwise_metrics.csv.",
    )
    return parser.parse_args()


def _sorted_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def main() -> None:
    args = parse_args()
    run_root = args.transfer_run_dir.resolve()
    results_dir = run_root / "results"
    metrics_path = results_dir / "pairwise_metrics.csv"
    fit_artifacts_path = results_dir / "source_fit_artifacts.jsonl"
    manifest_path = run_root / "meta" / "run_manifest.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing pairwise_metrics.csv in {results_dir}")

    if manifest_path.exists():
        manifest = read_json(manifest_path)
        methods = manifest.get("methods", [])
        layer_specs = manifest.get("layer_specs", [])
        datasets = manifest.get("datasets", [])
    else:
        manifest = {}
        methods = []
        layer_specs = []
        datasets = []

    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))

    if not metric_rows:
        raise ValueError(f"No metric rows found in {metrics_path}")

    if not methods:
        methods = _sorted_unique([row["method"] for row in metric_rows])
    if not layer_specs:
        layer_specs = _sorted_unique([row["layer_spec"] for row in metric_rows])
    if not datasets:
        datasets = _sorted_unique(
            [
                row["target_dataset"]
                for row in metric_rows
                if row["target_dataset"] and row["target_dataset"] != "__zero_shot__"
            ]
        )

    write_stage_status(
        run_root,
        "postprocess_role_axis_transfer",
        "running",
        {
            "metric_rows": len(metric_rows),
            "methods": methods,
            "layer_specs": layer_specs,
            "datasets": datasets,
        },
    )
    print(
        f"[role-axis-transfer-postprocess] regenerating summaries/plots for {run_root} "
        f"from {len(metric_rows)} saved metric rows"
    )

    write_summary_csv(results_dir / "summary_by_method.csv", metric_rows)
    write_fit_summary_csv(results_dir / "fit_summary.csv", fit_artifacts_path)
    heatmap_paths = save_metric_heatmaps(
        results_dir / "plots",
        metric_rows,
        methods,
        layer_specs,
        datasets,
    )

    payload = {
        "metric_rows": len(metric_rows),
        "methods": methods,
        "layer_specs": layer_specs,
        "datasets": datasets,
        "heatmaps": heatmap_paths,
    }
    write_stage_status(run_root, "postprocess_role_axis_transfer", "completed", payload)
    print(
        f"[role-axis-transfer-postprocess] wrote summary_by_method.csv, fit_summary.csv, "
        f"and {len(heatmap_paths)} heatmaps under {results_dir}"
    )


if __name__ == "__main__":
    main()
