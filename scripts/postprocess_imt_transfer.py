#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from deception_honesty_axis.common import make_run_id, read_json
from deception_honesty_axis.imt_config import imt_run_root, load_imt_config
from deception_honesty_axis.imt_plotting import save_axis_variant_heatmap, save_four_axis_grouped_bars
from deception_honesty_axis.imt_recovery import IMT_AXIS_ORDER, write_imt_summary_csv, write_imt_wide_summary_csv
from deception_honesty_axis.metadata import write_stage_status


VARIANT_LABELS = {"raw": "Raw", "orth": "Orth"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate IMT transfer summaries and plots from saved metrics.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the IMT recovery config.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_imt_config(args.config.resolve())
    run_id = args.run_id or make_run_id()
    run_root = imt_run_root(config, run_id)
    results_dir = run_root / "results" / "eval"
    metrics_path = results_dir / "pairwise_metrics.csv"
    manifest_path = run_root / "meta" / "run_manifest.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing IMT metrics CSV: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    if not metric_rows:
        raise ValueError(f"No metric rows found in {metrics_path}")

    run_manifest = read_json(manifest_path) if manifest_path.exists() else {}
    write_stage_status(
        run_root,
        "postprocess_imt_transfer",
        "running",
        {"metric_rows": len(metric_rows), "bank_name": run_manifest.get("bank_name", config.bank_name)},
    )

    write_imt_summary_csv(results_dir / "summary_by_axis.csv", metric_rows)
    write_imt_wide_summary_csv(results_dir / "summary_wide.csv", metric_rows, config.target_datasets)

    plot_rows = [
        {
            "axis_name": row["axis_name"],
            "variant": VARIANT_LABELS.get(row["axis_variant"], row["axis_variant"]),
            "target_dataset": row["target_dataset"],
            "value": float(row["auroc"]),
        }
        for row in metric_rows
    ]
    plots_dir = results_dir / "plots"
    grouped_bar_path = save_four_axis_grouped_bars(
        plots_dir / "imt_zero_shot_grouped_bars__auroc.png",
        plot_rows,
        variant_order=["Raw", "Orth"],
        dataset_order=config.target_datasets,
        title=f"Recovered IMT Zero-Shot AUROC by Dataset\n{config.bank_name}",
    )
    heatmap_path = save_axis_variant_heatmap(
        plots_dir / "imt_zero_shot_heatmap__auroc.png",
        plot_rows,
        variant_order=["Raw", "Orth"],
        dataset_order=config.target_datasets,
        title=f"Recovered IMT Zero-Shot AUROC Heatmap\n{config.bank_name}",
    )
    payload = {
        "metric_rows": len(metric_rows),
        "bank_name": run_manifest.get("bank_name", config.bank_name),
        "summary_by_axis": str(results_dir / "summary_by_axis.csv"),
        "summary_wide": str(results_dir / "summary_wide.csv"),
        "plots": [grouped_bar_path, heatmap_path],
    }
    write_stage_status(run_root, "postprocess_imt_transfer", "completed", payload)
    print(
        f"[imt-postprocess] wrote summary_by_axis.csv, summary_wide.csv, "
        f"and 2 plots under {results_dir}"
    )


if __name__ == "__main__":
    main()
