#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from deception_honesty_axis.analysis import write_pca_report_artifacts
from deception_honesty_axis.common import read_json, write_json
from deception_honesty_axis.metadata import write_stage_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill PCA tables and plots from an existing PCA run directory.")
    parser.add_argument("--pca-run-dir", type=Path, required=True, help="Existing PCA run directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.pca_run_dir.resolve()
    results_payload = read_json(run_root / "results" / "results.json")
    manifest_payload = read_json(run_root / "meta" / "run_manifest.json")

    anchor_role = str(results_payload.get("anchor_role") or manifest_payload["anchor_role"])
    role_names = list(results_payload.get("role_names") or manifest_payload["role_names"])
    role_metadata = dict(results_payload.get("role_metadata") or manifest_payload.get("role_metadata") or {})
    layers = list(results_payload["layers"])

    write_stage_status(run_root, "materialize_pca_reports", "running", {"layer_count": len(layers)})
    report_artifacts = write_pca_report_artifacts(
        run_root,
        layers,
        anchor_role,
        role_names,
        role_metadata=role_metadata,
    )
    write_json(
        run_root / "checkpoints" / "materialize_pca_reports_progress.json",
        {"layer_count": len(layers), "completed": True},
    )
    write_stage_status(
        run_root,
        "materialize_pca_reports",
        "completed",
        {"layer_count": len(layers), "report_artifacts": report_artifacts},
    )
    print(f"Materialized PCA reports in {run_root}")


if __name__ == "__main__":
    main()
