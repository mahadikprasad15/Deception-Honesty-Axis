#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from deception_honesty_axis.common import make_run_id
from deception_honesty_axis.imt_config import imt_run_root, load_imt_config
from deception_honesty_axis.imt_recovery import build_imt_bank_records, write_bank_artifacts
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a unified IMT source bank from existing source corpora.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the IMT recovery config.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_imt_config(args.config.resolve())
    run_id = args.run_id or make_run_id()
    run_root = imt_run_root(config, run_id)

    write_analysis_manifest(
        run_root,
        {
            "imt_config_path": str(config.path),
            "run_id": run_id,
            "bank_name": config.bank_name,
            "source_specs": [
                {
                    "axis_name": spec.axis_name,
                    "experiment_config": str(spec.experiment_config_path),
                    "analysis_run_id": spec.analysis_run_id,
                }
                for spec in config.source_specs
            ],
        },
    )
    write_stage_status(run_root, "build_imt_bank", "running", {"bank_name": config.bank_name})

    records, summary = build_imt_bank_records(config)
    outputs = write_bank_artifacts(run_root, records, summary)
    write_stage_status(
        run_root,
        "build_imt_bank",
        "completed",
        {
            "bank_name": config.bank_name,
            "record_count": summary.total_records,
            "by_axis": summary.by_axis,
            "by_role_side": summary.by_role_side,
            "artifacts": outputs,
        },
    )
    print(f"[imt-bank] wrote {summary.total_records} records under {run_root}")


if __name__ == "__main__":
    main()
