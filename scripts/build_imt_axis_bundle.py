#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from deception_honesty_axis.common import make_run_id
from deception_honesty_axis.imt_config import imt_run_root, load_imt_config
from deception_honesty_axis.imt_recovery import load_fit_artifacts, write_imt_axis_bundle
from deception_honesty_axis.metadata import write_stage_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a reusable IMT axis bundle from fitted PCA+ridge artifacts.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the IMT recovery config.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_imt_config(args.config.resolve())
    run_id = args.run_id or make_run_id()
    run_root = imt_run_root(config, run_id)
    fit_artifacts_path = run_root / "results" / "fit" / "fit_artifacts.pt"
    if not fit_artifacts_path.exists():
        raise FileNotFoundError(f"Missing fit artifacts: {fit_artifacts_path}")

    write_stage_status(run_root, "build_imt_axis_bundle", "running", {"bank_name": config.bank_name})
    fit_artifacts = load_fit_artifacts(fit_artifacts_path)
    outputs = write_imt_axis_bundle(run_root, fit_artifacts, bank_name=config.bank_name)
    write_stage_status(
        run_root,
        "build_imt_axis_bundle",
        "completed",
        {"bank_name": config.bank_name, "artifacts": outputs},
    )
    print(f"[imt-axis-bundle] wrote axis bundle to {run_root / 'results'}")


if __name__ == "__main__":
    main()
