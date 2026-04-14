#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from deception_honesty_axis.imt_transfer_comparison import load_imt_comparison_manifest, run_imt_transfer_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare completed IMT transfer runs across source banks.")
    parser.add_argument("--run-manifest", type=Path, required=True, help="JSON manifest describing completed IMT run dirs.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed comparison run id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_imt_comparison_manifest(args.run_manifest)
    outputs = run_imt_transfer_comparison(manifest, run_id=args.run_id)
    print(
        "[imt-transfer-compare] wrote "
        f"{outputs['record_count']} rows, {len(outputs['plots'])} plots, "
        f"and CSVs under {outputs['run_root']}"
    )


if __name__ == "__main__":
    main()
