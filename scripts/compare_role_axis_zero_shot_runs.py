#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from deception_honesty_axis.transfer_comparison import (
    load_comparison_manifest,
    run_zero_shot_transfer_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare completed zero-shot role-axis transfer runs across datasets."
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        required=True,
        help="JSON manifest describing the completed transfer run dirs to compare.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional fixed comparison run id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_comparison_manifest(args.run_manifest)
    outputs = run_zero_shot_transfer_comparison(manifest, run_id=args.run_id)
    print(
        "[role-axis-transfer-compare] wrote "
        f"{outputs['record_count']} zero-shot rows, {len(outputs['plots'])} grouped-bar plots, "
        f"and overview plot to {outputs['run_root']}"
    )


if __name__ == "__main__":
    main()
