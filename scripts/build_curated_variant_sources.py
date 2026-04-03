#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from deception_honesty_axis.config import find_repo_root
from deception_honesty_axis.source_curation import materialize_curated_variant_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build curated assistant-axis source files from an upstream spec.")
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="Path to the curated source spec JSON.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild curated outputs even if they already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(args.spec.resolve().parent)
    payload = materialize_curated_variant_sources(args.spec.resolve(), repo_root, force=args.force)
    print(
        f"Built curated variant sources from {args.spec.resolve()} "
        f"into {payload['manifest_path']} ({len(payload['files'])} files)"
    )


if __name__ == "__main__":
    main()
