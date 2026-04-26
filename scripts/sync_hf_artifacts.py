#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from deception_honesty_axis.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push or pull artifact directories with the Hugging Face Hub.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/deception_v1_llama32_3b.json"),
        help="Path to the experiment config file.",
    )
    parser.add_argument("--direction", choices=["push", "pull"], required=True, help="Sync direction.")
    parser.add_argument("--local-dir", type=Path, default=None, help="Optional local path override.")
    parser.add_argument("--repo-id", type=str, default=None, help="Optional HF dataset repo override.")
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=None,
        help=(
            "Optional Hugging Face snapshot allow patterns for selective pulls, for example "
            "'runs/role-axis-bundles/**'."
        ),
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="*",
        default=None,
        help="Optional Hugging Face snapshot ignore patterns for pulls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download, upload_folder
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub to use HF artifact syncing.") from exc

    config = load_config(args.config.resolve())
    repo_id = args.repo_id or config.hf.get("dataset_repo_id") or config.hf["dataset_repo"]
    local_dir = (args.local_dir or config.artifact_root).resolve()

    if args.direction == "push":
        upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(local_dir))
        print(f"Pushed {local_dir} to hf://datasets/{repo_id}")
        return

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
    )
    pattern_text = ""
    if args.allow_patterns:
        pattern_text += f" allow_patterns={args.allow_patterns}"
    if args.ignore_patterns:
        pattern_text += f" ignore_patterns={args.ignore_patterns}"
    print(f"Pulled hf://datasets/{repo_id} into {local_dir}{pattern_text}")


if __name__ == "__main__":
    main()
