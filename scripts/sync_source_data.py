#!/usr/bin/env python3
from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from pathlib import Path

from deception_honesty_axis.common import ensure_dir, sha256_file, utc_now_iso, write_json
from deception_honesty_axis.config import load_config
from deception_honesty_axis.work_units import load_role_manifest


RAW_BASE = "https://raw.githubusercontent.com/safety-research/assistant-axis/master"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download assistant-axis source inputs into the repo.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/deception_v1_llama32_3b.json"),
        help="Path to the experiment config file.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download files even if they already exist.")
    return parser.parse_args()


def download(url: str, output_path: Path, force: bool) -> dict[str, str]:
    ensure_dir(output_path.parent)
    if output_path.exists() and not force:
        status = "skipped"
    else:
        try:
            urllib.request.urlretrieve(url, output_path)
            status = "downloaded"
        except urllib.error.HTTPError:
            if output_path.exists():
                status = "retained_local"
            else:
                raise
    return {
        "url": url,
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "status": status,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    role_manifest = load_role_manifest(config.role_manifest_path)

    entries = [
        download(
            f"{RAW_BASE}/data/extraction_questions.jsonl",
            config.question_file_path,
            force=args.force,
        )
    ]
    for role_entry in role_manifest["roles"]:
        source_name = role_entry.get("source") or Path(role_entry["instruction_file"]).name
        instruction_path = config.repo_root / "data" / "roles" / "instructions" / source_name
        entries.append(
            download(
                f"{RAW_BASE}/data/roles/instructions/{instruction_path.name}",
                instruction_path,
                force=args.force,
            )
        )

    write_json(
        config.repo_root / "data" / "manifests" / "source_manifest.json",
        {
            "upstream_repo": "https://github.com/safety-research/assistant-axis",
            "fetched_at": utc_now_iso(),
            "config_path": str(config.path),
            "files": entries,
        },
    )
    print(f"Synced {len(entries)} source files.")


if __name__ == "__main__":
    main()
