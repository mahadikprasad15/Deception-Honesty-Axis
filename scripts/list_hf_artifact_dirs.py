#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from deception_honesty_axis.common import write_json
from deception_honesty_axis.paper_plotting import (
    canonical_axis_display_name,
    infer_artifact_kind,
    infer_axis_key_from_path,
    parse_pc_count_from_path,
    resolve_run_root_from_artifact_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List local or HF artifact run directories relevant to the paper plots.")
    parser.add_argument("--output", type=Path, required=True, help="Path to write the JSON inventory.")
    parser.add_argument("--local-root", type=Path, default=Path("artifacts"), help="Local artifact root to scan.")
    parser.add_argument("--repo-id", default=None, help="Optional HF dataset repo id to query.")
    return parser.parse_args()


def local_inventory_paths(local_root: Path) -> list[str]:
    if not local_root.exists():
        return []
    return [str(path.relative_to(local_root)) for path in local_root.rglob("*") if path.is_file()]


def hf_inventory_paths(repo_id: str) -> list[str]:
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub to query HF artifacts.") from exc

    try:
        return list(HfApi().list_repo_files(repo_id, repo_type="dataset"))
    except RepositoryNotFoundError as exc:
        raise SystemExit(
            "HF artifact listing failed. Ensure the repo id is correct and you are authenticated "
            "with huggingface-cli login or HF_TOKEN."
        ) from exc


def build_inventory(records: list[str], source_name: str) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        artifact_kind = infer_artifact_kind(record)
        if artifact_kind is None:
            continue
        axis_key = infer_axis_key_from_path(record) or "unknown_axis"
        grouped.setdefault(axis_key, []).append(
            {
                "source": source_name,
                "artifact_kind": artifact_kind,
                "run_root": resolve_run_root_from_artifact_path(record, artifact_kind),
                "path": record,
                "selected_pc_count": parse_pc_count_from_path(record),
            }
        )

    inventory_axes: list[dict[str, object]] = []
    for axis_key in sorted(grouped):
        entries = sorted(grouped[axis_key], key=lambda item: (str(item["artifact_kind"]), str(item["run_root"])))
        inventory_axes.append(
            {
                "axis_key": axis_key,
                "axis_display_name": canonical_axis_display_name(axis_key),
                "entries": entries,
            }
        )
    return {
        "source": source_name,
        "axis_count": len(inventory_axes),
        "axes": inventory_axes,
    }


def main() -> None:
    args = parse_args()
    local_paths = local_inventory_paths(args.local_root.resolve())
    inventory = {
        "local": build_inventory(local_paths, "local"),
    }
    if args.repo_id:
        inventory["hf"] = build_inventory(hf_inventory_paths(str(args.repo_id)), "hf")
    write_json(args.output.resolve(), inventory)
    print(f"[hf-artifact-dirs] wrote inventory to {args.output.resolve()}")


if __name__ == "__main__":
    main()
