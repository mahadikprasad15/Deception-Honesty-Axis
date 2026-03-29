from __future__ import annotations

from pathlib import Path
from typing import Any

from deception_honesty_axis.common import append_jsonl, ensure_dir, load_jsonl, write_json


def ensure_corpus_layout(corpus_root: Path) -> None:
    for relative in (
        "meta",
        "indexes",
        "checkpoints",
        "rollouts",
        "activations",
        "reports",
        "logs",
    ):
        ensure_dir(corpus_root / relative)


def shard_dir(corpus_root: Path, stage: str, role: str) -> Path:
    return ensure_dir(corpus_root / stage / f"role={role}")


def next_shard_path(corpus_root: Path, stage: str, role: str, suffix: str) -> Path:
    directory = shard_dir(corpus_root, stage, role)
    existing = sorted(directory.glob(f"part-*{suffix}"))
    next_index = len(existing) + 1
    return directory / f"part-{next_index:05d}{suffix}"


def index_path(corpus_root: Path, stage: str) -> Path:
    return corpus_root / "indexes" / f"{stage}.jsonl"


def load_index(corpus_root: Path, stage: str) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(index_path(corpus_root, stage))
    return {row["item_id"]: row for row in rows}


def append_index_records(corpus_root: Path, stage: str, records: list[dict[str, Any]]) -> None:
    if records:
        append_jsonl(index_path(corpus_root, stage), records)


def update_coverage(corpus_root: Path, payload: dict[str, Any]) -> None:
    write_json(corpus_root / "meta" / "coverage.json", payload)
