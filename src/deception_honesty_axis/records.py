from __future__ import annotations

from pathlib import Path
from typing import Any

from deception_honesty_axis.common import load_jsonl


def load_rollout_records(corpus_root: Path, index_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_shard: dict[str, list[str]] = {}
    for row in index_rows:
        by_shard.setdefault(row["shard_path"], []).append(row["item_id"])

    loaded: dict[str, dict[str, Any]] = {}
    for shard_path, item_ids in by_shard.items():
        records = load_jsonl(corpus_root / shard_path)
        allowed = set(item_ids)
        for record in records:
            if record["item_id"] in allowed:
                loaded[record["item_id"]] = record
    return loaded


def load_activation_records(corpus_root: Path, index_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import torch

    by_shard: dict[str, list[str]] = {}
    for row in index_rows:
        by_shard.setdefault(row["shard_path"], []).append(row["item_id"])

    records_out: list[dict[str, Any]] = []
    for shard_path, item_ids in by_shard.items():
        payload = torch.load(corpus_root / shard_path, map_location="cpu")
        allowed = set(item_ids)
        for record in payload["records"]:
            if record["item_id"] in allowed:
                records_out.append(record)
    return records_out
