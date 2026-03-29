from __future__ import annotations

from pathlib import Path
from typing import Any

from deception_honesty_axis.common import ensure_dir, utc_now_iso, write_json


def write_corpus_manifest(corpus_root: Path, payload: dict[str, Any]) -> None:
    ensure_dir(corpus_root / "meta")
    write_json(corpus_root / "meta" / "corpus_manifest.json", payload)


def write_stage_status(root: Path, stage_name: str, state: str, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "stage": stage_name,
        "state": state,
        "status": state,
        "updated_at": utc_now_iso(),
    }
    if extra:
        payload.update(extra)
    write_json(root / "meta" / f"{stage_name}_status.json", payload)
    write_json(root / "meta" / "status.json", payload)


def write_analysis_manifest(run_root: Path, payload: dict[str, Any]) -> None:
    ensure_dir(run_root / "meta")
    write_json(run_root / "meta" / "run_manifest.json", payload)
