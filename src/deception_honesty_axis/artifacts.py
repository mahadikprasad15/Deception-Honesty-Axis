from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .config import ExperimentConfig
from .io_utils import append_jsonl, read_json, read_jsonl, write_json


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = uuid4().hex[:6]
    return f"{stamp}-{token}"


def _next_part_path(directory: Path, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    existing = sorted(path for path in directory.glob(f"part-*.{suffix}") if path.is_file())
    next_index = len(existing) + 1
    return directory / f"part-{next_index:05d}.{suffix}"


@dataclass(frozen=True)
class CorpusPaths:
    root: Path
    meta_dir: Path
    checkpoints_dir: Path
    indexes_dir: Path
    rollouts_dir: Path
    activations_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class AnalysisPaths:
    root: Path
    inputs_dir: Path
    checkpoints_dir: Path
    results_dir: Path
    logs_dir: Path
    meta_dir: Path


def corpus_paths(config: ExperimentConfig) -> CorpusPaths:
    root = (
        config.artifact_root
        / "corpora"
        / config.model_slug
        / config.dataset_slug
        / config.role_set_slug
    )
    return CorpusPaths(
        root=root,
        meta_dir=root / "meta",
        checkpoints_dir=root / "checkpoints",
        indexes_dir=root / "indexes",
        rollouts_dir=root / "rollouts",
        activations_dir=root / "activations",
        logs_dir=root / "logs",
    )


def analysis_paths(config: ExperimentConfig, run_id: str) -> AnalysisPaths:
    root = (
        config.artifact_root
        / "runs"
        / config.experiment_name
        / config.model_slug
        / config.dataset_slug
        / config.role_set_slug
        / str(config.raw["artifacts"]["analysis_variant"])
        / run_id
    )
    return AnalysisPaths(
        root=root,
        inputs_dir=root / "inputs",
        checkpoints_dir=root / "checkpoints",
        results_dir=root / "results",
        logs_dir=root / "logs",
        meta_dir=root / "meta",
    )


def ensure_corpus_dirs(paths: CorpusPaths) -> None:
    for directory in (
        paths.root,
        paths.meta_dir,
        paths.checkpoints_dir,
        paths.indexes_dir,
        paths.rollouts_dir,
        paths.activations_dir,
        paths.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def ensure_analysis_dirs(paths: AnalysisPaths) -> None:
    for directory in (
        paths.root,
        paths.inputs_dir,
        paths.checkpoints_dir,
        paths.results_dir,
        paths.logs_dir,
        paths.meta_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def write_corpus_manifest(config: ExperimentConfig, paths: CorpusPaths) -> Path:
    manifest_path = paths.meta_dir / "corpus_manifest.json"
    payload = {
        "experiment_name": config.experiment_name,
        "model_id": config.model_id,
        "dataset_name": config.dataset_name,
        "role_manifest": os.path.relpath(config.role_manifest_path, config.repo_root),
        "question_file": os.path.relpath(config.question_file_path, config.repo_root),
        "config_path": os.path.relpath(config.path, config.repo_root),
        "subset": config.subset,
        "updated_at": utc_timestamp(),
    }
    write_json(manifest_path, payload)
    return manifest_path


def write_analysis_manifest(config: ExperimentConfig, paths: AnalysisPaths, run_id: str) -> Path:
    manifest_path = paths.meta_dir / "run_manifest.json"
    payload = {
        "run_id": run_id,
        "experiment_name": config.experiment_name,
        "model_id": config.model_id,
        "dataset_name": config.dataset_name,
        "role_manifest": os.path.relpath(config.role_manifest_path, config.repo_root),
        "question_file": os.path.relpath(config.question_file_path, config.repo_root),
        "config_path": os.path.relpath(config.path, config.repo_root),
        "subset": config.subset,
        "analysis": config.analysis,
        "updated_at": utc_timestamp(),
    }
    write_json(manifest_path, payload)
    return manifest_path


def write_status(path: Path, status: str, extra: dict[str, Any] | None = None) -> None:
    payload = {"status": status, "updated_at": utc_timestamp()}
    if extra:
        payload.update(extra)
    write_json(path, payload)


def append_rollout_shard(paths: CorpusPaths, role_name: str, records: list[dict[str, Any]]) -> str:
    role_dir = paths.rollouts_dir / f"role={role_name}"
    shard_path = _next_part_path(role_dir, "jsonl")
    append_jsonl(shard_path, records)
    return shard_path.relative_to(paths.root).as_posix()


def append_activation_shard(paths: CorpusPaths, role_name: str, records: list[dict[str, Any]]) -> str:
    import torch

    role_dir = paths.activations_dir / f"role={role_name}"
    shard_path = _next_part_path(role_dir, "pt")
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, shard_path)
    return shard_path.relative_to(paths.root).as_posix()


def append_rollout_index(paths: CorpusPaths, rows: Iterable[dict[str, Any]]) -> int:
    return append_jsonl(paths.indexes_dir / "rollouts.jsonl", rows)


def append_activation_index(paths: CorpusPaths, rows: Iterable[dict[str, Any]]) -> int:
    return append_jsonl(paths.indexes_dir / "activations.jsonl", rows)


def load_index_item_ids(index_path: Path) -> set[str]:
    return {row["item_id"] for row in read_jsonl(index_path)}


def load_rollout_index(paths: CorpusPaths) -> list[dict[str, Any]]:
    return read_jsonl(paths.indexes_dir / "rollouts.jsonl")


def load_activation_index(paths: CorpusPaths) -> list[dict[str, Any]]:
    return read_jsonl(paths.indexes_dir / "activations.jsonl")


def update_coverage(
    paths: CorpusPaths,
    target_item_ids: Iterable[str],
    completed_rollouts: set[str],
    completed_activations: set[str],
) -> Path:
    target_ids = sorted(set(target_item_ids))
    coverage_path = paths.meta_dir / "coverage.json"
    payload = {
        "target_item_count": len(target_ids),
        "rollout_count": len(completed_rollouts),
        "activation_count": len(completed_activations),
        "missing_rollouts": sorted(set(target_ids) - completed_rollouts),
        "missing_activations": sorted(set(target_ids) - completed_activations),
        "updated_at": utc_timestamp(),
    }
    write_json(coverage_path, payload)
    return coverage_path


def load_coverage(paths: CorpusPaths) -> dict[str, Any]:
    return read_json(paths.meta_dir / "coverage.json", default={}) or {}
