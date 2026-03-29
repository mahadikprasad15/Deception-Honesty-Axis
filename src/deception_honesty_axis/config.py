from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import ensure_dir


def find_repo_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError("Could not locate repo root from current working directory.")


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    pieces = [char if char.isalnum() else "-" for char in lowered]
    slug = "".join(pieces)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]
    path: Path
    repo_root: Path

    @property
    def experiment_name(self) -> str:
        return str(self.raw["experiment_name"])

    @property
    def model_id(self) -> str:
        model = self.raw["model"]
        return str(model.get("id") or model.get("name"))

    @property
    def model_slug(self) -> str:
        return slugify(self.model_id)

    @property
    def dataset_name(self) -> str:
        if "dataset" in self.raw:
            return str(self.raw["dataset"]["name"])
        return str(self.raw["artifacts"]["dataset_name"])

    @property
    def dataset_slug(self) -> str:
        return slugify(self.dataset_name)

    @property
    def role_set_slug(self) -> str:
        manifest = self.role_manifest_path
        return slugify(manifest.stem)

    @property
    def role_manifest_path(self) -> Path:
        if "roles" in self.raw:
            value = self.raw["roles"]["manifest"]
        else:
            value = self.raw["data"]["role_manifest_file"]
        return (self.repo_root / value).resolve()

    @property
    def question_file_path(self) -> Path:
        if "dataset" in self.raw:
            value = self.raw["dataset"]["question_file"]
        else:
            value = self.raw["data"]["question_file"]
        return (self.repo_root / value).resolve()

    @property
    def artifact_root(self) -> Path:
        if "storage" in self.raw:
            value = self.raw["storage"]["artifact_root"]
        else:
            value = self.raw["artifacts"]["root"]
        return (self.repo_root / value).resolve()

    @property
    def subset(self) -> dict[str, Any]:
        if "subset" in self.raw:
            return dict(self.raw["subset"])
        return {
            "instruction_count": self.raw["roles"]["instruction_count"],
            "instruction_selection": self.raw["roles"]["instruction_selection"],
            "question_count": self.raw["dataset"]["question_count"],
            "question_selection": self.raw["dataset"]["question_selection"],
        }

    @property
    def generation(self) -> dict[str, Any]:
        model = self.raw["model"]
        if "generation" in model:
            return dict(model["generation"])
        return {
            "max_new_tokens": model["max_new_tokens"],
            "temperature": model["temperature"],
            "top_p": model["top_p"],
            "do_sample": model["do_sample"],
        }

    @property
    def saving(self) -> dict[str, Any]:
        if "saving" in self.raw:
            return dict(self.raw["saving"])
        return {"save_every": self.raw["storage"]["save_every"]}

    @property
    def analysis(self) -> dict[str, Any]:
        return dict(self.raw["analysis"])

    @property
    def hf(self) -> dict[str, Any]:
        if "hf" in self.raw:
            return dict(self.raw["hf"])
        return {"dataset_repo": self.raw["storage"]["hf_dataset_repo"]}


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    repo_root = find_repo_root(config_path.parent)
    return ExperimentConfig(raw=raw, path=config_path, repo_root=repo_root)


def resolve_repo_path(config: ExperimentConfig, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config.repo_root / path)


def corpus_root(config: ExperimentConfig) -> Path:
    root = (
        config.artifact_root
        / "corpora"
        / config.model_slug
        / config.dataset_slug
        / config.role_set_slug
    )
    ensure_dir(root)
    return root


def analysis_run_root(
    config: ExperimentConfig,
    experiment_name: str,
    variant: str,
    run_id: str,
) -> Path:
    root = (
        config.artifact_root
        / "runs"
        / experiment_name
        / config.model_slug
        / config.dataset_slug
        / config.role_set_slug
        / variant
        / run_id
    )
    ensure_dir(root)
    return root
