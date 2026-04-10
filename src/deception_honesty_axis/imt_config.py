from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deception_honesty_axis.common import ensure_dir, slugify
from deception_honesty_axis.config import find_repo_root, load_config


@dataclass(frozen=True)
class IMTSourceSpec:
    axis_name: str
    experiment_config_path: Path
    analysis_run_id: str | None

    @property
    def experiment_config(self):  # noqa: ANN201
        return load_config(self.experiment_config_path)


@dataclass(frozen=True)
class IMTRecoveryConfig:
    raw: dict[str, Any]
    path: Path
    repo_root: Path

    @property
    def artifact_root(self) -> Path:
        value = self.raw.get("artifacts", {}).get("root", "artifacts")
        path = Path(value)
        resolved = path if path.is_absolute() else (self.repo_root / path)
        ensure_dir(resolved)
        return resolved

    @property
    def hf_repo_id(self) -> str:
        return str(self.raw.get("hf", {}).get("dataset_repo_id", ""))

    @property
    def bank_name(self) -> str:
        return str(self.raw["source_bank"]["name"])

    @property
    def bank_slug(self) -> str:
        return slugify(self.bank_name)

    @property
    def source_specs(self) -> list[IMTSourceSpec]:
        specs: list[IMTSourceSpec] = []
        for entry in self.raw["source_bank"]["sources"]:
            path = Path(entry["experiment_config"])
            resolved = path if path.is_absolute() else (self.repo_root / path).resolve()
            specs.append(
                IMTSourceSpec(
                    axis_name=str(entry["axis_name"]),
                    experiment_config_path=resolved,
                    analysis_run_id=(
                        None
                        if entry.get("analysis_run_id") in (None, "")
                        else str(entry["analysis_run_id"])
                    ),
                )
            )
        return specs

    @property
    def source_experiments(self) -> list[Any]:
        return [spec.experiment_config for spec in self.source_specs]

    @property
    def model_slug(self) -> str:
        return self.source_experiments[0].model_slug

    @property
    def dataset_slug(self) -> str:
        return self.source_experiments[0].dataset_slug

    @property
    def score_model(self) -> dict[str, Any]:
        return dict(self.raw["scoring"]["model"])

    @property
    def score_templates(self) -> list[str]:
        return [str(item) for item in self.raw["scoring"]["templates"]]

    @property
    def role_instruction_max_chars(self) -> int:
        return int(self.raw["scoring"].get("role_instruction_max_chars", 320))

    @property
    def fit_granularity(self) -> str:
        return str(self.raw["fit"].get("granularity", "instance"))

    @property
    def fit_layer_number(self) -> int:
        raw_value = self.raw["fit"].get("layer", 14)
        return int(str(raw_value))

    @property
    def pca_components(self) -> int:
        return int(self.raw["fit"].get("pca_components", 8))

    @property
    def ridge_alpha(self) -> float:
        return float(self.raw["fit"].get("ridge_alpha", 1.0))

    @property
    def target_activations_root(self) -> Path:
        return Path(self.raw["target"]["activations_root"])

    @property
    def target_datasets(self) -> list[str]:
        return [str(item) for item in self.raw["target"]["datasets"]]

    @property
    def eval_split(self) -> str:
        return str(self.raw["target"].get("eval_split", "test"))

    @property
    def pooling(self) -> str:
        return str(self.raw["target"].get("pooling", "completion_mean"))


def load_imt_config(path: str | Path) -> IMTRecoveryConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    repo_root = find_repo_root(config_path.parent)
    config = IMTRecoveryConfig(raw=raw, path=config_path, repo_root=repo_root)
    if not config.source_specs:
        raise ValueError("IMT config must include at least one source bank spec")
    if config.fit_granularity not in {"instance", "role_mean"}:
        raise ValueError(f"Unsupported fit.granularity: {config.fit_granularity}")
    if config.pooling != "completion_mean":
        raise ValueError("IMT recovery currently supports only completion_mean target pooling")
    first = config.source_experiments[0]
    for experiment in config.source_experiments[1:]:
        if experiment.model_id != first.model_id:
            raise ValueError("All IMT source bank experiments must share the same model id")
        if experiment.dataset_name != first.dataset_name:
            raise ValueError("All IMT source bank experiments must share the same dataset name")
    return config


def imt_run_root(config: IMTRecoveryConfig, run_id: str) -> Path:
    root = (
        config.artifact_root
        / "runs"
        / "imt-recovery"
        / config.model_slug
        / config.dataset_slug
        / config.bank_slug
        / run_id
    )
    ensure_dir(root)
    return root
