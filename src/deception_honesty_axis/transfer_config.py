from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deception_honesty_axis.common import ensure_dir
from deception_honesty_axis.config import find_repo_root, load_config


@dataclass(frozen=True)
class RoleAxisTransferConfig:
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
        return str(self.raw["hf"]["dataset_repo_id"])

    @property
    def role_experiment_config_path(self) -> Path:
        value = self.raw["role_axis"]["experiment_config"]
        path = Path(value)
        return path if path.is_absolute() else (self.repo_root / path)

    @property
    def role_experiment_config(self):  # noqa: ANN201
        return load_config(self.role_experiment_config_path)

    @property
    def layer_specs(self) -> list[str]:
        return [str(item) for item in self.raw["role_axis"]["layer_specs"]]

    @property
    def pc_count(self) -> int:
        return int(self.raw["role_axis"].get("pc_count", 3))

    @property
    def honest_roles(self) -> list[str]:
        return [str(item) for item in self.raw["role_axis"]["honest_roles"]]

    @property
    def deceptive_roles(self) -> list[str]:
        return [str(item) for item in self.raw["role_axis"]["deceptive_roles"]]

    @property
    def anchor_role(self) -> str:
        return str(self.raw["role_axis"]["anchor_role"])

    @property
    def target_activations_root(self) -> Path:
        return Path(self.raw["target"]["activations_root"])

    @property
    def target_datasets(self) -> list[str]:
        return [str(item) for item in self.raw["target"]["datasets"]]

    @property
    def source_split(self) -> str:
        return str(self.raw["target"].get("source_split", "train"))

    @property
    def eval_split(self) -> str:
        return str(self.raw["target"].get("eval_split", "test"))

    @property
    def pooling(self) -> str:
        return str(self.raw["target"].get("pooling", "completion_mean"))

    @property
    def methods(self) -> list[str]:
        return [str(item) for item in self.raw["evaluation"]["methods"]]

    @property
    def random_seed(self) -> int:
        return int(self.raw["evaluation"].get("random_seed", 42))

    @property
    def logistic_max_iter(self) -> int:
        return int(self.raw["evaluation"].get("logistic_max_iter", 1000))


def load_transfer_config(path: str | Path) -> RoleAxisTransferConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    repo_root = find_repo_root(config_path.parent)
    return RoleAxisTransferConfig(raw=raw, path=config_path, repo_root=repo_root)
