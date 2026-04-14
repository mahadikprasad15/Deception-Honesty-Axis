from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deception_honesty_axis.common import ensure_dir
from deception_honesty_axis.config import find_repo_root


SUPPORTED_ACTIVATION_ROW_TRANSFER_METHODS = {"activation_logistic"}
SUPPORTED_SOURCE_KINDS = {"activation_rows", "completion_split"}


@dataclass(frozen=True)
class ActivationRowSourceConfig:
    raw: dict[str, Any]
    repo_root: Path

    def _resolve_optional_path(self, key: str) -> Path | None:
        value = self.raw.get(key)
        if value in (None, ""):
            return None
        path = Path(str(value))
        return path if path.is_absolute() else (self.repo_root / path)

    @property
    def source_kind(self) -> str:
        value = self.raw.get("source_kind", "activation_rows")
        resolved = str(value)
        if resolved not in SUPPORTED_SOURCE_KINDS:
            raise ValueError(
                f"Unsupported source_kind {resolved!r}; expected one of {sorted(SUPPORTED_SOURCE_KINDS)!r}"
            )
        return resolved

    @property
    def activation_dataset_repo_id(self) -> str | None:
        value = self.raw.get("activation_dataset_repo_id")
        if value in (None, ""):
            return None
        return str(value)

    @property
    def activation_dataset_dir(self) -> Path | None:
        return self._resolve_optional_path("activation_dataset_dir")

    @property
    def activation_jsonl(self) -> Path | None:
        return self._resolve_optional_path("activation_jsonl")

    @property
    def split_dir(self) -> Path | None:
        return self._resolve_optional_path("split_dir")

    @property
    def split(self) -> str:
        if "split" in self.raw and self.raw.get("split") not in (None, ""):
            return str(self.raw["split"])
        if self.source_kind == "completion_split" and self.split_dir is not None:
            return str(self.split_dir.name)
        return "train"

    @property
    def group_field(self) -> str:
        return str(self.raw.get("group_field", "dataset_source"))

    @property
    def group_value(self) -> str | None:
        value = self.raw.get("group_value")
        if value in (None, ""):
            return None
        return str(value)

    def resolved_load_kwargs(self) -> dict[str, Any]:
        if self.source_kind != "activation_rows":
            raise ValueError(f"resolved_load_kwargs is only valid for activation_rows, got {self.source_kind!r}")
        sources = {
            "activation_dataset_repo_id": self.activation_dataset_repo_id,
            "activation_dataset_dir": self.activation_dataset_dir,
            "activation_jsonl": self.activation_jsonl,
        }
        present = [key for key, value in sources.items() if value is not None]
        if len(present) != 1:
            raise ValueError(
                "Each activation-row source must specify exactly one of "
                "activation_dataset_repo_id, activation_dataset_dir, or activation_jsonl; "
                f"found {present!r}"
            )
        kwargs = {key: value for key, value in sources.items() if value is not None}
        kwargs["split"] = self.split
        return kwargs

    def manifest_row(self) -> dict[str, Any]:
        payload = {
            "source_kind": self.source_kind,
            "split": self.split,
            "group_field": self.group_field,
            "group_value": self.group_value,
        }
        if self.split_dir is not None:
            payload["split_dir"] = str(self.split_dir)
        if self.activation_dataset_repo_id is not None:
            payload["activation_dataset_repo_id"] = self.activation_dataset_repo_id
        if self.activation_dataset_dir is not None:
            payload["activation_dataset_dir"] = str(self.activation_dataset_dir)
        if self.activation_jsonl is not None:
            payload["activation_jsonl"] = str(self.activation_jsonl)
        return payload


@dataclass(frozen=True)
class ActivationRowDatasetConfig:
    raw: dict[str, Any]
    repo_root: Path

    @property
    def name(self) -> str:
        value = self.raw.get("name")
        if value in (None, ""):
            raise ValueError("Each dataset entry requires a non-empty 'name'")
        return str(value)

    def _shared_source_payload(self) -> dict[str, Any] | None:
        if isinstance(self.raw.get("source"), dict):
            return dict(self.raw["source"])
        top_level_keys = {
            "source_kind",
            "split_dir",
            "activation_dataset_repo_id",
            "activation_dataset_dir",
            "activation_jsonl",
            "split",
            "group_field",
            "group_value",
        }
        if any(key in self.raw for key in top_level_keys):
            return {key: self.raw[key] for key in top_level_keys if key in self.raw}
        return None

    @property
    def behavior(self) -> str | None:
        value = self.raw.get("behavior")
        if value in (None, ""):
            return None
        return str(value)

    @property
    def train_source(self) -> ActivationRowSourceConfig:
        if isinstance(self.raw.get("train"), dict):
            return ActivationRowSourceConfig(dict(self.raw["train"]), self.repo_root)
        shared = self._shared_source_payload()
        if shared is None and isinstance(self.raw.get("eval"), dict):
            return ActivationRowSourceConfig(dict(self.raw["eval"]), self.repo_root)
        if shared is None:
            raise ValueError(
                f"Dataset {self.name!r} must define either a shared source, "
                "or a nested train/eval source configuration"
            )
        return ActivationRowSourceConfig(shared, self.repo_root)

    @property
    def eval_source(self) -> ActivationRowSourceConfig:
        if isinstance(self.raw.get("eval"), dict):
            return ActivationRowSourceConfig(dict(self.raw["eval"]), self.repo_root)
        shared = self._shared_source_payload()
        if shared is None and isinstance(self.raw.get("train"), dict):
            return ActivationRowSourceConfig(dict(self.raw["train"]), self.repo_root)
        if shared is None:
            raise ValueError(
                f"Dataset {self.name!r} must define either a shared source, "
                "or a nested train/eval source configuration"
            )
        return ActivationRowSourceConfig(shared, self.repo_root)

    @property
    def uses_shared_source(self) -> bool:
        train_payload = self.train_source.manifest_row()
        eval_payload = self.eval_source.manifest_row()
        return train_payload == eval_payload

    def manifest_row(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "behavior": self.behavior,
            "uses_shared_source": self.uses_shared_source,
            "train_source": self.train_source.manifest_row(),
            "eval_source": self.eval_source.manifest_row(),
        }
        return payload


@dataclass(frozen=True)
class ActivationRowTransferConfig:
    raw: dict[str, Any]
    path: Path
    repo_root: Path

    @property
    def artifact_root(self) -> Path:
        value = self.raw.get("artifacts", {}).get("root", "artifacts")
        path = Path(str(value))
        resolved = path if path.is_absolute() else (self.repo_root / path)
        ensure_dir(resolved)
        return resolved

    @property
    def experiment_name(self) -> str:
        return str(self.raw.get("experiment", {}).get("name", "activation-row-transfer"))

    @property
    def behavior_name(self) -> str:
        value = self.raw.get("experiment", {}).get("behavior")
        if value in (None, ""):
            raise ValueError("Activation-row transfer config requires experiment.behavior")
        return str(value)

    @property
    def model_name(self) -> str:
        value = self.raw.get("experiment", {}).get("model_name")
        if value in (None, ""):
            raise ValueError("Activation-row transfer config requires experiment.model_name")
        return str(value)

    @property
    def dataset_set_name(self) -> str:
        return str(self.raw.get("experiment", {}).get("dataset_set", self.behavior_name))

    @property
    def expected_pooling(self) -> str | None:
        value = self.raw.get("activation_rows", {}).get("expected_pooling")
        if value in (None, ""):
            return None
        return str(value)

    @property
    def expected_layer_number(self) -> int | None:
        value = self.raw.get("activation_rows", {}).get("expected_layer_number")
        if value in (None, ""):
            return None
        return int(value)

    @property
    def datasets(self) -> list[ActivationRowDatasetConfig]:
        raw_datasets = self.raw.get("datasets")
        if not isinstance(raw_datasets, list) or not raw_datasets:
            raise ValueError("Activation-row transfer config requires a non-empty datasets list")
        return [ActivationRowDatasetConfig(dict(item), self.repo_root) for item in raw_datasets]

    @property
    def methods(self) -> list[str]:
        values = self.raw.get("evaluation", {}).get("methods", ["activation_logistic"])
        methods = [str(value) for value in values]
        invalid = [method for method in methods if method not in SUPPORTED_ACTIVATION_ROW_TRANSFER_METHODS]
        if invalid:
            raise ValueError(f"Unsupported activation-row transfer methods: {invalid}")
        return methods

    @property
    def random_seed(self) -> int:
        return int(self.raw.get("evaluation", {}).get("random_seed", 42))

    @property
    def logistic_max_iter(self) -> int:
        return int(self.raw.get("evaluation", {}).get("logistic_max_iter", 1000))


def load_activation_row_transfer_config(path: str | Path) -> ActivationRowTransferConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    repo_root = find_repo_root(config_path.parent)
    return ActivationRowTransferConfig(raw=raw, path=config_path, repo_root=repo_root)
