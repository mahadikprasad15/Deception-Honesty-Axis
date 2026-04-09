from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from .config import ExperimentConfig


@dataclass(frozen=True)
class RoleDefinition:
    name: str
    source: str
    anchor_only: bool
    metadata: dict[str, Any]


@dataclass(frozen=True)
class QuestionRecord:
    question_id: str
    question: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class WorkUnit:
    item_id: str
    role_name: str
    role_source: str
    role_metadata: dict[str, Any]
    prompt_id: int
    prompt_text: str
    question_id: str
    question_offset: int
    question_text: str
    question_metadata: dict[str, Any]
    anchor_only: bool


def load_role_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def role_definitions(config: ExperimentConfig) -> list[RoleDefinition]:
    manifest = load_role_manifest(config.role_manifest_path)
    return [
        RoleDefinition(
            name=str(role["name"]),
            source=str(role.get("source") or Path(role["instruction_file"]).name),
            anchor_only=bool(role.get("anchor_only", False)),
            metadata={
                key: value
                for key, value in dict(role).items()
                if key not in {"name", "source", "instruction_file", "anchor_only"}
            },
        )
        for role in manifest["roles"]
    ]


def pca_roles(config: ExperimentConfig) -> list[str]:
    manifest = load_role_manifest(config.role_manifest_path)
    if "pca_roles" in manifest:
        return [str(role_name) for role_name in manifest["pca_roles"]]
    return [role.name for role in role_definitions(config) if not role.anchor_only]


def anchor_role(config: ExperimentConfig) -> str:
    manifest = load_role_manifest(config.role_manifest_path)
    return str(manifest["anchor_role"])


def _extract_questions(raw: dict[str, Any], index: int) -> QuestionRecord:
    question_id = str(raw.get("question_id") or raw.get("id") or raw.get("name") or f"q{index:03d}")
    question = raw.get("question") or raw.get("prompt") or raw.get("text")
    if not isinstance(question, str):
        for value in raw.values():
            if isinstance(value, str) and value.strip():
                question = value
                break
    if not isinstance(question, str):
        raise ValueError(f"Could not extract question text from row {index}: {raw}")
    return QuestionRecord(question_id=question_id, question=question.strip(), metadata=dict(raw))


def load_questions(question_file: Path) -> list[QuestionRecord]:
    rows: list[QuestionRecord] = []
    with question_file.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(_extract_questions(json.loads(stripped), index))
    return rows


def _stringify_prompt(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("instruction", "prompt", "system_prompt", "text", "content", "message", "pos"):
            value = item.get(key)
            if isinstance(value, str):
                return value
    raise ValueError(f"Unsupported prompt payload: {item!r}")


def load_role_prompts(role_path: Path) -> list[str]:
    with role_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if isinstance(raw, list):
        prompts = [_stringify_prompt(item) for item in raw]
    elif isinstance(raw, dict):
        prompt_container = None
        for key in (
            "instructions",
            "prompts",
            "system_prompts",
            "variants",
            "instruction_variants",
            "messages",
            "instruction",
        ):
            if key in raw:
                prompt_container = raw[key]
                break
        if prompt_container is None:
            if any(key in raw for key in ("instruction", "prompt", "system_prompt", "text", "content", "message")):
                prompt_container = [raw]
            else:
                raise ValueError(f"Unsupported role file structure: {role_path}")
        prompts = [_stringify_prompt(item) for item in prompt_container]
    else:
        raise ValueError(f"Unsupported role file payload in {role_path}")

    return [prompt.strip() for prompt in prompts if prompt and prompt.strip()]


T = TypeVar("T")


def select_first_n(items: list[T], count: int) -> list[T]:
    return items[: min(len(items), count)]


def expand_work_units(config: ExperimentConfig) -> list[WorkUnit]:
    subset = config.subset
    if subset["instruction_selection"] != "first_n" or subset["question_selection"] != "first_n":
        raise ValueError("Only first_n selection is implemented in this scaffold.")

    questions = select_first_n(load_questions(config.question_file_path), int(subset["question_count"]))
    units: list[WorkUnit] = []

    for role in role_definitions(config):
        role_path = config.repo_root / "data" / "roles" / "instructions" / role.source
        prompts = select_first_n(load_role_prompts(role_path), int(subset["instruction_count"]))
        for prompt_id, prompt_text in enumerate(prompts):
            for question_offset, question in enumerate(questions):
                units.append(
                    WorkUnit(
                        item_id=f"{role.name}__p{prompt_id:02d}__q{question_offset:03d}",
                        role_name=role.name,
                        role_source=role.source,
                        role_metadata=dict(role.metadata),
                        prompt_id=prompt_id,
                        prompt_text=prompt_text,
                        question_id=question.question_id,
                        question_offset=question_offset,
                        question_text=question.question,
                        question_metadata=dict(question.metadata),
                        anchor_only=role.anchor_only,
                    )
                )
    return units


def build_work_units(repo_root: Path, config: dict[str, Any] | ExperimentConfig) -> list[dict[str, Any]]:
    experiment_config = config if isinstance(config, ExperimentConfig) else ExperimentConfig(raw=config, path=Path(config["_config_path"]), repo_root=repo_root)
    return [
        {
            "item_id": unit.item_id,
            "role": unit.role_name,
            "role_source": unit.role_source,
            "role_metadata": dict(unit.role_metadata),
            "anchor_only": unit.anchor_only,
            "prompt_id": unit.prompt_id,
            "system_prompt": unit.prompt_text,
            "question_id": unit.question_id,
            "question_offset": unit.question_offset,
            "question_text": unit.question_text,
            "question_metadata": dict(unit.question_metadata),
        }
        for unit in expand_work_units(experiment_config)
    ]
