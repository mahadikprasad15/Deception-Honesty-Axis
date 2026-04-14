from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Callable

from deception_honesty_axis.common import slugify, write_jsonl


EXTERNAL_SYCOPHANCY_ADAPTERS = {
    "gsm8k_sycophancy": "praneethd7/gsm8k_sycophancy",
    "ariana_sycophancy": "arianaazarbal/sycophancy_dataset",
    "open_ended_sycophancy": "henrypapadatos/Open-ended_sycophancy",
    "oeq_prepared": "oeq_prepared",
}

LABEL_NAME_TO_VALUE = {
    "sycophantic": 1,
    "agreeable": 1,
    "reasoning": 0,
    "not_sycophantic": 0,
    "non_sycophantic": 0,
    "honest": 0,
    "truthful": 0,
}

LABEL_CANDIDATE_KEYS = (
    "label",
    "labels",
    "completion_label",
    "completion_type",
    "response_label",
    "response_type",
    "target_label",
    "target_type",
    "variant",
    "kind",
    "type",
)


@dataclass(frozen=True)
class ExternalSycophancyExample:
    ids: str
    pair_id: str
    dataset_source: str
    source_repo_id: str
    source_split: str
    source_record_id: str
    adapter_name: str
    label_name: str
    label: int
    category: str
    prompt_messages: tuple[dict[str, str], ...]
    assistant_output: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest_row(self) -> dict[str, Any]:
        row = {
            "ids": self.ids,
            "pair_id": self.pair_id,
            "dataset_source": self.dataset_source,
            "source_repo_id": self.source_repo_id,
            "source_split": self.source_split,
            "source_record_id": self.source_record_id,
            "adapter_name": self.adapter_name,
            "label_name": self.label_name,
            "label": self.label,
            "category": self.category,
            "prompt_messages": list(self.prompt_messages),
            "assistant_output": self.assistant_output,
        }
        for key, value in self.metadata.items():
            if key not in row:
                row[key] = value
        return row


@dataclass(frozen=True)
class MaterializedExternalSycophancyRecord:
    ids: str
    pair_id: str
    dataset_source: str
    source_repo_id: str
    source_split: str
    source_record_id: str
    adapter_name: str
    label_name: str
    label: int
    category: str
    prompt_text: str
    activation_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def output_without_activation(self) -> dict[str, Any]:
        row = {
            "ids": self.ids,
            "pair_id": self.pair_id,
            "dataset_source": self.dataset_source,
            "source_repo_id": self.source_repo_id,
            "source_split": self.source_split,
            "source_record_id": self.source_record_id,
            "adapter_name": self.adapter_name,
            "label_name": self.label_name,
            "label": self.label,
            "category": self.category,
        }
        for key, value in self.metadata.items():
            if key not in row:
                row[key] = value
        return row


class ExternalSycophancyActivationDataset:
    def __init__(self, records: list[MaterializedExternalSycophancyRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> MaterializedExternalSycophancyRecord:
        return self.records[index]


def _stable_hash(*parts: str) -> str:
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]


def _normalize_label_name(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text


def map_external_label(value: Any) -> tuple[int, str]:
    if isinstance(value, bool):
        return int(value), "sycophantic" if value else "not_sycophantic"
    if isinstance(value, (int, float)) and int(value) in {0, 1}:
        numeric = int(value)
        return numeric, "sycophantic" if numeric == 1 else "not_sycophantic"

    normalized = _normalize_label_name(value)
    if normalized in LABEL_NAME_TO_VALUE:
        return LABEL_NAME_TO_VALUE[normalized], normalized
    raise ValueError(f"Unsupported external sycophancy label {value!r}")


def resolve_external_label_from_row(row: dict[str, Any]) -> tuple[int, str]:
    for key in LABEL_CANDIDATE_KEYS:
        if key not in row:
            continue
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return map_external_label(value)
        except ValueError:
            continue

    discovered: dict[tuple[int, str], str] = {}
    for key, value in row.items():
        if not isinstance(value, str):
            continue
        try:
            mapped = map_external_label(value)
        except ValueError:
            continue
        discovered.setdefault(mapped, key)
    if len(discovered) == 1:
        return next(iter(discovered))
    if len(discovered) > 1:
        raise ValueError(f"Multiple possible external labels found in row: {sorted(discovered.values())}")
    raise ValueError("Could not infer an external sycophancy label from the dataset row")


def _coerce_source_record_id(row: dict[str, Any], row_index: int) -> str:
    for key in ("ids", "id", "row_id", "uuid"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return f"row-{row_index:06d}"


def _require_text(row: dict[str, Any], key: str, *, context: str) -> str:
    value = str(row.get(key) or "")
    if not value:
        raise ValueError(f"Missing {key!r} for {context}")
    return value


def _dataset_source_slug(repo_id: str, split: str) -> str:
    return slugify(f"{repo_id.split('/')[-1]}-{split}")


def _normalize_question_text(text: str) -> str:
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"^\s*#+\s*question\s*", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


_GSM8K_STANCE_PATTERNS = [
    r"\s+The answer most definitely is\s+.+$",
    r"\s+I think the answer is\s+.+$",
    r"\s+I bet the answer is\s+.+$",
    r"\s+It seems like\s+.+$",
]


def _canonicalize_gsm8k_question_text(text: str) -> str:
    normalized = _normalize_question_text(text)
    for pattern in _GSM8K_STANCE_PATTERNS:
        normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _message_list_from_value(value: Any) -> list[dict[str, str]] | None:
    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, list):
        return None

    messages: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            return None
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "")
        if not role:
            return None
        messages.append({"role": role, "content": content})
    return messages or None


def _find_message_list(row: dict[str, Any]) -> list[dict[str, str]]:
    candidate_keys = (
        "messages",
        "conversation",
        "conversations",
        "chat",
        "dialogue",
        "train_SFT",
        "sft_messages",
    )
    for key in candidate_keys:
        parsed = _message_list_from_value(row.get(key))
        if parsed is not None:
            return parsed
    for value in row.values():
        parsed = _message_list_from_value(value)
        if parsed is not None:
            return parsed
    raise ValueError("Could not find a chat message list in the dataset row")


def _split_prompt_messages(messages: list[dict[str, str]]) -> tuple[tuple[dict[str, str], ...], str]:
    if not messages:
        raise ValueError("Expected at least one chat message")
    if messages[-1].get("role") != "assistant":
        raise ValueError("Expected the last chat message to be the target assistant output")
    assistant_output = str(messages[-1].get("content") or "")
    prompt_messages = tuple(
        {
            "role": str(message.get("role") or ""),
            "content": str(message.get("content") or ""),
        }
        for message in messages[:-1]
    )
    if not assistant_output:
        raise ValueError("Assistant output is empty")
    if not prompt_messages:
        raise ValueError("Prompt messages are empty")
    return prompt_messages, assistant_output


def render_chat_text(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:  # noqa: ANN001
    if hasattr(tokenizer, "apply_chat_template"):
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        )
    lines = [f"{message['role'].capitalize()}: {message['content']}" for message in messages]
    if add_generation_prompt:
        lines.append("Assistant:")
    return "\n".join(lines)


def materialize_external_examples(
    tokenizer,  # noqa: ANN001
    examples: list[ExternalSycophancyExample],
) -> list[MaterializedExternalSycophancyRecord]:
    materialized: list[MaterializedExternalSycophancyRecord] = []
    for example in examples:
        prompt_messages = list(example.prompt_messages)
        prompt_text = render_chat_text(tokenizer, prompt_messages, add_generation_prompt=True)
        activation_text = render_chat_text(
            tokenizer,
            prompt_messages + [{"role": "assistant", "content": example.assistant_output}],
            add_generation_prompt=False,
        )
        materialized.append(
            MaterializedExternalSycophancyRecord(
                ids=example.ids,
                pair_id=example.pair_id,
                dataset_source=example.dataset_source,
                source_repo_id=example.source_repo_id,
                source_split=example.source_split,
                source_record_id=example.source_record_id,
                adapter_name=example.adapter_name,
                label_name=example.label_name,
                label=example.label,
                category=example.category,
                prompt_text=prompt_text,
                activation_text=activation_text,
                metadata=dict(example.metadata),
            )
        )
    return materialized


def write_external_example_manifest(run_root: Path, examples: list[ExternalSycophancyExample]) -> None:
    write_jsonl(run_root / "inputs" / "normalized_examples.jsonl", (example.manifest_row() for example in examples))


def _paired_completion_examples(
    *,
    rows: list[dict[str, Any]],
    repo_id: str,
    split: str,
    adapter_name: str,
) -> list[ExternalSycophancyExample]:
    dataset_source = _dataset_source_slug(repo_id, split)
    examples: list[ExternalSycophancyExample] = []
    for row_index, row in enumerate(rows):
        source_record_id = _coerce_source_record_id(row, row_index)
        prompt = _require_text(row, "prompt", context=source_record_id)
        pair_id = _stable_hash(dataset_source, prompt.strip().lower())
        category = str(row.get("category") or "")
        prompt_messages = ({"role": "user", "content": prompt},)
        variants = [
            ("sycophantic", 1, _require_text(row, "sycophantic", context=source_record_id)),
            ("not_sycophantic", 0, _require_text(row, "not_sycophantic", context=source_record_id)),
        ]
        for label_name, label, assistant_output in variants:
            examples.append(
                ExternalSycophancyExample(
                    ids=f"{dataset_source}::{source_record_id}::{label_name}",
                    pair_id=pair_id,
                    dataset_source=dataset_source,
                    source_repo_id=repo_id,
                    source_split=split,
                    source_record_id=source_record_id,
                    adapter_name=adapter_name,
                    label_name=label_name,
                    label=label,
                    category=category,
                    prompt_messages=prompt_messages,
                    assistant_output=assistant_output,
                )
            )
    return examples


def _gsm8k_sycophancy_examples(
    *,
    rows: list[dict[str, Any]],
    repo_id: str,
    split: str,
    adapter_name: str,
) -> list[ExternalSycophancyExample]:
    dataset_source = _dataset_source_slug(repo_id, split)
    examples: list[ExternalSycophancyExample] = []
    for row_index, row in enumerate(rows):
        source_record_id = _coerce_source_record_id(row, row_index)
        prompt_messages, assistant_output = _split_prompt_messages(_find_message_list(row))
        label_value, label_name = resolve_external_label_from_row(row)
        user_prompt = ""
        for message in reversed(prompt_messages):
            if str(message.get("role")) == "user":
                user_prompt = str(message.get("content") or "")
                break
        question_text = str(row.get("question") or row.get("prompt") or user_prompt)
        pair_seed = _canonicalize_gsm8k_question_text(question_text) or json.dumps(list(prompt_messages), sort_keys=True)
        examples.append(
            ExternalSycophancyExample(
                ids=f"{dataset_source}::{source_record_id}",
                pair_id=_stable_hash(dataset_source, pair_seed.strip().lower()),
                dataset_source=dataset_source,
                source_repo_id=repo_id,
                source_split=split,
                source_record_id=source_record_id,
                adapter_name=adapter_name,
                label_name=label_name,
                label=label_value,
                category="",
                prompt_messages=prompt_messages,
                assistant_output=assistant_output,
            )
        )
    return examples


def _prepared_singleton_completion_examples(
    *,
    rows: list[dict[str, Any]],
    repo_id: str,
    split: str,
    adapter_name: str,
) -> list[ExternalSycophancyExample]:
    default_dataset_source = _dataset_source_slug(repo_id, split)
    examples: list[ExternalSycophancyExample] = []
    for row_index, row in enumerate(rows):
        source_record_id = str(
            row.get("source_record_id")
            or row.get("id")
            or row.get("row_id")
            or row.get("Unnamed: 0")
            or row.get("")
            or f"row-{row_index:06d}"
        )
        dataset_source = str(row.get("dataset_source") or default_dataset_source)
        prompt = _require_text(row, "prompt", context=source_record_id)
        assistant_output = str(row.get("assistant_output") or row.get("human") or "")
        if not assistant_output:
            raise ValueError(f"Missing assistant output for {source_record_id}")
        label_value, inferred_label_name = map_external_label(row.get("label"))
        row_label_name = str(row.get("label_name") or "").strip()
        if row_label_name:
            label_name = row_label_name
        else:
            label_name = inferred_label_name
        category = str(row.get("category") or "")
        prompt_messages = ({"role": "user", "content": prompt},)
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {
                "",
                "id",
                "ids",
                "pair_id",
                "dataset_source",
                "source_repo_id",
                "source_split",
                "source_record_id",
                "adapter_name",
                "label_name",
                "label",
                "category",
                "prompt",
                "assistant_output",
                "human",
            }
            and value not in (None, "")
        }
        examples.append(
            ExternalSycophancyExample(
                ids=str(row.get("ids") or f"{dataset_source}::{source_record_id}"),
                pair_id=str(row.get("pair_id") or _stable_hash(dataset_source, source_record_id)),
                dataset_source=dataset_source,
                source_repo_id=repo_id,
                source_split=split,
                source_record_id=source_record_id,
                adapter_name=adapter_name,
                label_name=label_name,
                label=label_value,
                category=category,
                prompt_messages=prompt_messages,
                assistant_output=assistant_output,
                metadata=metadata,
            )
        )
    return examples


def cap_external_examples(
    examples: list[ExternalSycophancyExample],
    max_samples: int | None,
) -> list[ExternalSycophancyExample]:
    if max_samples is None:
        return examples
    limit = int(max_samples)
    if limit <= 0 or len(examples) <= limit:
        return examples

    capped = list(examples[:limit])
    total_by_pair = {example.pair_id: 0 for example in examples}
    capped_by_pair = {example.pair_id: 0 for example in capped}
    for example in examples:
        total_by_pair[example.pair_id] = total_by_pair.get(example.pair_id, 0) + 1
    for example in capped:
        capped_by_pair[example.pair_id] = capped_by_pair.get(example.pair_id, 0) + 1

    incomplete_pair_ids = {
        pair_id
        for pair_id, capped_count in capped_by_pair.items()
        if capped_count < total_by_pair.get(pair_id, capped_count)
    }
    if incomplete_pair_ids:
        capped = [example for example in capped if example.pair_id not in incomplete_pair_ids]
    return capped


def summarize_external_examples(examples: list[ExternalSycophancyExample]) -> dict[str, Any]:
    pair_groups: dict[str, list[ExternalSycophancyExample]] = {}
    for example in examples:
        pair_groups.setdefault(example.pair_id, []).append(example)

    pair_size_histogram = Counter(len(group) for group in pair_groups.values())
    pair_label_pattern_counts = Counter(
        ",".join(str(example.label) for example in sorted(group, key=lambda row: (row.label, row.ids)))
        for group in pair_groups.values()
    )
    complete_balanced_pair_count = sum(
        1
        for group in pair_groups.values()
        if len(group) == 2 and sorted(example.label for example in group) == [0, 1]
    )
    return {
        "record_count": len(examples),
        "unique_pair_count": len(pair_groups),
        "dataset_source_counts": dict(sorted(Counter(example.dataset_source for example in examples).items())),
        "label_counts": dict(sorted(Counter(int(example.label) for example in examples).items())),
        "label_name_counts": dict(sorted(Counter(example.label_name for example in examples).items())),
        "category_counts": dict(sorted(Counter(example.category for example in examples if example.category).items())),
        "pair_count_histogram": dict(sorted(pair_size_histogram.items())),
        "pair_label_pattern_counts": dict(sorted(pair_label_pattern_counts.items())),
        "complete_balanced_pair_count": int(complete_balanced_pair_count),
        "complete_balanced_pair_record_count": int(complete_balanced_pair_count * 2),
    }


def sample_external_examples(
    examples: list[ExternalSycophancyExample],
    *,
    max_samples: int | None,
    balanced_labels: bool,
    sample_seed: int,
) -> list[ExternalSycophancyExample]:
    if max_samples is None and not balanced_labels:
        return examples
    if not examples:
        return examples

    rng = random.Random(int(sample_seed))
    groups: dict[str, list[ExternalSycophancyExample]] = {}
    for example in examples:
        groups.setdefault(example.pair_id, []).append(example)

    paired_groups = [
        group
        for group in groups.values()
        if len(group) == 2 and sorted(example.label for example in group) == [0, 1]
    ]
    limit = None if max_samples is None else max(0, int(max_samples))

    if len(paired_groups) == len(groups):
        rng.shuffle(paired_groups)
        if limit is None:
            selected_groups = paired_groups
        else:
            pair_limit = limit // 2
            selected_groups = paired_groups[:pair_limit]
        selected_examples = [example for group in selected_groups for example in group]
        if balanced_labels and limit is not None and len(selected_examples) > limit:
            selected_examples = selected_examples[:limit - (limit % 2)]
        return sorted(selected_examples, key=lambda example: example.ids)

    shuffled_examples = list(examples)
    rng.shuffle(shuffled_examples)
    if balanced_labels and paired_groups and limit is not None:
        pair_groups = list(paired_groups)
        rng.shuffle(pair_groups)
        pair_limit = limit // 2
        selected_from_pairs = pair_groups[:pair_limit]
        selected_examples = [example for group in selected_from_pairs for example in group]
        selected_ids = {example.ids for example in selected_examples}
        remaining_slots = limit - len(selected_examples)
        if remaining_slots <= 0:
            return sorted(selected_examples, key=lambda example: example.ids)

        per_label_limit = remaining_slots // 2
        supplemental_counts = {0: 0, 1: 0}
        supplemental: list[ExternalSycophancyExample] = []
        for example in shuffled_examples:
            if example.ids in selected_ids:
                continue
            if supplemental_counts[example.label] >= per_label_limit:
                continue
            supplemental_counts[example.label] += 1
            supplemental.append(example)
            if supplemental_counts[0] >= per_label_limit and supplemental_counts[1] >= per_label_limit:
                break
        return sorted(selected_examples + supplemental, key=lambda example: example.ids)

    if balanced_labels:
        if limit is None:
            label_counts = {0: 0, 1: 0}
            selected: list[ExternalSycophancyExample] = []
            for example in shuffled_examples:
                label_counts[example.label] += 1
                selected.append(example)
            target_per_label = min(label_counts.values()) if label_counts else 0
            selected = []
            label_counts = {0: 0, 1: 0}
            for example in shuffled_examples:
                if label_counts[example.label] >= target_per_label:
                    continue
                label_counts[example.label] += 1
                selected.append(example)
            return sorted(selected, key=lambda example: example.ids)

        limit -= limit % 2
        per_label_limit = max(0, limit // 2)
        selected: list[ExternalSycophancyExample] = []
        label_counts = {0: 0, 1: 0}
        for example in shuffled_examples:
            if label_counts[example.label] >= per_label_limit:
                continue
            label_counts[example.label] += 1
            selected.append(example)
            if label_counts[0] >= per_label_limit and label_counts[1] >= per_label_limit:
                break
        return sorted(selected, key=lambda example: example.ids)

    if limit is None:
        return shuffled_examples
    return sorted(shuffled_examples[:limit], key=lambda example: example.ids)


ADAPTER_BUILDERS: dict[str, Callable[..., list[ExternalSycophancyExample]]] = {
    "gsm8k_sycophancy": _gsm8k_sycophancy_examples,
    "ariana_sycophancy": _paired_completion_examples,
    "open_ended_sycophancy": _paired_completion_examples,
    "oeq_prepared": _prepared_singleton_completion_examples,
}


def resolve_external_sycophancy_adapter(adapter_name: str | None, repo_id: str) -> str:
    if adapter_name:
        normalized = str(adapter_name).strip()
        if normalized not in ADAPTER_BUILDERS:
            raise ValueError(f"Unsupported adapter {adapter_name!r}; choices={sorted(ADAPTER_BUILDERS)}")
        return normalized
    for candidate_name, candidate_repo_id in EXTERNAL_SYCOPHANCY_ADAPTERS.items():
        if candidate_repo_id == repo_id:
            return candidate_name
    if repo_id.split("/")[-1].startswith("oeq-"):
        return "oeq_prepared"
    raise ValueError(f"Could not infer adapter for repo_id={repo_id!r}; pass --adapter explicitly")


def load_external_sycophancy_examples(
    *,
    repo_id: str,
    split: str,
    adapter_name: str | None = None,
    max_samples: int | None = None,
    balanced_labels: bool = False,
    sample_seed: int = 42,
) -> tuple[list[ExternalSycophancyExample], dict[str, Any]]:
    from datasets import load_dataset

    resolved_adapter = resolve_external_sycophancy_adapter(adapter_name, repo_id)
    dataset = load_dataset(repo_id, split=split, token=os.environ.get("HF_TOKEN"))
    rows = [dict(row) for row in dataset]
    examples = ADAPTER_BUILDERS[resolved_adapter](
        rows=rows,
        repo_id=repo_id,
        split=split,
        adapter_name=resolved_adapter,
    )
    capped_examples = sample_external_examples(
        examples,
        max_samples=max_samples,
        balanced_labels=balanced_labels,
        sample_seed=sample_seed,
    )
    manifest = {
        "source_type": "hf_dataset",
        "repo_id": repo_id,
        "split": split,
        "adapter_name": resolved_adapter,
        "dataset_source": _dataset_source_slug(repo_id, split),
        "raw_row_count": len(rows),
        "normalized_record_count": len(examples),
        "max_samples": max_samples,
        "balanced_labels": bool(balanced_labels),
        "sample_seed": int(sample_seed),
        "selected_record_count": len(capped_examples),
    }
    return capped_examples, manifest
