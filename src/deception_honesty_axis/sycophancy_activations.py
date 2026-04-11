from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


DEFAULT_SOURCE_REPO_ID = "AISC-Linear-Probe-Gen/sycophancy-activations"
DEFAULT_SOURCE_FILES = {
    "factual": "multichoice/llama_3b_incentivised_test.jsonl",
    "evaluative": "arguments/llama_3b_incentivised_test.jsonl",
    "aesthetic": "haikus/llama_3b_incentivised_test.jsonl",
}


@dataclass(frozen=True)
class SycophancyRecord:
    ids: str
    pair_id: str
    dataset_source: str
    user_framing: str
    system_nudge_direction: str
    activation_text: str
    label: int

    def output_without_activation(self) -> dict[str, Any]:
        return {
            "ids": self.ids,
            "pair_id": self.pair_id,
            "dataset_source": self.dataset_source,
            "user_framing": self.user_framing,
            "system_nudge_direction": self.system_nudge_direction,
            "label": self.label,
        }


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return rows


def download_sycophancy_jsonls(
    *,
    repo_id: str = DEFAULT_SOURCE_REPO_ID,
    source_files: dict[str, str] | None = None,
    cache_dir: Path | None = None,
    token: str | None = None,
) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    resolved: dict[str, Path] = {}
    for dataset_source, filename in (source_files or DEFAULT_SOURCE_FILES).items():
        resolved[dataset_source] = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                token=token,
            )
        )
    return resolved


def remap_scale_label(value: Any) -> int:
    label = int(float(value))
    if label == 1:
        return 1
    if label == 10:
        return 0
    raise ValueError(f"Unsupported scale_labels value {value!r}; expected 1 or 10")


def build_activation_text(record: dict[str, Any]) -> str:
    formatted = str(record.get("input_formatted") or "")
    output = str(record.get("model_outputs") or "")
    if not formatted:
        raise ValueError(f"Record {record.get('ids')} is missing input_formatted")
    if output and formatted.rstrip().endswith(output.strip()):
        return formatted
    return f"{formatted}{output}"


def infer_system_nudge_direction(input_formatted: str) -> str:
    text = input_formatted.lower()
    if "agree with the user even if they are wrong" in text:
        return "pro_sycophancy"
    if "don't take the user's opinion into account" in text or "do not take the user's opinion into account" in text:
        return "anti_sycophancy"
    if "self esteem is quite low" in text or "make me feel sad to be told i'm wrong" in text:
        return "emotional_pressure"
    if "self esteem is great" in text or "totally fine with being told i'm wrong" in text:
        return "no_pressure"
    return "unknown"


def infer_user_framing(raw_input: str, dataset_source: str) -> str:
    text = raw_input.lower()
    if dataset_source == "factual":
        match = re.search(r"\bi\s+think\s+(?:the\s+)?(?:answer|correct\s+answer)\s*(?:is|:)?\s*([a-z])\b", text)
        return f"stated_answer_{match.group(1)}" if match else "stated_answer_unknown"

    if dataset_source == "evaluative":
        if re.search(r"\bi\s+(?:did\s+not|didn't)\s+write\s+this\b", text) or "really dislike it" in text:
            return "disliked"
        if re.search(r"\bi\s+wrote\s+this\b", text) or "really like it" in text:
            return "liked"
        return "unknown"

    if dataset_source == "aesthetic":
        match = re.search(r"\bi\s+wrote\s+haiku\s+(?:number\s+)?([12])\b", text)
        if match:
            return f"liked_haiku_{match.group(1)}"
        match = re.search(r"\bi\s+(?:did\s+not|didn't)\s+write\s+haiku\s+(?:number\s+)?([12])\b", text)
        if match:
            return f"disliked_haiku_{match.group(1)}"
        return "unknown"

    return "unknown"


def strip_social_framing(raw_input: str, dataset_source: str) -> str:
    text = raw_input
    if dataset_source == "factual":
        text = re.sub(
            r"\bI\s+think\s+(?:the\s+)?(?:answer|correct\s+answer)\s*(?:is|:)?\s*[A-Za-z]\b\.?",
            "",
            text,
            flags=re.IGNORECASE,
        )
    elif dataset_source == "evaluative":
        text = re.sub(
            r"\bI\s+wrote\s+this(?:\s+argument)?\s+and\s+I\s+really\s+like\s+it\.?",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bI\s+(?:did\s+not|didn't)\s+write\s+this(?:\s+argument)?\s+and\s+I\s+really\s+dislike\s+it\.?",
            "",
            text,
            flags=re.IGNORECASE,
        )
    elif dataset_source == "aesthetic":
        text = re.sub(
            r"\bI\s+wrote\s+haiku\s+(?:number\s+)?[12]\s+and\s+I\s+really\s+like\s+it\.?",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bI\s+(?:did\s+not|didn't)\s+write\s+haiku\s+(?:number\s+)?[12]\s+and\s+I\s+really\s+dislike\s+it\.?",
            "",
            text,
            flags=re.IGNORECASE,
        )
    return text


def normalize_for_pair_id(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def derive_pair_id(raw_input: str, dataset_source: str, system_nudge_direction: str) -> str:
    core = normalize_for_pair_id(strip_social_framing(raw_input, dataset_source))
    payload = f"{dataset_source}|{system_nudge_direction}|{core}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def derive_sycophancy_metadata(record: dict[str, Any], dataset_source: str) -> dict[str, Any]:
    raw_input = str(record.get("input") or "")
    input_formatted = str(record.get("input_formatted") or "")
    if not raw_input:
        raise ValueError(f"Record {record.get('ids')} is missing input")
    system_nudge_direction = infer_system_nudge_direction(input_formatted)
    return {
        "pair_id": derive_pair_id(raw_input, dataset_source, system_nudge_direction),
        "user_framing": infer_user_framing(raw_input, dataset_source),
        "system_nudge_direction": system_nudge_direction,
    }


def prepare_sycophancy_record(record: dict[str, Any], dataset_source: str) -> SycophancyRecord:
    record_id = record.get("ids") or record.get("id")
    if record_id is None:
        raise ValueError("Record is missing ids")
    if "scale_labels" not in record:
        raise ValueError(f"Record {record_id} is missing scale_labels")
    metadata = derive_sycophancy_metadata(record, dataset_source)
    return SycophancyRecord(
        ids=str(record_id),
        pair_id=str(metadata["pair_id"]),
        dataset_source=dataset_source,
        user_framing=str(metadata["user_framing"]),
        system_nudge_direction=str(metadata["system_nudge_direction"]),
        activation_text=build_activation_text(record),
        label=remap_scale_label(record["scale_labels"]),
    )


class SycophancyActivationDataset(Dataset):
    def __init__(self, records_by_source: dict[str, Iterable[dict[str, Any]]]) -> None:
        self.records: list[SycophancyRecord] = []
        for dataset_source in ("factual", "evaluative", "aesthetic"):
            for record in records_by_source.get(dataset_source, []):
                self.records.append(prepare_sycophancy_record(record, dataset_source))

    @classmethod
    def from_jsonl_paths(cls, paths_by_source: dict[str, Path]) -> "SycophancyActivationDataset":
        return cls({
            dataset_source: load_jsonl_records(path)
            for dataset_source, path in paths_by_source.items()
        })

    @classmethod
    def from_hub(
        cls,
        *,
        repo_id: str = DEFAULT_SOURCE_REPO_ID,
        source_files: dict[str, str] | None = None,
        cache_dir: Path | None = None,
        token: str | None = None,
    ) -> "SycophancyActivationDataset":
        return cls.from_jsonl_paths(
            download_sycophancy_jsonls(
                repo_id=repo_id,
                source_files=source_files,
                cache_dir=cache_dir,
                token=token,
            )
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SycophancyRecord:
        return self.records[index]
