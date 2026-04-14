from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    prompt_text: str
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


@dataclass(frozen=True)
class ActivationPoolingMetadata:
    prompt_token_count: int
    response_token_count: int
    used_last_token_fallback: bool


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


def normalize_activation_pooling(value: str | None) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return None
    # Activation-row caches and older completion-split caches use different names
    # for the same operation: mean-pool over completion/response tokens.
    if text in {"response_mean", "mean_response", "completion_mean", "mean_completion"}:
        return "mean_response"
    if text in {"last_non_pad_token", "last_non_pad", "last_token"}:
        return "last_token"
    return text


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_token_counts: Sequence[int],
    pooling: str,
) -> tuple[torch.Tensor, list[ActivationPoolingMetadata]]:
    normalized_pooling = normalize_activation_pooling(pooling)
    if normalized_pooling not in {"mean_response", "last_token"}:
        raise ValueError(f"Unsupported activation pooling {pooling!r}")
    if hidden_states.ndim != 3:
        raise ValueError(f"Expected hidden_states shape (B,T,D), got {tuple(hidden_states.shape)}")
    if attention_mask.shape[:2] != hidden_states.shape[:2]:
        raise ValueError(
            "attention_mask must match hidden_states batch/time dimensions: "
            f"mask={tuple(attention_mask.shape)} hidden={tuple(hidden_states.shape)}"
        )
    if len(prompt_token_counts) != int(hidden_states.shape[0]):
        raise ValueError(
            "prompt_token_counts length must equal batch size: "
            f"{len(prompt_token_counts)} != {int(hidden_states.shape[0])}"
        )

    pooled_rows: list[torch.Tensor] = []
    metadata_rows: list[ActivationPoolingMetadata] = []
    non_pad_counts = attention_mask.sum(dim=1).to(dtype=torch.int64)
    for batch_index in range(int(hidden_states.shape[0])):
        total_tokens = int(non_pad_counts[batch_index].item())
        if total_tokens <= 0:
            raise ValueError(f"Encountered empty sequence at batch index {batch_index}")
        prompt_token_count = max(0, min(int(prompt_token_counts[batch_index]), total_tokens))
        response_token_count = max(0, total_tokens - prompt_token_count)
        token_slice = hidden_states[batch_index, :total_tokens, :]

        if normalized_pooling == "mean_response" and response_token_count > 0:
            pooled_rows.append(token_slice[prompt_token_count:total_tokens, :].mean(dim=0))
            metadata_rows.append(
                ActivationPoolingMetadata(
                    prompt_token_count=prompt_token_count,
                    response_token_count=response_token_count,
                    used_last_token_fallback=False,
                )
            )
            continue

        pooled_rows.append(token_slice[total_tokens - 1, :])
        metadata_rows.append(
            ActivationPoolingMetadata(
                prompt_token_count=prompt_token_count,
                response_token_count=response_token_count,
                used_last_token_fallback=normalized_pooling == "mean_response" and response_token_count == 0,
            )
        )

    return torch.stack(pooled_rows, dim=0), metadata_rows


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
    prompt_text = str(record.get("input_formatted") or "")
    if not prompt_text:
        raise ValueError(f"Record {record_id} is missing input_formatted")
    metadata = derive_sycophancy_metadata(record, dataset_source)
    return SycophancyRecord(
        ids=str(record_id),
        pair_id=str(metadata["pair_id"]),
        dataset_source=dataset_source,
        user_framing=str(metadata["user_framing"]),
        system_nudge_direction=str(metadata["system_nudge_direction"]),
        prompt_text=prompt_text,
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
