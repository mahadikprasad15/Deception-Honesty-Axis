from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Any

from deception_honesty_axis.common import slugify


OEQ_TASK_LABEL_FIELDS = (
    "validation_human",
    "indirectness_human",
    "framing_human",
)


def normalize_oeq_binary_flag(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in {"1", "1.0"}:
        return 1
    if text in {"0", "0.0"}:
        return 0
    raise ValueError(f"Unsupported OEQ binary label value {value!r}")


def load_oeq_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_oeq_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        source_record_id = str(
            row.get("Unnamed: 0")
            or row.get("")
            or row.get("id")
            or row.get("row_id")
            or row_index
        )
        prompt = str(row.get("prompt") or "").strip()
        human = str(row.get("human") or "").strip()
        if not prompt or not human:
            continue

        normalized = {
            "source_record_id": source_record_id,
            "prompt": prompt,
            "human": human,
            "assistant_output": human,
            "source": str(row.get("source") or "").strip(),
        }
        positives = 0
        for field_name in OEQ_TASK_LABEL_FIELDS:
            normalized_value = normalize_oeq_binary_flag(row.get(field_name))
            normalized[field_name] = normalized_value
            positives += int(normalized_value == 1)
        normalized["sycophantic_any"] = 1 if positives >= 1 else 0
        normalized_rows.append(normalized)
    return normalized_rows


def is_oeq_clean_negative(row: dict[str, Any]) -> bool:
    return all(row.get(field_name) == 0 for field_name in OEQ_TASK_LABEL_FIELDS)


def _task_sample_seed(sample_seed: int, target_label: str) -> int:
    digest = hashlib.sha256(f"{sample_seed}:{target_label}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def prepare_oeq_task_rows(
    rows: list[dict[str, Any]],
    *,
    target_label: str,
    positive_count: int,
    negative_count: int,
    sample_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_label not in OEQ_TASK_LABEL_FIELDS:
        raise ValueError(f"Unsupported OEQ target label {target_label!r}; choices={OEQ_TASK_LABEL_FIELDS!r}")

    positives = [row for row in rows if row.get(target_label) == 1]
    negatives = [row for row in rows if is_oeq_clean_negative(row)]
    if len(positives) < int(positive_count):
        raise ValueError(
            f"Not enough positive rows for {target_label}: required={positive_count}, available={len(positives)}"
        )
    if len(negatives) < int(negative_count):
        raise ValueError(
            f"Not enough clean negative rows for {target_label}: required={negative_count}, available={len(negatives)}"
        )

    task_seed = _task_sample_seed(int(sample_seed), target_label)
    rng = random.Random(task_seed)
    selected_positives = rng.sample(positives, int(positive_count))
    selected_negatives = rng.sample(negatives, int(negative_count))

    dataset_source = f"oeq-{slugify(target_label)}-balanced-{int(positive_count) + int(negative_count)}"
    prepared_rows: list[dict[str, Any]] = []
    for label_value, selected_rows in ((1, selected_positives), (0, selected_negatives)):
        label_name = target_label if label_value == 1 else f"not_{target_label}"
        for row in selected_rows:
            prepared_rows.append(
                {
                    "id": f"{dataset_source}::{row['source_record_id']}",
                    "ids": f"{dataset_source}::{row['source_record_id']}",
                    "dataset_source": dataset_source,
                    "source_record_id": str(row["source_record_id"]),
                    "task_label_field": target_label,
                    "prompt": row["prompt"],
                    "human": row["human"],
                    "assistant_output": row["assistant_output"],
                    "source": row.get("source", ""),
                    "label": int(label_value),
                    "label_name": label_name,
                    "validation_human": row.get("validation_human"),
                    "indirectness_human": row.get("indirectness_human"),
                    "framing_human": row.get("framing_human"),
                    "sycophantic_any": row.get("sycophantic_any"),
                }
            )
    rng.shuffle(prepared_rows)
    summary = {
        "target_label": target_label,
        "dataset_source": dataset_source,
        "task_sample_seed": int(task_seed),
        "record_count": len(prepared_rows),
        "label_counts": dict(sorted(Counter(int(row["label"]) for row in prepared_rows).items())),
        "source_counts": dict(sorted(Counter(str(row.get("source") or "") for row in prepared_rows).items())),
        "positive_pool_size": len(positives),
        "negative_pool_size": len(negatives),
    }
    return prepared_rows, summary


def summarize_oeq_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive_count_histogram = Counter(int(sum(int(row.get(field_name) == 1) for field_name in OEQ_TASK_LABEL_FIELDS)) for row in rows)
    return {
        "record_count": len(rows),
        "source_counts": dict(sorted(Counter(str(row.get("source") or "") for row in rows).items())),
        "label_counts": {
            field_name: {
                "0": int(sum(row.get(field_name) == 0 for row in rows)),
                "1": int(sum(row.get(field_name) == 1 for row in rows)),
                "missing": int(sum(row.get(field_name) is None for row in rows)),
            }
            for field_name in OEQ_TASK_LABEL_FIELDS
        },
        "clean_negative_count": int(sum(is_oeq_clean_negative(row) for row in rows)),
        "sycophantic_any_count": int(sum(int(row.get("sycophantic_any") == 1) for row in rows)),
        "positive_count_histogram": dict(sorted(positive_count_histogram.items())),
    }
