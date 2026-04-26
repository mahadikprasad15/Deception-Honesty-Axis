#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Subset

from deception_honesty_axis.activation_extraction import (
    dtype_from_name,
    extract_batch_activations,
    load_tokenizer_and_model,
    normalize_saved_records,
    save_activation_rows_dataset,
)
from deception_honesty_axis.common import append_jsonl, ensure_dir, load_jsonl, make_run_id, slugify, write_json, write_jsonl
from deception_honesty_axis.external_sycophancy_benchmarks import (
    ExternalSycophancyActivationDataset,
    ExternalSycophancyExample,
    materialize_external_examples,
    write_external_example_manifest,
)
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status


CORE_EXAMPLE_KEYS = {
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
    "prompt_messages",
    "assistant_output",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract activation rows from previously normalized sycophancy examples. "
            "This reuses inputs/normalized_examples.jsonl snapshots from earlier extraction runs."
        )
    )
    parser.add_argument("--normalized-examples-jsonl", type=Path, required=True, help="Path to inputs/normalized_examples.jsonl.")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-3B-Instruct", help="HF model id to load.")
    parser.add_argument("--batch-size", type=int, default=8, help="Forward-pass batch size.")
    parser.add_argument("--layer", type=int, default=14, help="1-based transformer layer to hook.")
    parser.add_argument(
        "--pooling",
        default="mean_response",
        choices=["mean_response", "last_token"],
        help="How to pool the hooked token states into one activation vector per sample.",
    )
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Model dtype.")
    parser.add_argument("--device", default="auto", help="Use 'auto' for device_map=auto, or an explicit device like cuda:0/cpu.")
    parser.add_argument("--max-length", type=int, default=None, help="Optional tokenizer truncation length.")
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Add tokenizer special tokens when encoding prompts and full conversations.",
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"), help="Repo artifact root.")
    parser.add_argument("--run-id", default=None, help="Optional fixed run id.")
    parser.add_argument("--save-every", type=int, default=100, help="Save progress every N records.")
    parser.add_argument("--push-to-hub-repo-id", default=None, help="Optional HF dataset repo to push output to.")
    parser.add_argument("--private", action="store_true", help="Create/push private HF dataset when pushing.")
    parser.add_argument(
        "--target-name",
        required=True,
        help="Target slug used for artifact path naming, e.g. oeq-framing-human.",
    )
    return parser.parse_args()


def _coerce_prompt_messages(value: Any, *, row_id: str) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Row {row_id!r} has missing or invalid prompt_messages")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise ValueError(f"Row {row_id!r} prompt_messages[{index}] is not an object")
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "")
        if not role:
            raise ValueError(f"Row {row_id!r} prompt_messages[{index}] has empty role")
        messages.append({"role": role, "content": content})
    return tuple(messages)


def load_normalized_examples(path: Path) -> list[ExternalSycophancyExample]:
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"No normalized examples found in {path}")

    examples: list[ExternalSycophancyExample] = []
    for index, row in enumerate(rows):
        row_id = str(row.get("ids") or f"row-{index:06d}")
        missing = [key for key in CORE_EXAMPLE_KEYS if key not in row and key not in {"category"}]
        if missing:
            raise ValueError(f"Row {row_id!r} in {path} is missing required fields: {missing}")
        metadata = {key: value for key, value in row.items() if key not in CORE_EXAMPLE_KEYS}
        examples.append(
            ExternalSycophancyExample(
                ids=row_id,
                pair_id=str(row["pair_id"]),
                dataset_source=str(row["dataset_source"]),
                source_repo_id=str(row["source_repo_id"]),
                source_split=str(row["source_split"]),
                source_record_id=str(row["source_record_id"]),
                adapter_name=str(row["adapter_name"]),
                label_name=str(row["label_name"]),
                label=int(row["label"]),
                category=str(row.get("category") or ""),
                prompt_messages=_coerce_prompt_messages(row["prompt_messages"], row_id=row_id),
                assistant_output=str(row["assistant_output"]),
                metadata=metadata,
            )
        )
    return examples


def _source_manifest_from_examples(path: Path, examples: list[ExternalSycophancyExample]) -> dict[str, Any]:
    dataset_sources = sorted({example.dataset_source for example in examples})
    source_repo_ids = sorted({example.source_repo_id for example in examples})
    source_splits = sorted({example.source_split for example in examples})
    adapter_names = sorted({example.adapter_name for example in examples})
    return {
        "source_type": "normalized_examples_jsonl",
        "normalized_examples_jsonl": str(path),
        "dataset_sources": dataset_sources,
        "source_repo_ids": source_repo_ids,
        "source_splits": source_splits,
        "adapter_names": adapter_names,
        "selected_record_count": len(examples),
    }


def main() -> None:
    args = parse_args()
    normalized_path = args.normalized_examples_jsonl.resolve()
    target_name = args.target_name
    run_id = args.run_id or make_run_id()
    run_root = ensure_dir(
        args.artifact_root
        / "runs"
        / "external-sycophancy-activation-extraction"
        / slugify(target_name)
        / slugify(args.model_name)
        / run_id
    )
    for relative in ("inputs", "results", "logs", "checkpoints", "meta"):
        ensure_dir(run_root / relative)

    examples = load_normalized_examples(normalized_path)
    source_manifest = _source_manifest_from_examples(normalized_path, examples)
    tokenizer, model = load_tokenizer_and_model(args.model_name, dtype_from_name(args.dtype), args.device)
    materialized = materialize_external_examples(tokenizer, examples)
    dataset = ExternalSycophancyActivationDataset(materialized)

    write_analysis_manifest(
        run_root,
        {
            "run_id": run_id,
            "target_name": target_name,
            "normalized_examples_jsonl": str(normalized_path),
            "model_name": args.model_name,
            "layer": args.layer,
            "activation_pooling": args.pooling,
            "batch_size": args.batch_size,
            "push_to_hub_repo_id": args.push_to_hub_repo_id,
            "add_special_tokens": bool(args.add_special_tokens),
        },
    )
    write_stage_status(run_root, "extract_external_sycophancy_activations", "running", {"run_id": run_id})
    write_json(run_root / "inputs" / "source_manifest.json", source_manifest)
    write_external_example_manifest(run_root, examples)

    input_summary = {
        "record_count": len(dataset),
        "dataset_source_counts": dict(Counter(record.dataset_source for record in dataset.records)),
        "label_counts": dict(Counter(int(record.label) for record in dataset.records)),
        "label_name_counts": dict(Counter(record.label_name for record in dataset.records)),
        "category_counts": dict(sorted(Counter(record.category for record in dataset.records if record.category).items())),
    }
    write_json(run_root / "inputs" / "normalized_summary.json", input_summary)

    records_path = run_root / "results" / "records.jsonl"
    existing_records = load_jsonl(records_path)
    completed_ids = {str(record["ids"]) for record in existing_records}
    pending_indices = [
        index
        for index, record in enumerate(dataset.records)
        if record.ids not in completed_ids
    ]
    write_json(
        run_root / "checkpoints" / "progress.json",
        {
            "state": "running",
            "run_id": run_id,
            "total_records": len(dataset),
            "completed_records": len(completed_ids),
            "pending_records": len(pending_indices),
        },
    )

    loader = DataLoader(
        Subset(dataset, pending_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda items: items,
    )

    completed_this_run = 0
    for batch in loader:
        activations = extract_batch_activations(
            tokenizer=tokenizer,
            model=model,
            records=batch,
            layer_number=args.layer,
            pooling=args.pooling,
            add_special_tokens=bool(args.add_special_tokens),
            max_length=args.max_length,
        )
        output_records = []
        for record, activation_payload in zip(batch, activations, strict=True):
            payload = record.output_without_activation()
            payload["activation"] = activation_payload["activation"]
            payload["activation_pooling"] = args.pooling
            payload["activation_layer_number"] = int(args.layer)
            payload["prompt_token_count"] = int(activation_payload["prompt_token_count"])
            payload["response_token_count"] = int(activation_payload["response_token_count"])
            payload["used_last_token_fallback"] = bool(activation_payload["used_last_token_fallback"])
            output_records.append(payload)
        append_jsonl(records_path, output_records)
        completed_this_run += len(output_records)
        completed_total = len(completed_ids) + completed_this_run
        if completed_this_run == len(output_records) or completed_this_run % args.save_every == 0:
            print(f"[normalized-sycophancy-activations] {completed_total}/{len(dataset)} records complete")
            write_json(
                run_root / "checkpoints" / "progress.json",
                {
                    "state": "running",
                    "run_id": run_id,
                    "total_records": len(dataset),
                    "completed_records": completed_total,
                    "pending_records": max(0, len(dataset) - completed_total),
                },
            )

    final_records = normalize_saved_records(
        load_jsonl(records_path),
        layer_number=args.layer,
        pooling=args.pooling,
    )
    write_jsonl(records_path, final_records)
    summary = {
        "record_count": len(final_records),
        "dataset_source_counts": dict(sorted(Counter(record["dataset_source"] for record in final_records).items())),
        "label_counts": dict(sorted(Counter(int(record["label"]) for record in final_records).items())),
        "label_name_counts": dict(sorted(Counter(record["label_name"] for record in final_records).items())),
        "category_counts": dict(sorted(Counter(record.get("category", "") for record in final_records if record.get("category")).items())),
        "pair_count_histogram": dict(sorted(Counter(Counter(record["pair_id"] for record in final_records).values()).items())),
        "activation_pooling": args.pooling,
        "activation_layer_number": int(args.layer),
        "fallback_row_count": int(sum(bool(record["used_last_token_fallback"]) for record in final_records)),
        "records_path": str(records_path),
    }
    write_json(run_root / "results" / "summary.json", summary)
    save_activation_rows_dataset(
        final_records,
        run_root=run_root,
        push_repo_id=args.push_to_hub_repo_id,
        private=args.private,
    )
    write_json(
        run_root / "checkpoints" / "progress.json",
        {
            "state": "completed",
            "run_id": run_id,
            "total_records": len(dataset),
            "completed_records": len(final_records),
            "pending_records": 0,
        },
    )
    write_stage_status(
        run_root,
        "extract_external_sycophancy_activations",
        "completed",
        {
            "run_id": run_id,
            "target_name": target_name,
            "record_count": len(final_records),
            "activation_pooling": args.pooling,
            "activation_layer_number": int(args.layer),
            "pushed_to_hub": bool(args.push_to_hub_repo_id),
        },
    )
    print(f"[normalized-sycophancy-activations] completed run_id={run_id} root={run_root}")


if __name__ == "__main__":
    main()
