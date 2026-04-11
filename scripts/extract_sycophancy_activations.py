#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

from deception_honesty_axis.common import append_jsonl, ensure_dir, load_jsonl, make_run_id, sha256_file, slugify, write_json, write_jsonl
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.sycophancy_activations import (
    DEFAULT_SOURCE_FILES,
    DEFAULT_SOURCE_REPO_ID,
    SycophancyActivationDataset,
    normalize_activation_pooling,
    pool_hidden_states,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract layer activations over existing sycophancy JSONL outputs and optionally push a HF Dataset."
    )
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct", help="HF model id to load.")
    parser.add_argument("--batch_size", type=int, default=8, help="Forward-pass batch size.")
    parser.add_argument("--layer", type=int, default=14, help="1-based transformer layer to hook.")
    parser.add_argument(
        "--pooling",
        default="mean_response",
        choices=["mean_response", "last_token"],
        help="How to pool the hooked token states into one activation vector per sample.",
    )
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Model dtype.")
    parser.add_argument("--device", default="auto", help="Use 'auto' for device_map=auto, or an explicit device like cuda:0/cpu.")
    parser.add_argument("--max_length", type=int, default=None, help="Optional tokenizer truncation length.")
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Add tokenizer special tokens. Default is false because input_formatted is already templated.",
    )
    parser.add_argument("--source_repo_id", default=DEFAULT_SOURCE_REPO_ID, help="HF dataset repo with source JSONLs.")
    parser.add_argument("--factual-jsonl", type=Path, default=None, help="Optional local factual/multichoice JSONL.")
    parser.add_argument("--evaluative-jsonl", type=Path, default=None, help="Optional local arguments JSONL.")
    parser.add_argument("--aesthetic-jsonl", type=Path, default=None, help="Optional local haikus JSONL.")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"), help="Repo artifact root.")
    parser.add_argument("--run-id", default=None, help="Optional fixed run id.")
    parser.add_argument("--save-every", type=int, default=100, help="Save progress every N records.")
    parser.add_argument("--push_to_hub_repo_id", default=None, help="Optional HF dataset repo to push output to.")
    parser.add_argument("--private", action="store_true", help="Create/push private HF dataset when pushing.")
    parser.add_argument(
        "--fail-on-unknown-metadata",
        action="store_true",
        help="Fail before model loading if any record has unknown user framing or system nudge metadata.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def resolve_transformer_layers(model) -> Any:  # noqa: ANN001
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not locate transformer layers at model.model.layers or model.transformer.h")


def load_tokenizer_and_model(model_name: str, dtype: torch.dtype, device: str):  # noqa: ANN201
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        model.to(torch.device(device))
    model.eval()
    return tokenizer, model


def extract_batch_activations(
    *,
    tokenizer,
    model,
    records: list[Any],
    layer_number: int,
    pooling: str,
    add_special_tokens: bool,
    max_length: int | None,
) -> list[dict[str, Any]]:
    layers = resolve_transformer_layers(model)
    if layer_number < 1 or layer_number > len(layers):
        raise ValueError(f"Layer {layer_number} is out of range for model with {len(layers)} layers")

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output) -> None:  # noqa: ANN001
        captured["hidden"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = layers[layer_number - 1].register_forward_hook(hook)
    try:
        encoded = tokenizer(
            [record.activation_text for record in records],
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        prompt_encoded = tokenizer(
            [record.prompt_text for record in records],
            return_tensors="pt",
            padding=True,
            add_special_tokens=add_special_tokens,
        )
        if "attention_mask" not in encoded:
            encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
        if "attention_mask" not in prompt_encoded:
            prompt_encoded["attention_mask"] = torch.ones_like(prompt_encoded["input_ids"])
        prompt_token_counts = [int(value) for value in prompt_encoded["attention_mask"].sum(dim=1).tolist()]
        model_device = next(model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        with torch.no_grad():
            model(**encoded, use_cache=False)
    finally:
        handle.remove()

    pooled, pooling_metadata = pool_hidden_states(
        captured["hidden"],
        encoded["attention_mask"],
        prompt_token_counts,
        pooling,
    )
    activations = pooled.to(dtype=torch.float32, device="cpu")
    return [
        {
            "activation": [float(value) for value in activation.tolist()],
            "prompt_token_count": int(metadata.prompt_token_count),
            "response_token_count": int(metadata.response_token_count),
            "used_last_token_fallback": bool(metadata.used_last_token_fallback),
        }
        for activation, metadata in zip(activations, pooling_metadata, strict=True)
    ]


def local_or_hub_dataset(args: argparse.Namespace, run_root: Path) -> tuple[SycophancyActivationDataset, dict[str, str]]:
    local_paths = {
        "factual": args.factual_jsonl,
        "evaluative": args.evaluative_jsonl,
        "aesthetic": args.aesthetic_jsonl,
    }
    if all(path is not None for path in local_paths.values()):
        paths = {name: path.resolve() for name, path in local_paths.items() if path is not None}
        dataset = SycophancyActivationDataset.from_jsonl_paths(paths)
        return dataset, {
            name: str(path)
            for name, path in paths.items()
        }

    cache_dir = ensure_dir(run_root / "inputs" / "hf_cache")
    dataset = SycophancyActivationDataset.from_hub(
        repo_id=args.source_repo_id,
        source_files=DEFAULT_SOURCE_FILES,
        cache_dir=cache_dir,
        token=os.environ.get("HF_TOKEN"),
    )
    return dataset, {
        "source_repo_id": args.source_repo_id,
        **DEFAULT_SOURCE_FILES,
    }


def save_hf_dataset(records: list[dict[str, Any]], run_root: Path, push_repo_id: str | None, private: bool) -> None:
    from datasets import Dataset, Features, Sequence, Value

    features = Features(
        {
            "ids": Value("string"),
            "pair_id": Value("string"),
            "dataset_source": Value("string"),
            "user_framing": Value("string"),
            "system_nudge_direction": Value("string"),
            "activation": Sequence(Value("float32")),
            "activation_pooling": Value("string"),
            "activation_layer_number": Value("int64"),
            "prompt_token_count": Value("int64"),
            "response_token_count": Value("int64"),
            "used_last_token_fallback": Value("bool"),
            "label": Value("int64"),
        }
    )
    dataset = Dataset.from_list(records, features=features)
    dataset_dir = run_root / "results" / "hf_dataset"
    dataset.save_to_disk(str(dataset_dir))
    if push_repo_id:
        dataset.push_to_hub(push_repo_id, private=private, token=os.environ.get("HF_TOKEN"))


def _normalize_record_field_int(value: Any, default: int) -> int:
    if value in (None, ""):
        return int(default)
    return int(value)


def normalize_saved_records(rows: list[dict[str, Any]], *, layer_number: int, pooling: str) -> list[dict[str, Any]]:
    normalized_pooling = normalize_activation_pooling(pooling) or "mean_response"
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        sample_id = str(row.get("ids", ""))
        if sample_id in seen_ids:
            continue
        seen_ids.add(sample_id)
        payload = dict(row)
        payload["activation_pooling"] = str(
            normalize_activation_pooling(payload.get("activation_pooling")) or normalized_pooling
        )
        payload["activation_layer_number"] = _normalize_record_field_int(
            payload.get("activation_layer_number"),
            layer_number,
        )
        payload["prompt_token_count"] = _normalize_record_field_int(payload.get("prompt_token_count"), 0)
        payload["response_token_count"] = _normalize_record_field_int(payload.get("response_token_count"), 0)
        payload["used_last_token_fallback"] = bool(payload.get("used_last_token_fallback", False))
        normalized.append(payload)
    return normalized


def main() -> None:
    args = parse_args()
    run_id = args.run_id or make_run_id()
    run_root = ensure_dir(args.artifact_root / "runs" / "sycophancy-activation-extraction" / slugify(args.model_name) / run_id)
    for relative in ("inputs", "results", "logs", "checkpoints", "meta"):
        ensure_dir(run_root / relative)

    write_analysis_manifest(
        run_root,
        {
            "run_id": run_id,
            "model_name": args.model_name,
            "layer": args.layer,
            "activation_pooling": args.pooling,
            "batch_size": args.batch_size,
            "source_repo_id": args.source_repo_id,
            "push_to_hub_repo_id": args.push_to_hub_repo_id,
            "add_special_tokens": bool(args.add_special_tokens),
        },
    )
    write_stage_status(run_root, "extract_sycophancy_activations", "running", {"run_id": run_id})

    dataset, source_manifest = local_or_hub_dataset(args, run_root)
    metadata_counts = {
        "system_nudge_direction": dict(Counter(record.system_nudge_direction for record in dataset.records)),
        "user_framing": dict(Counter(record.user_framing for record in dataset.records)),
        "dataset_source": dict(Counter(record.dataset_source for record in dataset.records)),
    }
    unknown_ids = [
        record.ids
        for record in dataset.records
        if record.system_nudge_direction == "unknown" or record.user_framing == "unknown"
    ]
    write_json(
        run_root / "inputs" / "metadata_summary.json",
        {
            **metadata_counts,
            "unknown_metadata_count": len(unknown_ids),
            "unknown_metadata_ids_preview": unknown_ids[:50],
        },
    )
    print(f"[sycophancy-activations] metadata counts: {metadata_counts}; unknown={len(unknown_ids)}")
    if args.fail_on_unknown_metadata and unknown_ids:
        raise RuntimeError(f"Found {len(unknown_ids)} records with unknown metadata; preview={unknown_ids[:10]}")

    source_manifest["record_count"] = len(dataset)
    for key in ("factual", "evaluative", "aesthetic"):
        path = source_manifest.get(key)
        if path and Path(path).exists():
            source_manifest[f"{key}_sha256"] = sha256_file(Path(path))
    write_json(run_root / "inputs" / "source_manifest.json", source_manifest)

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

    tokenizer, model = load_tokenizer_and_model(args.model_name, dtype_from_name(args.dtype), args.device)
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
            payload["activation_pooling"] = normalize_activation_pooling(args.pooling) or args.pooling
            payload["activation_layer_number"] = int(args.layer)
            payload["prompt_token_count"] = int(activation_payload["prompt_token_count"])
            payload["response_token_count"] = int(activation_payload["response_token_count"])
            payload["used_last_token_fallback"] = bool(activation_payload["used_last_token_fallback"])
            output_records.append(payload)
        append_jsonl(records_path, output_records)
        completed_this_run += len(output_records)
        completed_total = len(completed_ids) + completed_this_run
        if completed_this_run == len(output_records) or completed_this_run % args.save_every == 0:
            print(f"[sycophancy-activations] {completed_total}/{len(dataset)} records complete")
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
    pair_counts = Counter(record["pair_id"] for record in final_records)
    label_counts = Counter(int(record["label"]) for record in final_records)
    source_counts = Counter(record["dataset_source"] for record in final_records)
    write_json(
        run_root / "results" / "summary.json",
        {
            "record_count": len(final_records),
            "label_counts": dict(sorted(label_counts.items())),
            "dataset_source_counts": dict(sorted(source_counts.items())),
            "pair_count_histogram": dict(sorted(Counter(pair_counts.values()).items())),
            "activation_pooling": normalize_activation_pooling(args.pooling),
            "activation_layer_number": int(args.layer),
            "fallback_row_count": int(sum(bool(record["used_last_token_fallback"]) for record in final_records)),
            "records_path": str(records_path),
        },
    )
    save_hf_dataset(final_records, run_root, args.push_to_hub_repo_id, args.private)
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
        "extract_sycophancy_activations",
        "completed",
        {
            "run_id": run_id,
            "record_count": len(final_records),
            "activation_pooling": normalize_activation_pooling(args.pooling),
            "activation_layer_number": int(args.layer),
            "pushed_to_hub": bool(args.push_to_hub_repo_id),
        },
    )
    print(f"[sycophancy-activations] completed run_id={run_id} root={run_root}")


if __name__ == "__main__":
    main()
