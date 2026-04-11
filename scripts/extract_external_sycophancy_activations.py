#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

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
    load_external_sycophancy_examples,
    materialize_external_examples,
    write_external_example_manifest,
)
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract activation rows from external labeled sycophancy benchmarks hosted on Hugging Face."
    )
    parser.add_argument("--source-repo-id", required=True, help="HF dataset repo containing the source benchmark.")
    parser.add_argument("--source-split", required=True, help="HF dataset split to load.")
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional adapter name. If omitted, infer it from the dataset repo id.",
    )
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
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on normalized examples after adapter expansion.")
    parser.add_argument("--save-every", type=int, default=100, help="Save progress every N records.")
    parser.add_argument("--push-to-hub-repo-id", default=None, help="Optional HF dataset repo to push output to.")
    parser.add_argument("--private", action="store_true", help="Create/push private HF dataset when pushing.")
    parser.add_argument(
        "--target-name",
        default=None,
        help="Optional target slug used only for artifact path naming.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_name = args.target_name or f"{args.source_repo_id.split('/')[-1]}-{args.source_split}"
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

    examples, source_manifest = load_external_sycophancy_examples(
        repo_id=args.source_repo_id,
        split=args.source_split,
        adapter_name=args.adapter,
        max_samples=args.max_samples,
    )
    tokenizer, model = load_tokenizer_and_model(args.model_name, dtype_from_name(args.dtype), args.device)
    materialized = materialize_external_examples(tokenizer, examples)
    dataset = ExternalSycophancyActivationDataset(materialized)

    write_analysis_manifest(
        run_root,
        {
            "run_id": run_id,
            "target_name": target_name,
            "source_repo_id": args.source_repo_id,
            "source_split": args.source_split,
            "adapter_name": source_manifest["adapter_name"],
            "dataset_source": source_manifest["dataset_source"],
            "model_name": args.model_name,
            "layer": args.layer,
            "activation_pooling": args.pooling,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
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
            print(f"[external-sycophancy-activations] {completed_total}/{len(dataset)} records complete")
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
    print(f"[external-sycophancy-activations] completed run_id={run_id} root={run_root}")


if __name__ == "__main__":
    main()
