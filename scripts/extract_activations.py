#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import time

from deception_honesty_axis.common import ensure_dir, utc_now_iso, write_json
from deception_honesty_axis.config import corpus_root as resolve_corpus_root
from deception_honesty_axis.config import load_config
from deception_honesty_axis.indexes import (
    append_index_records,
    ensure_corpus_layout,
    load_index,
    next_shard_path,
    update_coverage,
)
from deception_honesty_axis.metadata import write_stage_status
from deception_honesty_axis.records import load_rollout_records
from deception_honesty_axis.work_units import build_work_units


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pooled response activations for rollout records.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/deception_v1_llama32_3b.json"),
        help="Path to the experiment config file.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Print progress every N completed activation items.",
    )
    return parser.parse_args()


def _resolve_activation_layer_numbers(model, requested_layers: list[int] | None) -> list[int]:
    layer_count = len(model.model.layers)
    if requested_layers is None:
        return list(range(1, layer_count + 1))

    seen: set[int] = set()
    resolved: list[int] = []
    for layer_number in requested_layers:
        if layer_number in seen:
            continue
        if layer_number < 1 or layer_number > layer_count:
            raise ValueError(f"Requested activation layer {layer_number} is out of range for {layer_count} model layers")
        resolved.append(layer_number)
        seen.add(layer_number)
    if not resolved:
        raise ValueError("At least one activation layer must be selected")
    return resolved


def _capture_layer_outputs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_numbers: list[int],
) -> torch.Tensor:
    import torch

    captured_by_layer: dict[int, torch.Tensor] = {}
    hooks = []
    def make_hook(layer_number: int):  # noqa: ANN202
        def hook(_module, _inputs, output) -> None:  # noqa: ANN001
            captured_by_layer[layer_number] = (output[0] if isinstance(output, tuple) else output).detach()

        return hook

    for layer_number in layer_numbers:
        layer = model.model.layers[layer_number - 1]
        hooks.append(layer.register_forward_hook(make_hook(layer_number)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    return torch.stack([captured_by_layer[layer_number] for layer_number in layer_numbers], dim=0)


def _activation_row_matches_request(row: dict, requested_layer_numbers: list[int] | None) -> bool:
    if requested_layer_numbers is None:
        return True
    raw_layers = row.get("activation_layer_numbers")
    if raw_layers is None:
        return False
    return [int(layer_number) for layer_number in raw_layers] == requested_layer_numbers


def flush_activation_buffer(
    corpus_path: Path,
    role: str,
    records: list[dict],
    index_records: list[dict],
    activation_layer_numbers: list[int],
) -> None:
    import torch

    if not records:
        return
    shard_path = next_shard_path(corpus_path, "activations", role, ".pt")
    ensure_dir(shard_path.parent)
    torch.save({"records": records, "activation_layer_numbers": activation_layer_numbers}, shard_path)
    relative = str(shard_path.relative_to(corpus_path))
    for record in index_records:
        record["shard_path"] = relative
    append_index_records(corpus_path, "activations", index_records)
    records.clear()
    index_records.clear()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    corpus_path = resolve_corpus_root(config)
    ensure_corpus_layout(corpus_path)

    work_units = build_work_units(config.repo_root, config)
    target_ids = {unit["item_id"] for unit in work_units}
    rollout_index = load_index(corpus_path, "rollouts")
    activation_index = load_index(corpus_path, "activations")
    requested_layer_numbers = config.activation_layer_numbers
    completed_activation_ids = {
        item_id
        for item_id, row in activation_index.items()
        if item_id in target_ids and _activation_row_matches_request(row, requested_layer_numbers)
    }
    rollout_rows = [row for item_id, row in rollout_index.items() if item_id in target_ids and item_id not in completed_activation_ids]

    write_stage_status(
        corpus_path,
        "extract_activations",
        "running",
        {"pending_items": len(rollout_rows), "target_items": len(work_units)},
    )

    if not rollout_rows:
        write_stage_status(
            corpus_path,
            "extract_activations",
            "completed",
            {"activated_items": len(completed_activation_ids), "requested_activation_layer_numbers": requested_layer_numbers},
        )
        print("No missing activation work units.")
        return

    import torch
    from deception_honesty_axis.modeling import build_messages, load_model_and_tokenizer

    model_settings = dict(config.raw["model"])
    model_settings["name"] = config.model_id
    tokenizer, model = load_model_and_tokenizer(model_settings)
    model_device = next(model.parameters()).device
    activation_layer_numbers = _resolve_activation_layer_numbers(model, config.activation_layer_numbers)
    print(f"[activations] capturing layers {activation_layer_numbers}")
    rollout_records = load_rollout_records(corpus_path, rollout_rows)
    save_every = int(config.saving["save_every"])
    activation_buffers: dict[str, list[dict]] = defaultdict(list)
    index_buffers: dict[str, list[dict]] = defaultdict(list)
    started_at = time.monotonic()

    for offset, index_row in enumerate(rollout_rows, start=1):
        rollout_record = rollout_records[index_row["item_id"]]
        prefix_encoding = tokenizer.apply_chat_template(
            build_messages(rollout_record["system_prompt"], rollout_record["question_text"]),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        full_encoding = tokenizer.apply_chat_template(
            build_messages(
                rollout_record["system_prompt"],
                rollout_record["question_text"],
                rollout_record["response_text"],
            ),
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        if hasattr(prefix_encoding, "input_ids"):
            prefix_ids = prefix_encoding["input_ids"]
        else:
            prefix_ids = prefix_encoding

        if hasattr(full_encoding, "input_ids"):
            full_ids = full_encoding["input_ids"]
            attention_mask = full_encoding.get("attention_mask")
        else:
            full_ids = full_encoding
            attention_mask = None

        if attention_mask is None:
            attention_mask = torch.ones_like(full_ids)

        full_ids = full_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)
        response_start = int(prefix_ids.shape[-1])
        captured = _capture_layer_outputs(model, full_ids, attention_mask, activation_layer_numbers)
        response_slice = captured[:, 0, response_start:, :]
        if response_slice.shape[1] == 0:
            response_slice = captured[:, 0, -1:, :]
        pooled = response_slice.mean(dim=1).to(dtype=torch.float16, device="cpu")

        role = rollout_record["role"]
        activation_buffers[role].append(
            {
                "item_id": rollout_record["item_id"],
                "role": role,
                "role_source": rollout_record.get("role_source"),
                "role_metadata": rollout_record.get("role_metadata", {}),
                "question_id": rollout_record["question_id"],
                "question_metadata": rollout_record.get("question_metadata", {}),
                "prompt_id": rollout_record["prompt_id"],
                "pooled_activations": pooled,
                "activation_layer_numbers": activation_layer_numbers,
                "response_token_count": int(response_slice.shape[1]),
            }
        )
        index_buffers[role].append(
            {
                "item_id": rollout_record["item_id"],
                "role": role,
                "role_source": rollout_record.get("role_source"),
                "question_id": rollout_record["question_id"],
                "prompt_id": rollout_record["prompt_id"],
                "activation_layer_numbers": activation_layer_numbers,
            }
        )
        if len(activation_buffers[role]) >= save_every:
            flush_activation_buffer(corpus_path, role, activation_buffers[role], index_buffers[role], activation_layer_numbers)

        write_json(
            corpus_path / "checkpoints" / "extract_activations_progress.json",
            {
                "target_items": len(work_units),
                "completed_items": len(load_index(corpus_path, "activations")) + sum(len(values) for values in activation_buffers.values()),
                "processed_this_run": offset,
                "updated_at": utc_now_iso(),
            },
        )
        if offset % args.progress_every == 0:
            elapsed = max(time.monotonic() - started_at, 1e-6)
            rate = offset / elapsed
            remaining = len(rollout_rows) - offset
            eta_seconds = remaining / rate if rate > 0 else float("inf")
            print(
                f"[activations] {offset}/{len(rollout_rows)} missing items done "
                f"({rate:.2f} items/s, eta {eta_seconds/60:.1f} min)"
            )

    for role in list(activation_buffers):
        flush_activation_buffer(corpus_path, role, activation_buffers[role], index_buffers[role], activation_layer_numbers)

    final_activations = load_index(corpus_path, "activations")
    final_matching_activation_ids = {
        item_id
        for item_id, row in final_activations.items()
        if item_id in target_ids and _activation_row_matches_request(row, activation_layer_numbers)
    }
    update_coverage(
        corpus_path,
        {
            "target_items": len(work_units),
            "rollouts_completed": len(rollout_index),
            "activations_completed": len(final_matching_activation_ids),
            "activation_layer_numbers": activation_layer_numbers,
            "updated_at": utc_now_iso(),
        },
    )
    write_stage_status(
        corpus_path,
        "extract_activations",
        "completed",
        {"activated_items": len(final_matching_activation_ids), "activation_layer_numbers": activation_layer_numbers},
    )
    total_elapsed = max(time.monotonic() - started_at, 1e-6)
    print(f"[activations] completed {len(rollout_rows)} items this run in {total_elapsed/60:.1f} min")
    print(f"Extracted activations for {len(final_activations)} items.")


if __name__ == "__main__":
    main()
