#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Extract pooled all-layer activations for rollout records.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/deception_v1_llama32_3b.json"),
        help="Path to the experiment config file.",
    )
    return parser.parse_args()


def _capture_layer_outputs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    import torch

    captured: list[torch.Tensor] = []
    hooks = []
    for layer in model.model.layers:
        hooks.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output: captured.append((output[0] if isinstance(output, tuple) else output).detach())
            )
        )
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    return torch.stack(captured, dim=0)


def flush_activation_buffer(corpus_path: Path, role: str, records: list[dict], index_records: list[dict]) -> None:
    import torch

    if not records:
        return
    shard_path = next_shard_path(corpus_path, "activations", role, ".pt")
    ensure_dir(shard_path.parent)
    torch.save({"records": records}, shard_path)
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
    rollout_rows = [row for item_id, row in rollout_index.items() if item_id in target_ids and item_id not in activation_index]

    write_stage_status(
        corpus_path,
        "extract_activations",
        "running",
        {"pending_items": len(rollout_rows), "target_items": len(work_units)},
    )

    if not rollout_rows:
        write_stage_status(corpus_path, "extract_activations", "completed", {"activated_items": len(activation_index)})
        print("No missing activation work units.")
        return

    import torch
    from deception_honesty_axis.modeling import build_messages, load_model_and_tokenizer

    model_settings = dict(config.raw["model"])
    model_settings["name"] = config.model_id
    tokenizer, model = load_model_and_tokenizer(model_settings)
    model_device = next(model.parameters()).device
    rollout_records = load_rollout_records(corpus_path, rollout_rows)
    save_every = int(config.saving["save_every"])
    activation_buffers: dict[str, list[dict]] = defaultdict(list)
    index_buffers: dict[str, list[dict]] = defaultdict(list)

    for offset, index_row in enumerate(rollout_rows, start=1):
        rollout_record = rollout_records[index_row["item_id"]]
        prefix_ids = tokenizer.apply_chat_template(
            build_messages(rollout_record["system_prompt"], rollout_record["question_text"]),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        full_ids = tokenizer.apply_chat_template(
            build_messages(
                rollout_record["system_prompt"],
                rollout_record["question_text"],
                rollout_record["response_text"],
            ),
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        attention_mask = torch.ones_like(full_ids)
        full_ids = full_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)
        response_start = int(prefix_ids.shape[-1])
        captured = _capture_layer_outputs(model, full_ids, attention_mask)
        response_slice = captured[:, 0, response_start:, :]
        if response_slice.shape[1] == 0:
            response_slice = captured[:, 0, -1:, :]
        pooled = response_slice.mean(dim=1).to(dtype=torch.float16, device="cpu")

        role = rollout_record["role"]
        activation_buffers[role].append(
            {
                "item_id": rollout_record["item_id"],
                "role": role,
                "question_id": rollout_record["question_id"],
                "prompt_id": rollout_record["prompt_id"],
                "pooled_activations": pooled,
                "response_token_count": int(response_slice.shape[1]),
            }
        )
        index_buffers[role].append(
            {
                "item_id": rollout_record["item_id"],
                "role": role,
                "question_id": rollout_record["question_id"],
                "prompt_id": rollout_record["prompt_id"],
            }
        )
        if len(activation_buffers[role]) >= save_every:
            flush_activation_buffer(corpus_path, role, activation_buffers[role], index_buffers[role])

        write_json(
            corpus_path / "checkpoints" / "extract_activations_progress.json",
            {
                "target_items": len(work_units),
                "completed_items": len(load_index(corpus_path, "activations")) + sum(len(values) for values in activation_buffers.values()),
                "processed_this_run": offset,
                "updated_at": utc_now_iso(),
            },
        )

    for role in list(activation_buffers):
        flush_activation_buffer(corpus_path, role, activation_buffers[role], index_buffers[role])

    final_activations = load_index(corpus_path, "activations")
    update_coverage(
        corpus_path,
        {
            "target_items": len(work_units),
            "rollouts_completed": len(rollout_index),
            "activations_completed": len(final_activations),
            "updated_at": utc_now_iso(),
        },
    )
    write_stage_status(corpus_path, "extract_activations", "completed", {"activated_items": len(final_activations)})
    print(f"Extracted activations for {len(final_activations)} items.")


if __name__ == "__main__":
    main()
