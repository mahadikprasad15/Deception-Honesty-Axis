#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import time

from deception_honesty_axis.common import append_jsonl, utc_now_iso, write_json
from deception_honesty_axis.config import corpus_root as resolve_corpus_root
from deception_honesty_axis.config import load_config
from deception_honesty_axis.indexes import (
    append_index_records,
    ensure_corpus_layout,
    load_index,
    next_shard_path,
    update_coverage,
)
from deception_honesty_axis.metadata import write_corpus_manifest, write_stage_status
from deception_honesty_axis.work_units import build_work_units


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate role-conditioned rollouts into the corpus store.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/deception_v1_llama32_3b.json"),
        help="Path to the experiment config file.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Number of prompts per generation batch.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Print progress every N completed items.",
    )
    return parser.parse_args()


def flush_role_buffer(corpus_path: Path, role: str, rollout_buffer: list[dict], index_buffer: list[dict]) -> None:
    if not rollout_buffer:
        return
    shard_path = next_shard_path(corpus_path, "rollouts", role, ".jsonl")
    append_jsonl(shard_path, rollout_buffer)
    relative = str(shard_path.relative_to(corpus_path))
    for record in index_buffer:
        record["shard_path"] = relative
    append_index_records(corpus_path, "rollouts", index_buffer)
    rollout_buffer.clear()
    index_buffer.clear()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    corpus_path = resolve_corpus_root(config)
    ensure_corpus_layout(corpus_path)

    work_units = build_work_units(config.repo_root, config)
    rollout_index = load_index(corpus_path, "rollouts")
    missing_units = [unit for unit in work_units if unit["item_id"] not in rollout_index]
    save_every = int(config.saving["save_every"])

    write_corpus_manifest(
        corpus_path,
        {
            "config_path": str(config.path),
            "model_name": config.model_id,
            "dataset_name": config.dataset_name,
            "created_at": utc_now_iso(),
        },
    )
    write_stage_status(
        corpus_path,
        "generate_rollouts",
        "running",
        {"target_items": len(work_units), "pending_items": len(missing_units)},
    )

    if not missing_units:
        update_coverage(
            corpus_path,
            {
                "target_items": len(work_units),
                "rollouts_completed": len(rollout_index),
                "activations_completed": len(load_index(corpus_path, "activations")),
                "updated_at": utc_now_iso(),
            },
        )
        write_stage_status(corpus_path, "generate_rollouts", "completed", {"generated_items": len(rollout_index)})
        print("No missing rollout work units.")
        return

    from deception_honesty_axis.modeling import build_messages, generate_responses, load_model_and_tokenizer

    model_settings = dict(config.raw["model"])
    model_settings["name"] = config.model_id
    tokenizer, model = load_model_and_tokenizer(model_settings)
    rollout_buffers: dict[str, list[dict]] = defaultdict(list)
    index_buffers: dict[str, list[dict]] = defaultdict(list)
    started_at = time.monotonic()
    processed_items = 0

    for batch_start in range(0, len(missing_units), args.batch_size):
        batch = missing_units[batch_start: batch_start + args.batch_size]
        messages = [build_messages(unit["system_prompt"], unit["question_text"]) for unit in batch]
        responses = generate_responses(tokenizer, model, messages, {**model_settings, **config.generation})
        for unit, response in zip(batch, responses):
            record = {
                **unit,
                "model_name": config.model_id,
                "response_text": response,
                "generated_at": utc_now_iso(),
            }
            rollout_buffers[unit["role"]].append(record)
            index_buffers[unit["role"]].append(
                {
                    "item_id": unit["item_id"],
                    "role": unit["role"],
                    "role_source": unit.get("role_source"),
                    "question_id": unit["question_id"],
                    "prompt_id": unit["prompt_id"],
                }
            )
            if len(rollout_buffers[unit["role"]]) >= save_every:
                flush_role_buffer(corpus_path, unit["role"], rollout_buffers[unit["role"]], index_buffers[unit["role"]])
            processed_items += 1

            if processed_items % args.progress_every == 0:
                elapsed = max(time.monotonic() - started_at, 1e-6)
                rate = processed_items / elapsed
                remaining = len(missing_units) - processed_items
                eta_seconds = remaining / rate if rate > 0 else float("inf")
                print(
                    f"[generate] {processed_items}/{len(missing_units)} missing items done "
                    f"({rate:.2f} items/s, eta {eta_seconds/60:.1f} min)"
                )

        write_json(
            corpus_path / "checkpoints" / "generate_rollouts_progress.json",
            {
                "target_items": len(work_units),
                "completed_batches": batch_start // args.batch_size + 1,
                "completed_items": len(load_index(corpus_path, "rollouts")) + sum(len(values) for values in rollout_buffers.values()),
                "processed_this_run": processed_items,
                "updated_at": utc_now_iso(),
            },
        )

    for role in list(rollout_buffers):
        flush_role_buffer(corpus_path, role, rollout_buffers[role], index_buffers[role])

    final_rollout_index = load_index(corpus_path, "rollouts")
    update_coverage(
        corpus_path,
        {
            "target_items": len(work_units),
            "rollouts_completed": len(final_rollout_index),
            "activations_completed": len(load_index(corpus_path, "activations")),
            "updated_at": utc_now_iso(),
        },
    )
    write_stage_status(corpus_path, "generate_rollouts", "completed", {"generated_items": len(final_rollout_index)})
    total_elapsed = max(time.monotonic() - started_at, 1e-6)
    print(f"[generate] completed {processed_items} items this run in {total_elapsed/60:.1f} min")
    print(f"Generated {len(final_rollout_index)} rollout records.")


if __name__ == "__main__":
    main()
