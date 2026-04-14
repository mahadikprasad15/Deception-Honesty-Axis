#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from deception_honesty_axis.common import append_jsonl, load_jsonl, make_run_id, utc_now_iso, write_json, write_jsonl
from deception_honesty_axis.imt_config import imt_run_root, load_imt_config
from deception_honesty_axis.imt_recovery import (
    TEMPLATE_LABELS,
    aggregate_template_scores,
    build_reusable_template_rows,
    build_judge_messages,
    load_bank_records,
    load_completed_template_keys,
    parse_imt_score_output,
)
from deception_honesty_axis.metadata import write_stage_status
from deception_honesty_axis.modeling import generate_responses, load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score IMT source-bank responses with the Llama judge model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the IMT recovery config.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of judge prompts per generation batch.")
    parser.add_argument("--progress-every", type=int, default=20, help="Print progress every N completed prompts.")
    parser.add_argument(
        "--reuse-score-config",
        type=Path,
        default=None,
        help="Optional prior IMT config whose parsed per-template scores should be reused.",
    )
    parser.add_argument(
        "--reuse-score-run-id",
        type=str,
        default=None,
        help="Optional prior IMT run id whose parsed per-template scores should be reused.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_imt_config(args.config.resolve())
    run_id = args.run_id or make_run_id()
    run_root = imt_run_root(config, run_id)
    bank_records_path = run_root / "results" / "bank_records.pt"
    if not bank_records_path.exists():
        raise FileNotFoundError(f"Missing bank records: {bank_records_path}")

    results_dir = run_root / "results" / "scoring"
    per_template_path = results_dir / "per_template_scores.jsonl"
    averaged_scores_path = results_dir / "averaged_scores.jsonl"
    summary_path = results_dir / "scoring_summary.json"
    progress_path = run_root / "checkpoints" / "imt_scores_progress.json"

    bank_records = load_bank_records(bank_records_path)
    completed_keys = load_completed_template_keys(per_template_path)
    reuse_spec = config.score_reuse
    reuse_config_path = None
    reuse_run_id = None
    reuse_bank_axes: list[str] = []
    if args.reuse_score_config and args.reuse_score_run_id:
        reuse_config_path = args.reuse_score_config.resolve()
        reuse_run_id = str(args.reuse_score_run_id)
    elif reuse_spec is not None:
        reuse_config_path = reuse_spec.imt_config_path
        reuse_run_id = reuse_spec.run_id
        reuse_bank_axes = list(reuse_spec.bank_axes)

    reused_prompt_scores = 0
    if reuse_config_path is not None and reuse_run_id is not None:
        from deception_honesty_axis.imt_config import imt_run_root as resolve_reuse_run_root, load_imt_config as load_reuse_config

        prior_config = load_reuse_config(reuse_config_path)
        prior_run_root = resolve_reuse_run_root(prior_config, reuse_run_id)
        reusable_rows = build_reusable_template_rows(
            bank_records,
            prior_bank_records_path=prior_run_root / "results" / "bank_records.pt",
            prior_per_template_scores_path=prior_run_root / "results" / "scoring" / "per_template_scores.jsonl",
            template_names=config.score_templates,
            bank_axes=reuse_bank_axes,
            existing_completed_keys=completed_keys,
            reused_from_run_id=reuse_run_id,
            reused_from_config_path=reuse_config_path,
        )
        if reusable_rows:
            append_jsonl(per_template_path, reusable_rows)
            reused_prompt_scores = len(reusable_rows)
            completed_keys = load_completed_template_keys(per_template_path)
            print(
                f"[imt-scores] reused {reused_prompt_scores} parsed prompt scores "
                f"from {reuse_run_id}"
            )

    tasks = [
        (record, template_name)
        for record in bank_records
        for template_name in config.score_templates
        if (str(record["item_id"]), template_name) not in completed_keys
    ]

    write_stage_status(
        run_root,
        "score_imt_bank",
        "running",
        {
            "bank_items": len(bank_records),
            "templates": config.score_templates,
            "completed_prompt_scores": len(completed_keys),
            "pending_prompt_scores": len(tasks),
            "reused_prompt_scores": reused_prompt_scores,
        },
    )
    if not tasks:
        aggregated_rows = aggregate_template_scores(load_jsonl(per_template_path), template_names=config.score_templates)
        write_jsonl(averaged_scores_path, aggregated_rows)
        write_json(
            summary_path,
            {
                "bank_items": len(bank_records),
                "templates": config.score_templates,
                "completed_prompt_scores": len(completed_keys),
                "averaged_item_scores": len(aggregated_rows),
                "reused_prompt_scores": sum(
                    1 for row in load_jsonl(per_template_path) if row.get("reused_from_run_id")
                ),
            },
        )
        write_stage_status(
            run_root,
            "score_imt_bank",
            "completed",
            {
                "bank_items": len(bank_records),
                "completed_prompt_scores": len(completed_keys),
                "averaged_item_scores": len(aggregated_rows),
                "reused_prompt_scores": sum(
                    1 for row in load_jsonl(per_template_path) if row.get("reused_from_run_id")
                ),
            },
        )
        print("[imt-scores] no missing prompt scores")
        return

    model_config = dict(config.score_model)
    generation = dict(model_config.get("generation", {}))
    if not generation:
        generation = {
            "max_new_tokens": 256,
            "temperature": 1.0,
            "top_p": 1.0,
            "do_sample": False,
        }
    tokenizer, model = load_model_and_tokenizer(model_config)

    started_at = time.monotonic()
    completed_now = 0
    parse_failures = 0

    for batch_start in range(0, len(tasks), args.batch_size):
        batch = tasks[batch_start : batch_start + args.batch_size]
        batch_messages = [
            build_judge_messages(
                template_name,
                question=str(record["question_text"]),
                role_instruction_short=str(record["role_instruction_short"]),
                response=str(record["response_text"]),
            )
            for record, template_name in batch
        ]
        outputs = generate_responses(
            tokenizer,
            model,
            batch_messages,
            generation,
        )

        rows_to_append: list[dict[str, object]] = []
        for (record, template_name), raw_output in zip(batch, outputs, strict=False):
            base_row = {
                "item_id": str(record["item_id"]),
                "bank_axis": str(record["bank_axis"]),
                "role": str(record["role"]),
                "question_id": str(record["question_id"]),
                "prompt_id": int(record["prompt_id"]),
                "template_name": template_name,
                "template_label": TEMPLATE_LABELS[template_name],
                "scored_at": utc_now_iso(),
            }
            try:
                parsed = parse_imt_score_output(raw_output)
                row = {**base_row, **parsed, "parsed_ok": True}
                completed_now += 1
                completed_keys.add((str(record["item_id"]), template_name))
            except Exception as exc:  # noqa: BLE001
                row = {
                    **base_row,
                    "parsed_ok": False,
                    "parse_error": str(exc),
                    "raw_output": raw_output,
                }
                parse_failures += 1
            rows_to_append.append(row)

        append_jsonl(per_template_path, rows_to_append)
        write_json(
            progress_path,
            {
                "state": "running",
                "bank_items": len(bank_records),
                "templates": config.score_templates,
                "completed_prompt_scores": len(completed_keys) + completed_now,
                "completed_this_run": completed_now,
                "parse_failures_this_run": parse_failures,
                "reused_prompt_scores": reused_prompt_scores,
            },
        )

        if completed_now and (
            completed_now == len(rows_to_append)
            or (args.progress_every > 0 and completed_now % args.progress_every == 0)
        ):
            elapsed = max(time.monotonic() - started_at, 1e-6)
            rate = completed_now / elapsed
            remaining = len(tasks) - (batch_start + len(batch))
            eta_seconds = remaining / rate if rate > 0 else float("inf")
            print(
                f"[imt-scores] {completed_now}/{len(tasks)} prompt scores completed "
                f"({rate:.2f} prompts/s, eta {eta_seconds/60:.1f} min)"
            )

    per_template_rows = load_jsonl(per_template_path)
    aggregated_rows = aggregate_template_scores(per_template_rows, template_names=config.score_templates)
    write_jsonl(averaged_scores_path, aggregated_rows)
    aggregated_count = len(aggregated_rows)
    summary_payload = {
        "bank_items": len(bank_records),
        "templates": config.score_templates,
        "completed_prompt_scores": len(load_completed_template_keys(per_template_path)),
        "averaged_item_scores": aggregated_count,
        "parse_failures_this_run": parse_failures,
        "reused_prompt_scores": sum(1 for row in per_template_rows if row.get("reused_from_run_id")),
    }
    write_json(summary_path, summary_payload)
    if aggregated_count != len(bank_records):
        missing_count = len(bank_records) - aggregated_count
        write_stage_status(
            run_root,
            "score_imt_bank",
            "failed",
            {**summary_payload, "missing_averaged_items": missing_count},
        )
        raise RuntimeError(
            f"IMT scoring completed with {missing_count} items missing a full 3-template score set; "
            f"inspect {per_template_path}"
        )

    write_stage_status(run_root, "score_imt_bank", "completed", summary_payload)
    print(f"[imt-scores] wrote averaged scores for {aggregated_count} bank items")


if __name__ == "__main__":
    main()
