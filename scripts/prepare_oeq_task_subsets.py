#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from deception_honesty_axis.common import ensure_dir, make_run_id, read_json, sha256_file, slugify, write_json, write_jsonl
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.oeq_preparation import (
    OEQ_TASK_LABEL_FIELDS,
    load_oeq_csv_rows,
    normalize_oeq_rows,
    prepare_oeq_task_rows,
    summarize_oeq_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare balanced OEQ task subsets for HF-backed activation extraction and evaluation."
    )
    parser.add_argument("--csv-path", type=Path, required=True, help="Local path to the raw OEQ CSV file.")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"), help="Repo artifact root.")
    parser.add_argument("--run-id", default=None, help="Optional fixed run id.")
    parser.add_argument(
        "--task-labels",
        nargs="+",
        default=list(OEQ_TASK_LABEL_FIELDS),
        help="OEQ label fields to prepare.",
    )
    parser.add_argument("--positive-count", type=int, default=100, help="Positive rows per task subset.")
    parser.add_argument("--negative-count", type=int, default=100, help="Clean negative rows per task subset.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Base random seed for task sampling.")
    parser.add_argument(
        "--push-to-hub-repo-id-prefix",
        default=None,
        help="Optional HF dataset repo id prefix like owner/oeq-prepared. Each task appends its own suffix.",
    )
    parser.add_argument("--private", action="store_true", help="Push private datasets when using --push-to-hub-repo-id-prefix.")
    return parser.parse_args()


def _task_repo_id(repo_id_prefix: str, dataset_source: str) -> str:
    owner, base = repo_id_prefix.split("/", 1)
    return f"{owner}/{slugify(base)}-{slugify(dataset_source)}"


def main() -> None:
    args = parse_args()
    run_id = args.run_id or make_run_id()
    run_root = ensure_dir(
        args.artifact_root
        / "runs"
        / "oeq-task-preparation"
        / "oeq"
        / f"balanced-{int(args.positive_count) + int(args.negative_count)}"
        / run_id
    )
    for relative in ("inputs", "results", "logs", "checkpoints", "meta"):
        ensure_dir(run_root / relative)

    csv_path = args.csv_path.resolve()
    write_analysis_manifest(
        run_root,
        {
            "run_id": run_id,
            "csv_path": str(csv_path),
            "csv_sha256": sha256_file(csv_path),
            "task_labels": [str(value) for value in args.task_labels],
            "positive_count": int(args.positive_count),
            "negative_count": int(args.negative_count),
            "sample_seed": int(args.sample_seed),
            "push_to_hub_repo_id_prefix": args.push_to_hub_repo_id_prefix,
        },
    )
    write_stage_status(run_root, "prepare_oeq_task_subsets", "running", {"run_id": run_id})

    raw_rows = load_oeq_csv_rows(csv_path)
    normalized_rows = normalize_oeq_rows(raw_rows)
    write_json(run_root / "inputs" / "raw_summary.json", summarize_oeq_rows(normalized_rows))
    write_jsonl(run_root / "inputs" / "normalized_rows.jsonl", normalized_rows)

    task_summaries: dict[str, dict] = {}
    progress_path = run_root / "checkpoints" / "progress.json"
    if progress_path.exists():
        progress = read_json(progress_path)
    else:
        progress = {"state": "running", "completed_tasks": []}
    completed_tasks = {str(value) for value in progress.get("completed_tasks", [])}
    progress["state"] = "running"
    write_json(progress_path, progress)

    from datasets import Dataset

    for task_label in args.task_labels:
        task_label_str = str(task_label)
        task_dir = ensure_dir(run_root / "results" / slugify(task_label_str))
        task_summary_path = task_dir / "summary.json"
        if task_label_str in completed_tasks and task_summary_path.exists():
            task_summaries[task_label_str] = read_json(task_summary_path)
            continue

        prepared_rows, summary = prepare_oeq_task_rows(
            normalized_rows,
            target_label=task_label_str,
            positive_count=int(args.positive_count),
            negative_count=int(args.negative_count),
            sample_seed=int(args.sample_seed),
        )
        write_jsonl(task_dir / "prepared_rows.jsonl", prepared_rows)
        dataset = Dataset.from_list(prepared_rows)
        dataset_dir = task_dir / "hf_dataset"
        dataset.save_to_disk(str(dataset_dir))

        repo_id = None
        if args.push_to_hub_repo_id_prefix:
            repo_id = _task_repo_id(str(args.push_to_hub_repo_id_prefix), str(summary["dataset_source"]))
            dataset.push_to_hub(repo_id, private=bool(args.private), token=os.environ.get("HF_TOKEN"))
        summary["dataset_dir"] = str(dataset_dir)
        if repo_id:
            summary["hf_repo_id"] = repo_id
        write_json(task_dir / "summary.json", summary)
        task_summaries[task_label_str] = summary
        completed_tasks.add(task_label_str)
        progress["completed_tasks"] = sorted(completed_tasks)
        write_json(progress_path, progress)

    write_json(run_root / "results" / "summary.json", {"tasks": task_summaries})
    write_json(
        progress_path,
        {"state": "completed", "completed_tasks": [str(value) for value in args.task_labels]},
    )
    write_stage_status(
        run_root,
        "prepare_oeq_task_subsets",
        "completed",
        {
            "run_id": run_id,
            "task_labels": [str(value) for value in args.task_labels],
            "task_count": len(task_summaries),
        },
    )
    print(f"[oeq-task-prep] completed run_id={run_id} root={run_root}")


if __name__ == "__main__":
    main()
