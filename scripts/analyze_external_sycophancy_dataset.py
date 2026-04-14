#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from deception_honesty_axis.common import ensure_dir, make_run_id, slugify, write_json, write_jsonl
from deception_honesty_axis.external_sycophancy_benchmarks import (
    load_external_sycophancy_examples,
    sample_external_examples,
    summarize_external_examples,
)
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze pair structure and label balance for an external sycophancy benchmark split."
    )
    parser.add_argument("--source-repo-id", required=True, help="HF dataset repo containing the source benchmark.")
    parser.add_argument("--source-split", required=True, help="HF dataset split to load.")
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional adapter name. If omitted, infer it from the dataset repo id.",
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"), help="Repo artifact root.")
    parser.add_argument("--run-id", default=None, help="Optional fixed run id.")
    parser.add_argument("--target-name", default=None, help="Optional target slug used only for artifact path naming.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for a sampled comparison slice.")
    parser.add_argument(
        "--balanced-labels",
        action="store_true",
        help="Apply balanced-label subsampling when building the sampled comparison slice.",
    )
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed for sampled comparison slice.")
    parser.add_argument(
        "--write-selected-examples",
        action="store_true",
        help="Persist the sampled normalized examples jsonl under inputs/selected_examples.jsonl.",
    )
    parser.add_argument(
        "--max-repeated-pairs",
        type=int,
        default=50,
        help="Maximum repeated pair groups to write for inspection.",
    )
    return parser.parse_args()


def _prompt_preview(example) -> str:  # noqa: ANN001
    for message in reversed(example.prompt_messages):
        if str(message.get("role") or "") == "user":
            return str(message.get("content") or "")[:240]
    return str(example.prompt_messages[-1].get("content") or "")[:240]


def main() -> None:
    args = parse_args()
    target_name = args.target_name or f"{args.source_repo_id.split('/')[-1]}-{args.source_split}"
    run_id = args.run_id or make_run_id()
    run_root = ensure_dir(
        args.artifact_root
        / "runs"
        / "external-sycophancy-dataset-analysis"
        / slugify(target_name)
        / run_id
    )
    for relative in ("inputs", "results", "logs", "meta"):
        ensure_dir(run_root / relative)

    write_analysis_manifest(
        run_root,
        {
            "run_id": run_id,
            "target_name": target_name,
            "source_repo_id": args.source_repo_id,
            "source_split": args.source_split,
            "adapter_name": args.adapter,
            "max_samples": args.max_samples,
            "balanced_labels": bool(args.balanced_labels),
            "sample_seed": int(args.sample_seed),
        },
    )
    write_stage_status(run_root, "analyze_external_sycophancy_dataset", "running", {"run_id": run_id})

    full_examples, source_manifest = load_external_sycophancy_examples(
        repo_id=args.source_repo_id,
        split=args.source_split,
        adapter_name=args.adapter,
        max_samples=None,
        balanced_labels=False,
        sample_seed=args.sample_seed,
    )
    sampled_examples = sample_external_examples(
        full_examples,
        max_samples=args.max_samples,
        balanced_labels=bool(args.balanced_labels),
        sample_seed=args.sample_seed,
    )

    full_summary = summarize_external_examples(full_examples)
    sampled_summary = summarize_external_examples(sampled_examples)

    repeated_pair_rows = []
    grouped_examples: dict[str, list] = {}
    for example in full_examples:
        grouped_examples.setdefault(example.pair_id, []).append(example)
    repeated_groups = [
        group
        for group in grouped_examples.values()
        if len(group) > 1
    ]
    repeated_groups.sort(key=lambda group: (-len(group), group[0].pair_id))
    for group in repeated_groups[: max(0, int(args.max_repeated_pairs))]:
        ordered_group = sorted(group, key=lambda example: (example.label, example.ids))
        repeated_pair_rows.append(
            {
                "pair_id": ordered_group[0].pair_id,
                "group_size": len(ordered_group),
                "labels": [int(example.label) for example in ordered_group],
                "label_names": [example.label_name for example in ordered_group],
                "source_record_ids": [example.source_record_id for example in ordered_group],
                "ids": [example.ids for example in ordered_group],
                "prompt_preview": _prompt_preview(ordered_group[0]),
            }
        )

    write_json(run_root / "inputs" / "source_manifest.json", source_manifest)
    write_json(run_root / "results" / "full_summary.json", full_summary)
    write_json(run_root / "results" / "sampled_summary.json", sampled_summary)
    write_jsonl(run_root / "results" / "repeated_pair_groups.jsonl", repeated_pair_rows)
    if args.write_selected_examples:
        write_jsonl(run_root / "inputs" / "selected_examples.jsonl", (example.manifest_row() for example in sampled_examples))

    write_stage_status(
        run_root,
        "analyze_external_sycophancy_dataset",
        "completed",
        {
            "run_id": run_id,
            "target_name": target_name,
            "record_count": full_summary["record_count"],
            "sampled_record_count": sampled_summary["record_count"],
            "complete_balanced_pair_count": full_summary["complete_balanced_pair_count"],
        },
    )
    print(f"[external-sycophancy-dataset-analysis] completed run_id={run_id} root={run_root}")


if __name__ == "__main__":
    main()
