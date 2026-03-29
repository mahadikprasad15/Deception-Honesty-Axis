#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from deception_honesty_axis.analysis import mean_role_vectors
from deception_honesty_axis.common import ensure_dir, make_run_id, write_json
from deception_honesty_axis.config import analysis_run_root, corpus_root as resolve_corpus_root, load_config
from deception_honesty_axis.indexes import load_index
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.records import load_activation_records
from deception_honesty_axis.work_units import build_work_units


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate pooled activations into per-role vectors.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/deception_v1_llama32_3b.json"),
        help="Path to the experiment config file.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    corpus_path = resolve_corpus_root(config)
    run_id = args.run_id or make_run_id()
    run_root = analysis_run_root(config, "role-vectors", config.analysis["filter_name"], run_id)
    ensure_dir(run_root / "results")
    ensure_dir(run_root / "inputs")
    ensure_dir(run_root / "meta")
    ensure_dir(run_root / "checkpoints")
    ensure_dir(run_root / "logs")

    target_ids = {unit["item_id"] for unit in build_work_units(config.repo_root, config)}
    activation_index = load_index(corpus_path, "activations")
    selected_rows = [row for item_id, row in activation_index.items() if item_id in target_ids]
    records = load_activation_records(corpus_path, selected_rows)

    write_analysis_manifest(
        run_root,
        {
            "config_path": str(config.path),
            "corpus_root": str(corpus_path),
            "selected_items": len(records),
            "filter_name": config.analysis["filter_name"],
        },
    )
    write_stage_status(run_root, "build_role_vectors", "running", {"selected_items": len(records)})

    import torch

    role_vectors = mean_role_vectors(records)
    if not role_vectors:
        raise RuntimeError("No activation records matched the selected work units.")
    output_path = run_root / "results" / "role_vectors.pt"
    torch.save(role_vectors, output_path)
    write_json(
        run_root / "results" / "results.json",
        {
            "role_counts": {
                role: sum(1 for record in records if record["role"] == role)
                for role in sorted(role_vectors)
            },
            "output_path": str(output_path),
        },
    )
    write_json(run_root / "checkpoints" / "progress.json", {"selected_items": len(records), "completed": True})
    write_stage_status(run_root, "build_role_vectors", "completed", {"role_count": len(role_vectors)})
    print(f"Wrote role vectors to {output_path}")


if __name__ == "__main__":
    main()
