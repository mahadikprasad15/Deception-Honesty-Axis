#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from deception_honesty_axis.common import utc_now_iso, write_json
from deception_honesty_axis.config import corpus_root as resolve_corpus_root
from deception_honesty_axis.config import load_config
from deception_honesty_axis.indexes import ensure_corpus_layout, load_index, update_coverage
from deception_honesty_axis.work_units import build_work_units


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report missing rollout and activation items for the current target set.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/deception_v1_llama32_3b.json"),
        help="Path to the experiment config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    corpus_path = resolve_corpus_root(config)
    ensure_corpus_layout(corpus_path)

    work_units = build_work_units(config.repo_root, config)
    target_ids = {unit["item_id"] for unit in work_units}
    rollout_index = load_index(corpus_path, "rollouts")
    activation_index = load_index(corpus_path, "activations")
    missing_rollouts = sorted(target_ids - set(rollout_index))
    missing_activations = sorted(target_ids - set(activation_index))

    report = {
        "updated_at": utc_now_iso(),
        "target_items": len(target_ids),
        "rollouts_completed": len(set(rollout_index) & target_ids),
        "activations_completed": len(set(activation_index) & target_ids),
        "missing_rollouts": missing_rollouts,
        "missing_activations": missing_activations,
    }
    write_json(corpus_path / "reports" / "audit.json", report)
    update_coverage(
        corpus_path,
        {
            "target_items": len(target_ids),
            "rollouts_completed": report["rollouts_completed"],
            "activations_completed": report["activations_completed"],
            "updated_at": report["updated_at"],
        },
    )
    print(
        f"Target items={report['target_items']} rollouts_missing={len(missing_rollouts)} "
        f"activations_missing={len(missing_activations)}"
    )


if __name__ == "__main__":
    main()
