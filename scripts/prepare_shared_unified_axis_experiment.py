#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from deception_honesty_axis.common import ensure_dir, write_json
from deception_honesty_axis.config import find_repo_root
from deception_honesty_axis.shared_unified_axis_prep import prepare_shared_unified_axis_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a shared-question quantity+sycophancy experiment and matching unified-axis bundle config."
    )
    parser.add_argument(
        "--quantity-config",
        type=Path,
        default=Path("configs/experiments/quantity_axis_v2_llama32_3b.json"),
        help="Quantity/deception experiment config.",
    )
    parser.add_argument(
        "--sycophancy-config",
        type=Path,
        default=Path("configs/experiments/sycophancy_pilot_v1_llama32_3b.json"),
        help="Sycophancy experiment config.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts"),
        help="Artifact root for prepared files and downstream runs.",
    )
    parser.add_argument(
        "--questions-per-source",
        type=int,
        default=None,
        help="Balanced number of questions to take from each source file. Defaults to min(native source counts).",
    )
    parser.add_argument("--pc-count", type=int, default=64, help="Number of PCs to persist in the shared bundle.")
    parser.add_argument("--layer-number", type=int, default=14, help="Activation layer number to retain.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id for the prepared experiment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(Path.cwd())
    artifact_root = args.artifact_root if args.artifact_root.is_absolute() else (repo_root / args.artifact_root)
    ensure_dir(artifact_root)

    prepared = prepare_shared_unified_axis_experiment(
        quantity_experiment_config_path=(repo_root / args.quantity_config).resolve(),
        sycophancy_experiment_config_path=(repo_root / args.sycophancy_config).resolve(),
        artifact_root=artifact_root.resolve(),
        questions_per_source=args.questions_per_source,
        pc_count=args.pc_count,
        layer_number=args.layer_number,
        run_id=args.run_id,
    )
    write_json(
        prepared.prep_root / "results.json",
        {
            "run_id": prepared.run_id,
            "prep_root": str(prepared.prep_root),
            "question_file": str(prepared.question_file),
            "role_manifest_file": str(prepared.role_manifest_file),
            "experiment_config_file": str(prepared.experiment_config_file),
            "bundle_config_file": str(prepared.bundle_config_file),
            "role_vectors_run_dir": str(prepared.role_vectors_run_dir),
            "safe_roles": prepared.safe_roles,
            "harmful_roles": prepared.harmful_roles,
        },
    )
    print(f"[shared-unified-axis-prep] run_id={prepared.run_id}")
    print(f"[shared-unified-axis-prep] question_file={prepared.question_file}")
    print(f"[shared-unified-axis-prep] role_manifest_file={prepared.role_manifest_file}")
    print(f"[shared-unified-axis-prep] experiment_config_file={prepared.experiment_config_file}")
    print(f"[shared-unified-axis-prep] bundle_config_file={prepared.bundle_config_file}")
    print(f"[shared-unified-axis-prep] role_vectors_run_dir={prepared.role_vectors_run_dir}")


if __name__ == "__main__":
    main()
