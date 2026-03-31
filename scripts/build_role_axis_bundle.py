#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from deception_honesty_axis.common import make_run_id
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.role_axis_transfer import build_role_axis_bundle, write_role_axis_bundle
from deception_honesty_axis.transfer_config import load_transfer_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reusable role-axis bundle from saved role vectors.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/probes/role_axis_transfer_v1.json"),
        help="Path to the role-axis transfer config file.",
    )
    parser.add_argument("--vectors-run-dir", type=Path, required=True, help="Role-vector run directory.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transfer_config = load_transfer_config(args.config.resolve())
    experiment_config = transfer_config.role_experiment_config
    run_id = args.run_id or make_run_id()
    run_root = (
        experiment_config.artifact_root
        / "runs"
        / "role-axis-bundles"
        / experiment_config.model_slug
        / experiment_config.dataset_slug
        / experiment_config.role_set_slug
        / experiment_config.analysis.get("pooling", "unknown-pooling")
        / run_id
    )
    vectors_run_dir = args.vectors_run_dir.resolve()

    import torch

    role_vectors = torch.load(vectors_run_dir / "results" / "role_vectors.pt", map_location="cpu")
    write_analysis_manifest(
        run_root,
        {
            "config_path": str(transfer_config.path),
            "role_experiment_config_path": str(experiment_config.path),
            "vectors_run_dir": str(vectors_run_dir),
            "honest_roles": transfer_config.honest_roles,
            "deceptive_roles": transfer_config.deceptive_roles,
            "anchor_role": transfer_config.anchor_role,
            "layer_specs": transfer_config.layer_specs,
        },
    )
    write_stage_status(run_root, "build_role_axis_bundle", "running", {"role_count": len(role_vectors)})

    bundle = build_role_axis_bundle(
        role_vectors=role_vectors,
        honest_roles=transfer_config.honest_roles,
        deceptive_roles=transfer_config.deceptive_roles,
        anchor_role=transfer_config.anchor_role,
        layer_specs=transfer_config.layer_specs,
    )
    artifacts = write_role_axis_bundle(run_root, bundle)
    write_stage_status(
        run_root,
        "build_role_axis_bundle",
        "completed",
        {
            "bundle_layers": len(bundle["resolved_layer_specs"]),
            "artifacts": artifacts,
        },
    )
    print(f"Wrote role-axis bundle to {run_root}")


if __name__ == "__main__":
    main()
