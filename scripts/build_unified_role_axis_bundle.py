#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from deception_honesty_axis.common import make_run_id, write_json
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.role_axis_transfer import write_role_axis_bundle
from deception_honesty_axis.unified_role_axis import (
    build_unified_role_axis_bundle_payload,
    load_unified_role_axis_config,
    unified_axis_run_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a unified harmful-behavior role-axis bundle from saved role-vector runs."
    )
    parser.add_argument("--config", type=Path, required=True, help="Unified role-axis config JSON.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if a completed unified axis bundle already exists for this run id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_unified_role_axis_config(args.config.resolve())
    run_id = args.run_id or make_run_id()
    run_root = unified_axis_run_root(config, run_id)
    for relative in ("inputs", "results", "logs", "checkpoints", "meta"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)

    bundle_path = run_root / "results" / "axis_bundle.pt"
    if bundle_path.exists() and not args.force:
        write_stage_status(
            run_root,
            "build_unified_role_axis_bundle",
            "completed",
            {
                "message": "Existing completed unified axis bundle found; skipping recompute.",
                "axis_bundle": str(bundle_path),
            },
        )
        print(f"[unified-role-axis] existing bundle found at {bundle_path}; skipping")
        return

    write_analysis_manifest(
        run_root,
        {
            "config_path": str(config.path),
            "run_id": run_id,
            "experiment_name": config.experiment_name,
            "behavior_name": config.behavior_name,
            "model_name": config.model_name,
            "axis_name": config.axis_name,
            "variant_name": config.variant_name,
            "question_mode": config.question_mode,
            "layer_specs": config.layer_specs,
            "pc_count": config.pc_count,
            "expected_pooling": config.expected_pooling,
            "question_sources": config.question_sources,
        },
    )
    write_json(run_root / "inputs" / "config.json", config.raw)
    write_json(run_root / "checkpoints" / "progress.json", {"state": "running", "run_id": run_id})
    write_stage_status(
        run_root,
        "build_unified_role_axis_bundle",
        "running",
        {"question_mode": config.question_mode, "layer_specs": config.layer_specs},
    )
    print(
        f"[unified-role-axis] building {config.axis_name}/{config.variant_name} "
        f"with mode={config.question_mode} and layers={config.layer_specs}"
    )

    payload = build_unified_role_axis_bundle_payload(config)
    bundle = payload["bundle"]

    import torch

    torch.save(payload["role_vectors"], run_root / "results" / "role_vectors.pt")
    write_json(
        run_root / "results" / "role_vectors_meta.json",
        {
            "activation_layer_numbers": payload["activation_layer_numbers"],
            "layer_count": len(payload["activation_layer_numbers"]),
            "pooling": config.expected_pooling,
            "pc_count": config.pc_count,
            "question_mode": config.question_mode,
            "source_manifests": payload["source_manifests"],
        },
    )
    write_json(run_root / "inputs" / "source_manifests.json", payload["source_manifests"])
    artifacts = write_role_axis_bundle(run_root, bundle)
    write_json(
        run_root / "results" / "results.json",
        {
            "axis_bundle": artifacts["axis_bundle"],
            "bundle_json": artifacts["bundle_json"],
            "layer_summary": artifacts["layer_summary"],
            "role_count": len(payload["role_vectors"]),
            "safe_role_count": len(payload["safe_roles"]),
            "harmful_role_count": len(payload["harmful_roles"]),
            "anchor_role": payload["anchor_role"],
            "activation_layer_numbers": payload["activation_layer_numbers"],
            "question_mode": config.question_mode,
            "pc_count": config.pc_count,
        },
    )
    write_json(
        run_root / "checkpoints" / "progress.json",
        {
            "state": "completed",
            "run_id": run_id,
            "role_count": len(payload["role_vectors"]),
            "safe_role_count": len(payload["safe_roles"]),
            "harmful_role_count": len(payload["harmful_roles"]),
            "axis_bundle": artifacts["axis_bundle"],
        },
    )
    write_stage_status(
        run_root,
        "build_unified_role_axis_bundle",
        "completed",
        {
            "question_mode": config.question_mode,
            "role_count": len(payload["role_vectors"]),
            "safe_role_count": len(payload["safe_roles"]),
            "harmful_role_count": len(payload["harmful_roles"]),
            "artifacts": artifacts,
            "pc_count": config.pc_count,
        },
    )
    print(
        f"[unified-role-axis] completed {config.axis_name}/{config.variant_name}; "
        f"wrote bundle to {run_root}"
    )


if __name__ == "__main__":
    main()
