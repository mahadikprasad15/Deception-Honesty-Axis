#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from deception_honesty_axis.analysis import (
    cosine_similarity,
    run_centered_pca,
    save_results_json,
    write_pca_report_artifacts,
)
from deception_honesty_axis.common import ensure_dir, make_run_id, read_json, write_json
from deception_honesty_axis.config import analysis_run_root, load_config
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PCA over role vectors and project the default anchor.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/deception_v1_llama32_3b.json"),
        help="Path to the experiment config file.",
    )
    parser.add_argument("--vectors-run-dir", type=Path, required=True, help="Role-vector run directory.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    run_id = args.run_id or make_run_id()
    run_root = analysis_run_root(config, config.experiment_name, "center-only", run_id)
    ensure_dir(run_root / "results" / "plots")
    ensure_dir(run_root / "meta")
    ensure_dir(run_root / "checkpoints")
    ensure_dir(run_root / "logs")

    vectors_run_dir = args.vectors_run_dir.resolve()
    import torch

    role_vectors = torch.load(vectors_run_dir / "results" / "role_vectors.pt", map_location="cpu")
    role_manifest = read_json(config.role_manifest_path)
    anchor_role = role_manifest["anchor_role"]
    role_names = [entry["name"] for entry in role_manifest["roles"] if not entry.get("anchor_only", False)]
    anchor_vector = role_vectors[anchor_role]
    mean_deception = torch.stack([role_vectors[name] for name in role_names], dim=0).mean(dim=0)
    contrast_vector = anchor_vector - mean_deception

    write_analysis_manifest(
        run_root,
        {
            "config_path": str(config.path),
            "vectors_run_dir": str(vectors_run_dir),
            "role_names": role_names,
            "anchor_role": anchor_role,
        },
    )
    write_stage_status(run_root, "run_pca_analysis", "running", {"role_count": len(role_names)})

    per_layer_results: list[dict] = []
    layer_count = int(next(iter(role_vectors.values())).shape[0])

    for layer_idx in range(layer_count):
        layer_role_vectors = {name: vector[layer_idx].to(torch.float32) for name, vector in role_vectors.items()}
        pca = run_centered_pca(layer_role_vectors, role_names)
        components = pca["components"]
        centered_anchor = layer_role_vectors[anchor_role] - pca["mean_vector"]
        anchor_projection = centered_anchor @ components.T
        layer_contrast = contrast_vector[layer_idx].to(torch.float32)
        pc_cosines = [cosine_similarity(layer_contrast, component) for component in components]
        per_layer_results.append(
            {
                "layer": layer_idx,
                "explained_variance_ratio": pca["explained_variance_ratio"],
                "role_projections": {
                    role: pca["projections"][role_offset].tolist()
                    for role_offset, role in enumerate(role_names)
                },
                "anchor_projection": anchor_projection.tolist(),
                "contrast_pc_cosines": pc_cosines,
            }
        )

    results_payload = {"anchor_role": anchor_role, "role_names": role_names, "layers": per_layer_results}
    save_results_json(run_root / "results" / "results.json", results_payload)
    report_artifacts = write_pca_report_artifacts(run_root, per_layer_results, anchor_role, role_names)
    write_json(run_root / "checkpoints" / "progress.json", {"layers_processed": layer_count, "completed": True})
    write_stage_status(
        run_root,
        "run_pca_analysis",
        "completed",
        {"layers_processed": layer_count, "report_artifacts": report_artifacts},
    )
    print(f"Wrote PCA analysis to {run_root}")


if __name__ == "__main__":
    main()
