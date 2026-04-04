#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from deception_honesty_axis.common import make_run_id, read_json
from deception_honesty_axis.config import analysis_run_root, corpus_root as resolve_corpus_root, load_config
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.transfer_config import load_transfer_config


STAGE_ORDER = [
    "rollouts",
    "activations",
    "role_vectors",
    "pca",
    "axis_bundle",
    "transfer",
    "postprocess",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a synthetic variant pipeline end-to-end with loud progress and resume-first stage handling."
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        required=True,
        help="Path to the experiment config file.",
    )
    parser.add_argument(
        "--probe-config",
        type=Path,
        default=None,
        help="Optional transfer config file. Required for axis_bundle, transfer, and postprocess stages.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id for analysis stages.")
    parser.add_argument(
        "--stages",
        nargs="*",
        default=None,
        help=f"Subset of stages to run. Defaults to {' '.join(STAGE_ORDER)} through postprocess when probe-config is set.",
    )
    parser.add_argument(
        "--force-stage",
        nargs="*",
        default=[],
        help="Stage names to run even if the wrapper sees completed outputs.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for rollout generation.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Progress cadence to pass to generation, activation, and transfer stages.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned commands without executing them.")
    return parser.parse_args()


def _resolve_stages(requested: list[str] | None, has_probe_config: bool) -> list[str]:
    if requested:
        unknown = sorted(set(requested) - set(STAGE_ORDER))
        if unknown:
            raise ValueError(f"Unknown stages requested: {unknown}")
        return requested
    if has_probe_config:
        return list(STAGE_ORDER)
    return STAGE_ORDER[:4]


def _read_stage_state(root: Path, stage_name: str) -> dict | None:
    status_path = root / "meta" / f"{stage_name}_status.json"
    if not status_path.exists():
        return None
    return read_json(status_path)


def _is_stage_complete(root: Path, stage_name: str, expected_artifact: Path | None = None) -> bool:
    payload = _read_stage_state(root, stage_name)
    if payload is None or str(payload.get("status") or payload.get("state")) != "completed":
        return False
    return expected_artifact is None or expected_artifact.exists()


def _stream_command(command: list[str], *, cwd: Path, log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
        handle.write(f"[exit-code] {return_code}\n")
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    args = parse_args()
    experiment_config = load_config(args.experiment_config.resolve())
    transfer_config = load_transfer_config(args.probe_config.resolve()) if args.probe_config else None
    stages = _resolve_stages(args.stages, transfer_config is not None)
    if any(stage in {"axis_bundle", "transfer", "postprocess"} for stage in stages) and transfer_config is None:
        raise ValueError("--probe-config is required for axis_bundle, transfer, or postprocess stages")

    run_id = args.run_id or make_run_id()
    repo_root = experiment_config.repo_root
    corpus_root = resolve_corpus_root(experiment_config)
    role_vectors_root = analysis_run_root(experiment_config, "role-vectors", experiment_config.analysis["filter_name"], run_id)
    pca_root = analysis_run_root(experiment_config, experiment_config.experiment_name, "center-only", run_id)
    axis_bundle_root = None
    transfer_root = None
    if transfer_config is not None:
        axis_bundle_root = (
            experiment_config.artifact_root
            / "runs"
            / "role-axis-bundles"
            / experiment_config.model_slug
            / experiment_config.dataset_slug
            / experiment_config.role_set_slug
            / experiment_config.analysis.get("pooling", "unknown-pooling")
            / run_id
        )
        transfer_root = (
            experiment_config.artifact_root
            / "runs"
            / "role-axis-transfer"
            / experiment_config.model_slug
            / experiment_config.dataset_slug
            / experiment_config.role_set_slug
            / transfer_config.pooling
            / run_id
        )

    pipeline_root = (
        experiment_config.artifact_root
        / "runs"
        / "variant-pipeline"
        / experiment_config.model_slug
        / experiment_config.dataset_slug
        / experiment_config.role_set_slug
        / run_id
    )
    pipeline_root.joinpath("logs").mkdir(parents=True, exist_ok=True)
    write_analysis_manifest(
        pipeline_root,
        {
            "experiment_config_path": str(experiment_config.path),
            "probe_config_path": str(transfer_config.path) if transfer_config is not None else None,
            "run_id": run_id,
            "planned_stages": stages,
            "force_stages": sorted(set(args.force_stage)),
            "child_roots": {
                "corpus_root": str(corpus_root),
                "role_vectors_root": str(role_vectors_root),
                "pca_root": str(pca_root),
                "axis_bundle_root": str(axis_bundle_root) if axis_bundle_root is not None else None,
                "transfer_root": str(transfer_root) if transfer_root is not None else None,
            },
        },
    )

    env = dict(os.environ)
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}:{env['PYTHONPATH']}"

    stage_specs: dict[str, dict[str, object]] = {
        "rollouts": {
            "root": corpus_root,
            "status_name": "generate_rollouts",
            "artifact": corpus_root / "indexes" / "rollouts.jsonl",
            "command": [
                sys.executable,
                str(repo_root / "scripts" / "generate_rollouts.py"),
                "--config",
                str(experiment_config.path),
                "--batch-size",
                str(args.batch_size),
                "--progress-every",
                str(args.progress_every),
            ],
        },
        "activations": {
            "root": corpus_root,
            "status_name": "extract_activations",
            "artifact": corpus_root / "indexes" / "activations.jsonl",
            "command": [
                sys.executable,
                str(repo_root / "scripts" / "extract_activations.py"),
                "--config",
                str(experiment_config.path),
                "--progress-every",
                str(args.progress_every),
            ],
        },
        "role_vectors": {
            "root": role_vectors_root,
            "status_name": "build_role_vectors",
            "artifact": role_vectors_root / "results" / "role_vectors.pt",
            "command": [
                sys.executable,
                str(repo_root / "scripts" / "build_role_vectors.py"),
                "--config",
                str(experiment_config.path),
                "--run-id",
                run_id,
            ],
        },
        "pca": {
            "root": pca_root,
            "status_name": "run_pca_analysis",
            "artifact": pca_root / "results" / "results.json",
            "command": [
                sys.executable,
                str(repo_root / "scripts" / "run_pca_analysis.py"),
                "--config",
                str(experiment_config.path),
                "--vectors-run-dir",
                str(role_vectors_root),
                "--run-id",
                run_id,
            ],
        },
    }
    if transfer_config is not None and axis_bundle_root is not None and transfer_root is not None:
        axis_bundle_command = [
            sys.executable,
            str(repo_root / "scripts" / "build_role_axis_bundle.py"),
            "--config",
            str(transfer_config.path),
            "--vectors-run-dir",
            str(role_vectors_root),
            "--run-id",
            run_id,
        ]
        if "axis_bundle" in args.force_stage:
            axis_bundle_command.append("--force")
        stage_specs["axis_bundle"] = {
            "root": axis_bundle_root,
            "status_name": "build_role_axis_bundle",
            "artifact": axis_bundle_root / "results" / "axis_bundle.pt",
            "command": axis_bundle_command,
        }
        stage_specs["transfer"] = {
            "root": transfer_root,
            "status_name": "evaluate_role_axis_transfer",
            "artifact": transfer_root / "results" / "pairwise_metrics.csv",
            "command": [
                sys.executable,
                str(repo_root / "scripts" / "evaluate_role_axis_transfer.py"),
                "--config",
                str(transfer_config.path),
                "--axis-bundle-run-dir",
                str(axis_bundle_root),
                "--run-id",
                run_id,
                "--print-every",
                str(args.progress_every),
            ],
        }
        stage_specs["postprocess"] = {
            "root": transfer_root,
            "status_name": "postprocess_role_axis_transfer",
            "artifact": transfer_root / "results" / "summary_by_method.csv",
            "command": [
                sys.executable,
                str(repo_root / "scripts" / "postprocess_role_axis_transfer.py"),
                "--transfer-run-dir",
                str(transfer_root),
            ],
        }

    write_stage_status(
        pipeline_root,
        "run_variant_pipeline",
        "running",
        {"planned_stages": stages, "current_stage": None, "run_id": run_id},
    )

    for stage_name in stages:
        spec = stage_specs[stage_name]
        stage_root = spec["root"]
        status_name = str(spec["status_name"])
        expected_artifact = spec["artifact"]
        command = [str(item) for item in spec["command"]]
        forced = stage_name in set(args.force_stage)
        completed = _is_stage_complete(stage_root, status_name, expected_artifact)
        action = "run"
        if completed and not forced:
            action = "skip-existing"

        print(f"[variant-pipeline] stage={stage_name} action={action} root={stage_root}")
        write_stage_status(
            pipeline_root,
            "run_variant_pipeline",
            "running",
            {"planned_stages": stages, "current_stage": stage_name, "action": action, "run_id": run_id},
        )
        if args.dry_run:
            print(f"[variant-pipeline] dry-run command: {' '.join(command)}")
            continue
        if action == "skip-existing":
            continue

        started = time.monotonic()
        log_path = pipeline_root / "logs" / f"{stage_name}.log"
        _stream_command(command, cwd=repo_root, log_path=log_path, env=env)
        elapsed = time.monotonic() - started
        print(
            f"[variant-pipeline] completed stage={stage_name} in {elapsed/60:.1f} min "
            f"artifact={expected_artifact}"
        )

    write_stage_status(
        pipeline_root,
        "run_variant_pipeline",
        "completed",
        {"planned_stages": stages, "current_stage": None, "run_id": run_id},
    )
    print(f"[variant-pipeline] completed run_id={run_id} manifest={pipeline_root / 'meta' / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
