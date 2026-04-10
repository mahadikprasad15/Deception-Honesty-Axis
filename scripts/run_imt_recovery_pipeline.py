#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from deception_honesty_axis.common import make_run_id
from deception_honesty_axis.imt_config import imt_run_root, load_imt_config
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status


STAGE_ORDER = [
    "imt_bank",
    "imt_scores",
    "imt_fit",
    "imt_axis_bundle",
    "imt_transfer",
    "imt_postprocess",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the IMT recovery pipeline end-to-end with resume-first stage handling.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the IMT recovery config.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id.")
    parser.add_argument("--stages", nargs="*", default=None, help=f"Subset of stages to run. Defaults to {' '.join(STAGE_ORDER)}.")
    parser.add_argument("--force-stage", nargs="*", default=[], help="Stage names to run even if outputs exist.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for scoring prompts.")
    parser.add_argument("--progress-every", type=int, default=20, help="Progress cadence for scoring and transfer stages.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def _resolve_stages(requested: list[str] | None) -> list[str]:
    if not requested:
        return list(STAGE_ORDER)
    unknown = sorted(set(requested) - set(STAGE_ORDER))
    if unknown:
        raise ValueError(f"Unknown IMT stages requested: {unknown}")
    return requested


def _read_stage_state(root: Path, stage_name: str) -> dict | None:
    status_path = root / "meta" / f"{stage_name}_status.json"
    if not status_path.exists():
        return None
    import json

    with status_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    config = load_imt_config(args.config.resolve())
    stages = _resolve_stages(args.stages)
    run_id = args.run_id or make_run_id()
    run_root = imt_run_root(config, run_id)
    repo_root = config.repo_root

    write_analysis_manifest(
        run_root,
        {
            "imt_config_path": str(config.path),
            "run_id": run_id,
            "bank_name": config.bank_name,
            "planned_stages": stages,
            "force_stages": sorted(set(args.force_stage)),
        },
    )

    env = dict(os.environ)
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}:{env['PYTHONPATH']}"

    stage_specs: dict[str, dict[str, object]] = {
        "imt_bank": {
            "status_name": "build_imt_bank",
            "artifact": run_root / "results" / "bank_records.pt",
            "command": [
                sys.executable,
                str(repo_root / "scripts" / "build_imt_bank.py"),
                "--config",
                str(config.path),
                "--run-id",
                run_id,
            ],
        },
        "imt_scores": {
            "status_name": "score_imt_bank",
            "artifact": run_root / "results" / "scoring" / "averaged_scores.jsonl",
            "command": [
                sys.executable,
                str(repo_root / "scripts" / "score_imt_bank.py"),
                "--config",
                str(config.path),
                "--run-id",
                run_id,
                "--batch-size",
                str(args.batch_size),
                "--progress-every",
                str(args.progress_every),
            ],
        },
        "imt_fit": {
            "status_name": "fit_imt_axes",
            "artifact": run_root / "results" / "fit" / "fit_artifacts.pt",
            "command": [
                sys.executable,
                str(repo_root / "scripts" / "fit_imt_axes.py"),
                "--config",
                str(config.path),
                "--run-id",
                run_id,
            ],
        },
        "imt_axis_bundle": {
            "status_name": "build_imt_axis_bundle",
            "artifact": run_root / "results" / "axis_bundle.pt",
            "command": [
                sys.executable,
                str(repo_root / "scripts" / "build_imt_axis_bundle.py"),
                "--config",
                str(config.path),
                "--run-id",
                run_id,
            ],
        },
        "imt_transfer": {
            "status_name": "evaluate_imt_transfer",
            "artifact": run_root / "results" / "eval" / "pairwise_metrics.csv",
            "command": [
                sys.executable,
                str(repo_root / "scripts" / "evaluate_imt_transfer.py"),
                "--config",
                str(config.path),
                "--run-id",
                run_id,
                "--print-every",
                str(args.progress_every),
            ],
        },
        "imt_postprocess": {
            "status_name": "postprocess_imt_transfer",
            "artifact": run_root / "results" / "eval" / "summary_by_axis.csv",
            "command": [
                sys.executable,
                str(repo_root / "scripts" / "postprocess_imt_transfer.py"),
                "--config",
                str(config.path),
                "--run-id",
                run_id,
            ],
        },
    }

    write_stage_status(
        run_root,
        "run_imt_recovery_pipeline",
        "running",
        {"planned_stages": stages, "current_stage": None, "run_id": run_id},
    )

    for stage_name in stages:
        spec = stage_specs[stage_name]
        status_name = str(spec["status_name"])
        expected_artifact = spec["artifact"]
        command = [str(item) for item in spec["command"]]
        forced = stage_name in set(args.force_stage)
        completed = _is_stage_complete(run_root, status_name, expected_artifact)
        action = "run"
        if completed and not forced:
            action = "skip-existing"

        print(f"[imt-pipeline] stage={stage_name} action={action} root={run_root}")
        write_stage_status(
            run_root,
            "run_imt_recovery_pipeline",
            "running",
            {"planned_stages": stages, "current_stage": stage_name, "action": action, "run_id": run_id},
        )
        if args.dry_run:
            print(f"[imt-pipeline] dry-run command: {' '.join(command)}")
            continue
        if action == "skip-existing":
            continue

        started = time.monotonic()
        log_path = run_root / "logs" / f"{stage_name}.log"
        _stream_command(command, cwd=repo_root, log_path=log_path, env=env)
        elapsed = time.monotonic() - started
        print(f"[imt-pipeline] completed stage={stage_name} in {elapsed/60:.1f} min artifact={expected_artifact}")

    write_stage_status(
        run_root,
        "run_imt_recovery_pipeline",
        "completed",
        {"planned_stages": stages, "current_stage": None, "run_id": run_id},
    )
    print(f"[imt-pipeline] completed run_id={run_id} manifest={run_root / 'meta' / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
