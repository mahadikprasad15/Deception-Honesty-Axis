#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from deception_honesty_axis.common import read_json, slugify, write_json
from deception_honesty_axis.config import find_repo_root


DEFAULT_TARGETS = [
    ("Sycophancy Dataset", "ariana-sycophancy-train"),
    ("Open-Ended Sycophancy", "open-ended-sycophancy-train"),
    ("OEQ Validation", "oeq-validation-human"),
    ("OEQ Indirectness", "oeq-indirectness-human"),
    ("OEQ Framing", "oeq-framing-human"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an activation-row transfer config from completed external sycophancy "
            "activation extraction runs."
        )
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"), help="Repo artifact root.")
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model id used for extraction.")
    parser.add_argument("--expected-layer-number", type=int, default=16, help="Expected activation layer.")
    parser.add_argument("--expected-pooling", default="mean_response", help="Expected activation pooling.")
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("configs/probes/activation_row_transfer_sycophancy_pilot_v1_llama3_8b_external.json"),
        help="Config path to write.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        metavar="DISPLAY=TARGET_SLUG",
        help=(
            "Optional dataset targets. Defaults to the five paper sycophancy datasets. "
            "Each value should be DISPLAY_NAME=external-extraction-target-slug."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional extraction run id to use for every target. By default, use the newest completed run per target.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing targets instead of failing.",
    )
    return parser.parse_args()


def parse_targets(values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        return list(DEFAULT_TARGETS)
    targets: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Target {value!r} must be DISPLAY_NAME=target-slug")
        display, slug = value.split("=", 1)
        display = display.strip()
        slug = slug.strip()
        if not display or not slug:
            raise ValueError(f"Target {value!r} must have non-empty display and slug")
        targets.append((display, slug))
    return targets


def run_is_completed(run_dir: Path) -> bool:
    status_path = run_dir / "meta" / "stage_status.json"
    if not status_path.exists():
        return False
    status = read_json(status_path)
    stages = status.get("stages", {})
    stage = stages.get("extract_external_sycophancy_activations", {})
    return stage.get("status") == "completed"


def find_records_jsonl(
    artifact_root: Path,
    *,
    model_name: str,
    target_slug: str,
    requested_run_id: str | None,
) -> Path | None:
    model_slug = slugify(model_name)
    target_root = (
        artifact_root
        / "runs"
        / "external-sycophancy-activation-extraction"
        / slugify(target_slug)
        / model_slug
    )
    if requested_run_id:
        records_path = target_root / requested_run_id / "results" / "records.jsonl"
        return records_path if records_path.exists() else None
    if not target_root.exists():
        return None
    candidates = [
        run_dir / "results" / "records.jsonl"
        for run_dir in sorted(target_root.iterdir())
        if run_dir.is_dir() and (run_dir / "results" / "records.jsonl").exists()
    ]
    completed = [path for path in candidates if run_is_completed(path.parents[1])]
    return (completed or candidates)[-1] if candidates else None


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(Path.cwd())
    artifact_root = args.artifact_root if args.artifact_root.is_absolute() else repo_root / args.artifact_root
    output_config = args.output_config if args.output_config.is_absolute() else repo_root / args.output_config

    datasets: list[dict[str, Any]] = []
    missing: list[str] = []
    for display_name, target_slug in parse_targets(args.targets):
        records_path = find_records_jsonl(
            artifact_root,
            model_name=args.model_name,
            target_slug=target_slug,
            requested_run_id=args.run_id,
        )
        if records_path is None:
            missing.append(f"{display_name} ({target_slug})")
            continue
        try:
            relative_records_path = records_path.relative_to(repo_root)
        except ValueError:
            relative_records_path = records_path
        datasets.append(
            {
                "name": display_name,
                "behavior": "sycophancy",
                "source_kind": "activation_rows",
                "activation_jsonl": str(relative_records_path),
            }
        )

    if missing and not args.allow_missing:
        raise FileNotFoundError(
            "Missing completed activation extraction records for: "
            + ", ".join(missing)
            + ". Re-run with --allow-missing to write a partial config."
        )
    if not datasets:
        raise ValueError("No sycophancy activation-row datasets were found.")

    config = {
        "artifacts": {"root": "artifacts"},
        "experiment": {
            "name": "activation-row-transfer",
            "behavior": "sycophancy",
            "model_name": args.model_name,
            "dataset_set": "sycophancy-external",
        },
        "activation_rows": {
            "expected_pooling": args.expected_pooling,
            "expected_layer_number": int(args.expected_layer_number),
        },
        "datasets": datasets,
        "evaluation": {
            "methods": ["activation_logistic"],
            "random_seed": 42,
            "logistic_max_iter": 1000,
        },
    }
    write_json(output_config, config)
    print(f"[sycophancy-transfer-config] wrote {len(datasets)} datasets to {output_config}")
    if missing:
        print("[sycophancy-transfer-config] skipped missing: " + ", ".join(missing))


if __name__ == "__main__":
    main()
