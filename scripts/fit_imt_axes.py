#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from deception_honesty_axis.common import make_run_id, load_jsonl
from deception_honesty_axis.imt_config import imt_run_root, load_imt_config
from deception_honesty_axis.imt_recovery import (
    fit_imt_axes,
    grouped_validation_summary,
    load_bank_records,
    write_fit_outputs,
    _prepare_fit_rows,  # noqa: PLC2701
)
from deception_honesty_axis.metadata import write_stage_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit PCA+ridge IMT axes from scored source-bank instances.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the IMT recovery config.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_imt_config(args.config.resolve())
    run_id = args.run_id or make_run_id()
    run_root = imt_run_root(config, run_id)
    bank_records_path = run_root / "results" / "bank_records.pt"
    averaged_scores_path = run_root / "results" / "scoring" / "averaged_scores.jsonl"
    if not bank_records_path.exists():
        raise FileNotFoundError(f"Missing bank records: {bank_records_path}")
    if not averaged_scores_path.exists():
        raise FileNotFoundError(f"Missing averaged scores: {averaged_scores_path}")

    write_stage_status(
        run_root,
        "fit_imt_axes",
        "running",
        {
            "fit_granularity": config.fit_granularity,
            "layer_number": config.fit_layer_number,
            "pca_components": config.pca_components,
            "ridge_alpha": config.ridge_alpha,
        },
    )

    bank_records = load_bank_records(bank_records_path)
    averaged_scores = {
        str(row["item_id"]): row
        for row in load_jsonl(averaged_scores_path)
    }
    fit_rows, features, targets = _prepare_fit_rows(
        bank_records,
        averaged_scores,
        layer_number=config.fit_layer_number,
        fit_granularity=config.fit_granularity,
    )
    if len(fit_rows) == 0:
        raise RuntimeError("No scored rows were available for IMT fitting.")

    role_groups = [str(row["role"]) for row in fit_rows]
    question_groups = [str(row["question_id"]) for row in fit_rows]
    fit_payload = fit_imt_axes(
        features,
        targets,
        pca_components=config.pca_components,
        ridge_alpha=config.ridge_alpha,
    )
    role_validation = grouped_validation_summary(
        features,
        targets,
        role_groups,
        pca_components=config.pca_components,
        ridge_alpha=config.ridge_alpha,
    )
    question_validation = grouped_validation_summary(
        features,
        targets,
        question_groups,
        pca_components=config.pca_components,
        ridge_alpha=config.ridge_alpha,
    )
    outputs = write_fit_outputs(
        run_root,
        fit_rows=fit_rows,
        fit_payload=fit_payload,
        fit_granularity=config.fit_granularity,
        layer_number=config.fit_layer_number,
        pca_components=config.pca_components,
        ridge_alpha=config.ridge_alpha,
        grouped_role_validation=role_validation,
        grouped_question_validation=question_validation,
    )
    write_stage_status(
        run_root,
        "fit_imt_axes",
        "completed",
        {
            "sample_count": len(fit_rows),
            "feature_dim": int(features.shape[1]),
            "pca_components_used": int(fit_payload["pca_components"].shape[0]),
            "grouped_role_validation": role_validation,
            "grouped_question_validation": question_validation,
            "artifacts": outputs,
        },
    )
    print(f"[imt-fit] wrote PCA+ridge artifacts under {run_root / 'results' / 'fit'}")


if __name__ == "__main__":
    main()
