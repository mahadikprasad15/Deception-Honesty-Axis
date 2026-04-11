#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from deception_honesty_axis.activation_row_targets import load_activation_rows
from deception_honesty_axis.common import load_jsonl, write_json, write_jsonl
from deception_honesty_axis.metadata import write_stage_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze within-pair ranking accuracy for cached sycophancy eval outputs."
    )
    parser.add_argument(
        "--eval-run-dir",
        type=Path,
        required=True,
        help="Existing sycophancy eval run directory containing results/per_example_scores.jsonl.",
    )
    parser.add_argument("--activation-dataset-repo-id", default=None, help="HF Dataset repo with activation rows.")
    parser.add_argument("--activation-dataset-dir", type=Path, default=None, help="Local datasets.save_to_disk directory.")
    parser.add_argument("--activation-jsonl", type=Path, default=None, help="Local records.jsonl from extraction run.")
    parser.add_argument("--split", default="train", help="HF dataset split when using --activation-dataset-repo-id.")
    return parser.parse_args()


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "method",
                    "layer_spec",
                    "target_dataset",
                    "eligible_pair_count",
                    "deceptive_score_ranking_accuracy",
                    "probability_ranking_accuracy",
                    "tie_count",
                    "mean_deceptive_score_margin",
                    "median_deceptive_score_margin",
                    "mean_probability_margin",
                    "median_probability_margin",
                ]
            )
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_root = args.eval_run_dir.resolve()
    results_dir = run_root / "results"
    plots_dir = results_dir / "plots"
    score_rows_path = results_dir / "per_example_scores.jsonl"

    if not score_rows_path.exists():
        raise FileNotFoundError(f"Missing per_example_scores.jsonl in {results_dir}")

    activation_rows, source_manifest = load_activation_rows(
        activation_dataset_repo_id=args.activation_dataset_repo_id,
        activation_dataset_dir=args.activation_dataset_dir,
        activation_jsonl=args.activation_jsonl,
        split=args.split,
    )
    sample_metadata = {
        str(row.get("ids", "")): {
            "pair_id": str(row.get("pair_id", "")),
            "label": int(row.get("label", -1)),
        }
        for row in activation_rows
    }
    score_rows = load_jsonl(score_rows_path)
    if not score_rows:
        raise ValueError(f"No score rows found in {score_rows_path}")

    methods = _ordered_unique([str(row["method"]) for row in score_rows])
    layer_specs = _ordered_unique([str(row["layer_spec"]) for row in score_rows])
    target_datasets = _ordered_unique([str(row["target_dataset"]) for row in score_rows])

    write_stage_status(
        run_root,
        "analyze_sycophancy_pair_rankings",
        "running",
        {
            "score_rows": len(score_rows),
            "methods": methods,
            "layer_specs": layer_specs,
            "target_datasets": target_datasets,
        },
    )

    enriched_rows: list[dict[str, object]] = []
    missing_metadata_count = 0
    for row in score_rows:
        sample_id = str(row.get("sample_id", ""))
        metadata = sample_metadata.get(sample_id)
        if metadata is None:
            missing_metadata_count += 1
            continue
        enriched_rows.append(
            {
                **row,
                "pair_id": str(metadata["pair_id"]),
                "activation_label": int(metadata["label"]),
            }
        )

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in enriched_rows:
        key = (str(row["method"]), str(row["layer_spec"]), str(row["target_dataset"]))
        grouped.setdefault(key, []).append(row)

    for (method, layer_spec, target_dataset), rows in grouped.items():
        pair_groups: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            pair_id = str(row.get("pair_id", ""))
            if not pair_id:
                continue
            pair_groups.setdefault(pair_id, []).append(row)

        pair_margins: list[float] = []
        probability_margins: list[float] = []
        score_wins = 0
        probability_wins = 0
        ties = 0
        eligible_pairs = 0

        for pair_id, pair_rows in pair_groups.items():
            by_label: dict[int, list[dict[str, object]]] = {}
            for row in pair_rows:
                by_label.setdefault(int(row["activation_label"]), []).append(row)
            if sorted(by_label) != [0, 1]:
                continue
            if len(by_label[0]) != 1 or len(by_label[1]) != 1:
                continue

            negative_row = by_label[0][0]
            positive_row = by_label[1][0]
            pos_score = float(positive_row["deceptive_score"])
            neg_score = float(negative_row["deceptive_score"])
            pos_prob = float(positive_row["probability"])
            neg_prob = float(negative_row["probability"])
            score_margin = pos_score - neg_score
            probability_margin = pos_prob - neg_prob

            eligible_pairs += 1
            score_wins += int(score_margin > 0.0)
            probability_wins += int(probability_margin > 0.0)
            ties += int(score_margin == 0.0)
            pair_margins.append(score_margin)
            probability_margins.append(probability_margin)
            detail_rows.append(
                {
                    "method": method,
                    "layer_spec": layer_spec,
                    "target_dataset": target_dataset,
                    "pair_id": pair_id,
                    "positive_sample_id": positive_row["sample_id"],
                    "negative_sample_id": negative_row["sample_id"],
                    "positive_deceptive_score": pos_score,
                    "negative_deceptive_score": neg_score,
                    "deceptive_score_margin": score_margin,
                    "positive_probability": pos_prob,
                    "negative_probability": neg_prob,
                    "probability_margin": probability_margin,
                    "score_pair_correct": int(score_margin > 0.0),
                    "probability_pair_correct": int(probability_margin > 0.0),
                }
            )

        summary_rows.append(
            {
                "method": method,
                "layer_spec": layer_spec,
                "target_dataset": target_dataset,
                "eligible_pair_count": eligible_pairs,
                "deceptive_score_ranking_accuracy": float(score_wins / eligible_pairs) if eligible_pairs else None,
                "probability_ranking_accuracy": float(probability_wins / eligible_pairs) if eligible_pairs else None,
                "tie_count": ties,
                "mean_deceptive_score_margin": float(np.mean(pair_margins)) if pair_margins else None,
                "median_deceptive_score_margin": float(np.median(pair_margins)) if pair_margins else None,
                "mean_probability_margin": float(np.mean(probability_margins)) if probability_margins else None,
                "median_probability_margin": float(np.median(probability_margins)) if probability_margins else None,
            }
        )

    pair_summary_path = results_dir / "pair_ranking_summary.csv"
    pair_details_path = results_dir / "pair_ranking_details.jsonl"
    _write_csv(pair_summary_path, summary_rows)
    write_jsonl(pair_details_path, detail_rows)
    write_json(
        results_dir / "pair_ranking_summary.json",
        {
            "source_manifest": source_manifest,
            "score_row_count": len(score_rows),
            "enriched_score_row_count": len(enriched_rows),
            "missing_metadata_count": missing_metadata_count,
            "summary_row_count": len(summary_rows),
            "detail_row_count": len(detail_rows),
        },
    )

    try:
        import matplotlib.pyplot as plt

        plot_paths: list[str] = []
        by_layer = {}
        for row in summary_rows:
            by_layer.setdefault(str(row["layer_spec"]), []).append(row)
        for layer_spec, rows in by_layer.items():
            filtered = [row for row in rows if row["deceptive_score_ranking_accuracy"] is not None]
            if not filtered:
                continue
            x = np.arange(len(filtered), dtype=np.float32)
            y = [float(row["deceptive_score_ranking_accuracy"]) for row in filtered]
            labels = [str(row["target_dataset"]) for row in filtered]
            fig, ax = plt.subplots(figsize=(max(10, len(filtered) * 1.8), 5.5))
            bars = ax.bar(x, y, color="#0b7285")
            for bar, value in zip(bars, y, strict=False):
                ax.text(bar.get_x() + bar.get_width() / 2.0, min(0.98, value + 0.02), f"{value:.2f}", ha="center", va="bottom")
            ax.set_xticks(x, labels, rotation=30, ha="right")
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Within-Pair Ranking Accuracy")
            ax.set_title(f"Within-Pair Ranking Accuracy @ Layer {layer_spec}")
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            output_path = plots_dir / f"pair_ranking_accuracy__{layer_spec}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            plot_paths.append(str(output_path))
    except Exception:
        plot_paths = []

    write_stage_status(
        run_root,
        "analyze_sycophancy_pair_rankings",
        "completed",
        {
            "score_rows": len(score_rows),
            "enriched_score_rows": len(enriched_rows),
            "missing_metadata_count": missing_metadata_count,
            "summary_rows": len(summary_rows),
            "detail_rows": len(detail_rows),
            "plots": plot_paths,
        },
    )
    print(
        f"[sycophancy-pair-rankings] wrote {pair_summary_path}, {pair_details_path}, "
        f"and {len(plot_paths)} plot(s) under {plots_dir}"
    )


if __name__ == "__main__":
    main()
