#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

from deception_honesty_axis.common import load_jsonl, read_json
from deception_honesty_axis.metadata import write_stage_status
from deception_honesty_axis.role_axis_transfer import DATASET_LABELS, ZERO_SHOT_SOURCE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate zero-shot score histograms from an existing role-axis transfer run."
    )
    parser.add_argument(
        "--transfer-run-dir",
        type=Path,
        required=True,
        help="Existing role-axis transfer run directory containing results/per_example_scores.jsonl.",
    )
    return parser.parse_args()


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    args = parse_args()
    run_root = args.transfer_run_dir.resolve()
    results_dir = run_root / "results"
    plots_dir = results_dir / "plots"
    score_rows_path = results_dir / "per_example_scores.jsonl"
    manifest_path = run_root / "meta" / "run_manifest.json"

    if not score_rows_path.exists():
        raise FileNotFoundError(f"Missing per_example_scores.jsonl in {results_dir}")

    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    methods = [str(item) for item in manifest.get("methods", [])]
    layer_specs = [str(item) for item in manifest.get("layer_specs", [])]
    datasets = [str(item) for item in manifest.get("datasets", [])]

    score_rows = [
        row
        for row in load_jsonl(score_rows_path)
        if str(row.get("source_dataset")) == ZERO_SHOT_SOURCE
    ]
    if not score_rows:
        raise ValueError(f"No zero-shot score rows found in {score_rows_path}")

    if not methods:
        methods = _ordered_unique([str(row["method"]) for row in score_rows])
    methods = [method for method in methods if method in {"contrast_zero_shot", "pc1_zero_shot"}]
    if not layer_specs:
        layer_specs = _ordered_unique([str(row["layer_spec"]) for row in score_rows])
    if not datasets:
        datasets = _ordered_unique([str(row["target_dataset"]) for row in score_rows])

    write_stage_status(
        run_root,
        "plot_role_axis_score_histograms",
        "running",
        {
            "score_rows": len(score_rows),
            "methods": methods,
            "layer_specs": layer_specs,
            "datasets": datasets,
        },
    )

    histogram_paths: list[str] = []
    bins = np.linspace(0.0, 1.0, 21)
    ncols = 3
    nrows = int(math.ceil(len(datasets) / ncols))

    for method in methods:
        for layer_spec in layer_specs:
            filtered = [
                row
                for row in score_rows
                if str(row["method"]) == method and str(row["layer_spec"]) == layer_spec
            ]
            if not filtered:
                continue

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(5.5 * ncols, 3.6 * max(1, nrows)),
                constrained_layout=True,
            )
            axes_array = np.atleast_1d(axes).reshape(-1)

            for ax in axes_array:
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(bottom=0.0)
                ax.grid(True, axis="y", color="#e9ecef")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            for ax, dataset in zip(axes_array, datasets, strict=False):
                dataset_rows = [row for row in filtered if str(row["target_dataset"]) == dataset]
                label_zero = [float(row["probability"]) for row in dataset_rows if int(row["label"]) == 0]
                label_one = [float(row["probability"]) for row in dataset_rows if int(row["label"]) == 1]
                ax.hist(label_zero, bins=bins, alpha=0.6, color="#4c78a8", label="label=0", density=True)
                ax.hist(label_one, bins=bins, alpha=0.6, color="#e45756", label="label=1", density=True)
                ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=12, fontweight="bold")
                ax.set_xlabel("Predicted deceptive probability")
                ax.set_ylabel("Density")

            for ax in axes_array[len(datasets):]:
                ax.axis("off")

            handles, labels = axes_array[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
            fig.suptitle(
                f"Zero-Shot Probability Histograms: {method} @ {layer_spec}",
                fontsize=16,
                fontweight="bold",
            )
            output_path = plots_dir / f"{method}__{layer_spec}__probability_histograms.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            histogram_paths.append(str(output_path))

    payload = {
        "score_rows": len(score_rows),
        "methods": methods,
        "layer_specs": layer_specs,
        "datasets": datasets,
        "histograms": histogram_paths,
    }
    write_stage_status(run_root, "plot_role_axis_score_histograms", "completed", payload)
    print(
        f"[role-axis-transfer-histograms] wrote {len(histogram_paths)} figures under {plots_dir}"
    )


if __name__ == "__main__":
    main()
