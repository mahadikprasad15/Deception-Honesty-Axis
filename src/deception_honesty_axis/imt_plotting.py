from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from deception_honesty_axis.common import ensure_dir


IMT_AXIS_ORDER = ["q1", "q2", "q3", "q4"]
IMT_AXIS_TITLES = {
    "q1": "Quality Axis",
    "q2": "Quantity Axis",
    "q3": "Relation Axis",
    "q4": "Manner Axis",
}
IMT_DATASET_ORDER = [
    "Deception-AILiar-completion",
    "Deception-ConvincingGame-completion",
    "Deception-HarmPressureChoice-completion",
    "Deception-InstructedDeception-completion",
    "Deception-Mask-completion",
    "Deception-Roleplaying-completion",
]
IMT_DATASET_LABELS = {
    "Deception-AILiar-completion": "AILiar",
    "Deception-ConvincingGame-completion": "Convincing\nGame",
    "Deception-HarmPressureChoice-completion": "HarmPressure\nChoice",
    "Deception-InstructedDeception-completion": "Instructed\nDeception",
    "Deception-Mask-completion": "Mask",
    "Deception-Roleplaying-completion": "Roleplaying",
}
IMT_AXIS_COLOR_PAIRS = {
    "q1": ("#1f77b4", "#9ecae1"),
    "q2": ("#2ca02c", "#98df8a"),
    "q3": ("#d62728", "#ff9896"),
    "q4": ("#9467bd", "#c5b0d5"),
}


def _ordered_datasets(datasets: list[str]) -> list[str]:
    ordered = [name for name in IMT_DATASET_ORDER if name in datasets]
    ordered.extend(name for name in datasets if name not in ordered)
    return ordered


def save_four_axis_grouped_bars(
    output_path: Path,
    records: list[dict[str, Any]],
    *,
    variant_order: list[str],
    dataset_order: list[str],
    title: str,
    y_label: str = "AUROC",
    y_limits: tuple[float, float] = (0.35, 1.0),
) -> str:
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    datasets = _ordered_datasets(dataset_order)
    value_index = {
        (str(row["axis_name"]), str(row["variant"]), str(row["target_dataset"])): float(row["value"])
        for row in records
        if row.get("value") not in (None, "")
    }

    fig, axes_arr = plt.subplots(2, 2, figsize=(17, 10), sharey=True, constrained_layout=True)
    axes_flat = axes_arr.flatten()
    x = np.arange(len(datasets))
    width = 0.38 if len(variant_order) == 2 else 0.82 / max(1, len(variant_order))

    for ax, axis_name in zip(axes_flat, IMT_AXIS_ORDER, strict=False):
        color_pair = IMT_AXIS_COLOR_PAIRS[axis_name]
        for variant_index, variant in enumerate(variant_order):
            vals = [value_index.get((axis_name, variant, dataset), np.nan) for dataset in datasets]
            if len(variant_order) == 2:
                offset = (-width / 2) if variant_index == 0 else (width / 2)
            else:
                offset = -0.41 + width * 0.5 + variant_index * width
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                label=variant,
                color=color_pair[min(variant_index, len(color_pair) - 1)],
                edgecolor="white",
                linewidth=0.8,
            )
            for bar, value in zip(bars, vals, strict=False):
                if np.isnan(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.012,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                    color="#333333",
                )

        ax.set_title(IMT_AXIS_TITLES[axis_name], fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([IMT_DATASET_LABELS.get(name, name) for name in datasets], fontsize=10)
        ax.set_ylim(*y_limits)
        ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.margins(x=0.02)

    for ax in axes_flat[::2]:
        ax.set_ylabel(y_label, fontsize=11)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, fontsize=18, fontweight="bold")
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output_path)


def save_axis_variant_heatmap(
    output_path: Path,
    records: list[dict[str, Any]],
    *,
    variant_order: list[str],
    dataset_order: list[str],
    title: str,
    vmin: float = 0.35,
    vmax: float = 1.0,
) -> str:
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    datasets = _ordered_datasets(dataset_order)
    matrix = np.full((len(IMT_AXIS_ORDER) * len(variant_order), len(datasets)), np.nan, dtype=np.float32)
    row_labels: list[str] = []
    index = {
        (str(row["axis_name"]), str(row["variant"]), str(row["target_dataset"])): float(row["value"])
        for row in records
        if row.get("value") not in (None, "")
    }

    row_index = 0
    for axis_name in IMT_AXIS_ORDER:
        for variant in variant_order:
            row_labels.append(f"{IMT_AXIS_TITLES[axis_name].replace(' Axis', '')} | {variant}")
            for col_index, dataset in enumerate(datasets):
                value = index.get((axis_name, variant, dataset))
                if value is not None:
                    matrix[row_index, col_index] = value
            row_index += 1

    fig, ax = plt.subplots(figsize=(max(12, len(datasets) * 1.8), max(5.5, len(row_labels) * 0.6)))
    image = ax.imshow(matrix, vmin=vmin, vmax=vmax, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(datasets)), [IMT_DATASET_LABELS.get(name, name) for name in datasets], rotation=35, ha="right", fontsize=11)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=11)
    ax.set_title(title, fontsize=15, fontweight="bold")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isnan(value):
                continue
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if value < 0.55 else "black",
                fontweight="bold",
            )
    fig.colorbar(image, ax=ax, label="AUROC", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output_path)
