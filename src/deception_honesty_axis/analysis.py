from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from deception_honesty_axis.common import ensure_dir, write_json


def mean_role_vectors(records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    import torch

    grouped: dict[str, list[torch.Tensor]] = {}
    for record in records:
        grouped.setdefault(record["role"], []).append(record["pooled_activations"])
    return {
        role: torch.stack(tensors, dim=0).mean(dim=0)
        for role, tensors in grouped.items()
    }


def infer_activation_layer_numbers(records: list[dict[str, Any]]) -> list[int]:
    if not records:
        return []

    first_tensor = records[0]["pooled_activations"]
    layer_count = int(first_tensor.shape[0])
    fallback = list(range(1, layer_count + 1))
    inferred: list[int] | None = None
    for record in records:
        raw_layers = record.get("activation_layer_numbers")
        if raw_layers is None:
            continue
        layer_numbers = [int(layer_number) for layer_number in raw_layers]
        if len(layer_numbers) != int(record["pooled_activations"].shape[0]):
            raise ValueError(
                "activation_layer_numbers length does not match pooled_activations layer dimension "
                f"for item {record.get('item_id')}"
            )
        if inferred is None:
            inferred = layer_numbers
        elif inferred != layer_numbers:
            raise ValueError("Mixed activation_layer_numbers metadata found in the selected activation records")
    return inferred or fallback


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    import torch

    denom = torch.linalg.norm(left) * torch.linalg.norm(right)
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(left, right) / denom)


def run_centered_pca(role_vectors: dict[str, torch.Tensor], role_names: list[str]) -> dict[str, Any]:
    import torch

    matrix = torch.stack([role_vectors[name] for name in role_names], dim=0)
    mean_vector = matrix.mean(dim=0)
    centered = matrix - mean_vector
    _, singular_values, right_vectors = torch.linalg.svd(centered, full_matrices=False)
    variance = singular_values.square()
    explained_ratio = (variance / variance.sum()).tolist() if float(variance.sum()) else [0.0] * len(variance)
    components = right_vectors
    projections = centered @ components.T
    return {
        "mean_vector": mean_vector,
        "centered": centered,
        "components": components,
        "explained_variance_ratio": explained_ratio,
        "projections": projections,
    }


def save_scatter_plot(
    output_path: Path,
    coordinates: dict[str, tuple[float, float]],
    x_label: str,
    y_label: str,
    role_metadata: dict[str, dict[str, Any]] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    class_colors = {
        "deceptive": "#c44e52",
        "non_deceptive": "#4c72b0",
        "neutral": "#7f7f7f",
        "other": "#55a868",
    }
    seen_classes: set[str] = set()
    for label, (x_coord, y_coord) in coordinates.items():
        metadata = (role_metadata or {}).get(label, {})
        raw_class = str(metadata.get("goal_side", metadata.get("class", "neutral" if label == "default" else "other")))
        normalized_class = raw_class.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized_class in {"honest", "candid", "transparent"}:
            normalized_class = "non_deceptive"
        if normalized_class not in class_colors:
            normalized_class = "other"
        legend_label = normalized_class.replace("_", " ")
        if normalized_class in seen_classes:
            legend_label = "_nolegend_"
        else:
            seen_classes.add(normalized_class)
        display_label = str(metadata.get("plot_label", label))
        ax.scatter(
            x_coord,
            y_coord,
            color=class_colors[normalized_class],
            s=52,
            edgecolors="black",
            linewidths=0.5,
            label=legend_label,
            zorder=3,
        )
        ax.annotate(display_label, (x_coord, y_coord), xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.axvline(0.0, color="gray", linewidth=0.8)
    ax.grid(alpha=0.18, linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    legend_labels = [value for value in labels if value != "_nolegend_"]
    if legend_labels:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=False,
            title="Class",
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    else:
        fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_results_json(path: Path, payload: dict[str, Any]) -> None:
    import torch

    serializable = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            serializable[key] = value.tolist()
        else:
            serializable[key] = value
    write_json(path, serializable)


def _safe_projection(projections: list[float], index: int) -> float | None:
    if index >= len(projections):
        return None
    return float(projections[index])


def _safe_abs_projection(projections: list[float], index: int) -> float | None:
    value = _safe_projection(projections, index)
    return None if value is None else abs(value)


def _rank_descending_abs(values: list[float]) -> dict[int, int]:
    ordered = sorted(range(len(values)), key=lambda idx: (-abs(values[idx]), idx))
    return {pc_index: rank + 1 for rank, pc_index in enumerate(ordered)}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _final_layer_coordinates(
    layers: list[dict[str, Any]],
    anchor_role: str,
    role_names: list[str],
    x_index: int,
    y_index: int,
) -> dict[str, tuple[float, float]]:
    if not layers:
        return {}

    last_layer = layers[-1]
    anchor_projection = list(last_layer["anchor_projection"])
    if len(anchor_projection) <= max(x_index, y_index):
        return {}

    coordinates: dict[str, tuple[float, float]] = {}
    for role in role_names:
        role_projection = list(last_layer["role_projections"][role])
        if len(role_projection) <= max(x_index, y_index):
            continue
        coordinates[role] = (float(role_projection[x_index]), float(role_projection[y_index]))
    coordinates[anchor_role] = (float(anchor_projection[x_index]), float(anchor_projection[y_index]))
    return coordinates


def _layer_coordinates(
    layers: list[dict[str, Any]],
    anchor_role: str,
    role_names: list[str],
    layer_index: int,
    x_index: int,
    y_index: int,
) -> dict[str, tuple[float, float]]:
    layer_entry = next((entry for entry in layers if int(entry.get("layer_number", int(entry["layer"]) + 1)) == layer_index), None)
    if layer_entry is None:
        return {}

    anchor_projection = list(layer_entry["anchor_projection"])
    if len(anchor_projection) <= max(x_index, y_index):
        return {}

    coordinates: dict[str, tuple[float, float]] = {}
    for role in role_names:
        role_projection = list(layer_entry["role_projections"][role])
        if len(role_projection) <= max(x_index, y_index):
            continue
        coordinates[role] = (float(role_projection[x_index]), float(role_projection[y_index]))
    coordinates[anchor_role] = (float(anchor_projection[x_index]), float(anchor_projection[y_index]))
    return coordinates


def _save_heatmap(
    output_path: Path,
    matrix: list[list[float]],
    x_labels: list[str],
    y_labels: list[str],
    title: str,
    colorbar_label: str,
    *,
    signed: bool,
) -> None:
    import matplotlib.pyplot as plt

    if not matrix:
        return

    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(max(6, len(x_labels) * 0.75), max(4, len(y_labels) * 0.35)))
    if signed:
        max_abs = max(abs(value) for row in matrix for value in row) if matrix and matrix[0] else 0.0
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
    else:
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(x_labels)), x_labels)
    ax.set_yticks(range(len(y_labels)), y_labels)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_pca_report_artifacts(
    run_root: Path,
    layers: list[dict[str, Any]],
    anchor_role: str,
    role_names: list[str],
    role_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tables_dir = ensure_dir(run_root / "results" / "tables")
    plots_dir = ensure_dir(run_root / "results" / "plots")

    role_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    layer_summary_rows: list[dict[str, Any]] = []
    default_abs_matrix: list[list[float]] = []
    contrast_matrix: list[list[float]] = []
    max_pc_count = 0

    for layer_entry in layers:
        layer_index = int(layer_entry["layer"])
        layer_number = int(layer_entry.get("layer_number", layer_index + 1))
        layer_label = str(layer_entry.get("layer_label", f"L{layer_number}"))
        anchor_projection = [float(value) for value in layer_entry["anchor_projection"]]
        contrast_pc_cosines = [float(value) for value in layer_entry["contrast_pc_cosines"]]
        explained = [float(value) for value in layer_entry["explained_variance_ratio"]]
        max_pc_count = max(max_pc_count, len(anchor_projection))

        all_roles = list(role_names) + [anchor_role]
        projection_lookup = {
            role_name: [float(value) for value in layer_entry["role_projections"][role_name]]
            for role_name in role_names
        }
        projection_lookup[anchor_role] = anchor_projection

        for role_name in all_roles:
            projections = projection_lookup[role_name]
            metadata = (role_metadata or {}).get(role_name, {})
            role_rows.append(
                {
                    "layer": layer_index,
                    "layer_number": layer_number,
                    "layer_label": layer_label,
                    "role": role_name,
                    "plot_label": metadata.get("plot_label", role_name),
                    "role_class": metadata.get("goal_side", metadata.get("class")),
                    "category": metadata.get("category"),
                    "is_anchor": role_name == anchor_role,
                    "pc1_projection": _safe_projection(projections, 0),
                    "pc2_projection": _safe_projection(projections, 1),
                    "pc3_projection": _safe_projection(projections, 2),
                    "pc1_abs_projection": _safe_abs_projection(projections, 0),
                    "pc2_abs_projection": _safe_abs_projection(projections, 1),
                    "pc3_abs_projection": _safe_abs_projection(projections, 2),
                }
            )

        default_ranks = _rank_descending_abs(anchor_projection)
        contrast_ranks = _rank_descending_abs(contrast_pc_cosines)
        for pc_index, default_projection in enumerate(anchor_projection):
            ranking_rows.append(
                {
                    "layer": layer_index,
                    "layer_number": layer_number,
                    "layer_label": layer_label,
                    "pc_index": pc_index + 1,
                    "explained_variance_ratio": explained[pc_index] if pc_index < len(explained) else None,
                    "default_projection": default_projection,
                    "default_abs_projection": abs(default_projection),
                    "default_abs_rank": default_ranks[pc_index],
                    "contrast_cosine": contrast_pc_cosines[pc_index],
                    "contrast_abs_cosine": abs(contrast_pc_cosines[pc_index]),
                    "contrast_abs_rank": contrast_ranks[pc_index],
                }
            )

        top_default_pc = min(default_ranks, key=default_ranks.get) + 1 if default_ranks else None
        top_contrast_pc = min(contrast_ranks, key=contrast_ranks.get) + 1 if contrast_ranks else None
        layer_summary_rows.append(
            {
                "layer": layer_index,
                "layer_number": layer_number,
                "layer_label": layer_label,
                "top_pc_by_default_abs": top_default_pc,
                "top_pc_by_contrast_abs_cosine": top_contrast_pc,
                "default_pc1_abs_projection": abs(anchor_projection[0]) if anchor_projection else None,
                "contrast_pc1_cosine": contrast_pc_cosines[0] if contrast_pc_cosines else None,
            }
        )
        default_abs_matrix.append([abs(value) for value in anchor_projection])
        contrast_matrix.append(contrast_pc_cosines)

    _write_csv(
        tables_dir / "role_pc_projections.csv",
        [
            "layer",
            "layer_number",
            "layer_label",
            "role",
            "plot_label",
            "role_class",
            "category",
            "is_anchor",
            "pc1_projection",
            "pc2_projection",
            "pc3_projection",
            "pc1_abs_projection",
            "pc2_abs_projection",
            "pc3_abs_projection",
        ],
        role_rows,
    )
    _write_csv(
        tables_dir / "pc_rankings.csv",
        [
            "layer",
            "layer_number",
            "layer_label",
            "pc_index",
            "explained_variance_ratio",
            "default_projection",
            "default_abs_projection",
            "default_abs_rank",
            "contrast_cosine",
            "contrast_abs_cosine",
            "contrast_abs_rank",
        ],
        ranking_rows,
    )
    _write_csv(
        tables_dir / "layer_summary.csv",
        [
            "layer",
            "layer_number",
            "layer_label",
            "top_pc_by_default_abs",
            "top_pc_by_contrast_abs_cosine",
            "default_pc1_abs_projection",
            "contrast_pc1_cosine",
        ],
        layer_summary_rows,
    )

    final_pair_paths: dict[str, str] = {}
    selected_layer_pair_paths: dict[str, dict[str, str]] = {
        "pc1_pc2": {},
        "pc1_pc3": {},
        "pc2_pc3": {},
    }
    for x_index, y_index, pair_name, x_label, y_label in (
        (0, 1, "pc1_pc2", "PC1", "PC2"),
        (0, 2, "pc1_pc3", "PC1", "PC3"),
        (1, 2, "pc2_pc3", "PC2", "PC3"),
    ):
        final_coordinates = _final_layer_coordinates(layers, anchor_role, role_names, x_index, y_index)
        if final_coordinates:
            final_path = plots_dir / f"{pair_name}.png"
            save_scatter_plot(final_path, final_coordinates, x_label, y_label, role_metadata=role_metadata)
            final_pair_paths[pair_name] = str(final_path)

        for selected_layer in (7, 14, 21, 28):
            selected_coordinates = _layer_coordinates(
                layers,
                anchor_role,
                role_names,
                selected_layer,
                x_index,
                y_index,
            )
            if not selected_coordinates:
                continue
            output_path = plots_dir / f"{pair_name}__L{selected_layer}.png"
            save_scatter_plot(output_path, selected_coordinates, x_label, y_label, role_metadata=role_metadata)
            selected_layer_pair_paths[pair_name][f"L{selected_layer}"] = str(output_path)

    if max_pc_count:
        pc_labels = [f"PC{index + 1}" for index in range(max_pc_count)]
        default_heatmap = [
            [row[index] if index < len(row) else 0.0 for index in range(max_pc_count)]
            for row in default_abs_matrix
        ]
        contrast_heatmap = [
            [row[index] if index < len(row) else 0.0 for index in range(max_pc_count)]
            for row in contrast_matrix
        ]
        layer_labels = [
            str(layer_entry.get("layer_label", f"L{int(layer_entry.get('layer_number', int(layer_entry['layer']) + 1))}"))
            for layer_entry in layers
        ]
        _save_heatmap(
            plots_dir / "default_abs_projection_heatmap.png",
            default_heatmap,
            pc_labels,
            layer_labels,
            "Default Projection Magnitudes by Layer",
            "|projection|",
            signed=False,
        )
        _save_heatmap(
            plots_dir / "contrast_cosine_heatmap.png",
            contrast_heatmap,
            pc_labels,
            layer_labels,
            "Contrast Cosines by Layer",
            "cosine",
            signed=True,
        )

    return {
        "role_projection_table": str(tables_dir / "role_pc_projections.csv"),
        "pc_rankings_table": str(tables_dir / "pc_rankings.csv"),
        "layer_summary_table": str(tables_dir / "layer_summary.csv"),
        "plots": {
            "pc1_pc2": final_pair_paths.get("pc1_pc2", str(plots_dir / "pc1_pc2.png")),
            "pc1_pc3": final_pair_paths.get("pc1_pc3", str(plots_dir / "pc1_pc3.png")),
            "pc2_pc3": final_pair_paths.get("pc2_pc3", str(plots_dir / "pc2_pc3.png")),
            "pc1_pc2_selected_layers": selected_layer_pair_paths["pc1_pc2"],
            "pc1_pc3_selected_layers": selected_layer_pair_paths["pc1_pc3"],
            "pc2_pc3_selected_layers": selected_layer_pair_paths["pc2_pc3"],
            "default_abs_projection_heatmap": str(plots_dir / "default_abs_projection_heatmap.png"),
            "contrast_cosine_heatmap": str(plots_dir / "contrast_cosine_heatmap.png"),
        },
    }
