from __future__ import annotations

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
) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (x_coord, y_coord) in coordinates.items():
        ax.scatter(x_coord, y_coord)
        ax.text(x_coord, y_coord, label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.axvline(0.0, color="gray", linewidth=0.8)
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
