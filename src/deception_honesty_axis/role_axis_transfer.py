from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from deception_honesty_axis.analysis import cosine_similarity, run_centered_pca
from deception_honesty_axis.common import append_jsonl, ensure_dir, load_jsonl, write_json


ZERO_SHOT_SOURCE = "__zero_shot__"
ZERO_SHOT_METHODS = {"contrast_zero_shot", "pc1_zero_shot"}
TRAINED_METHODS = {"contrast_logistic", "pc123_linear"}
SUPPORTED_METHODS = ZERO_SHOT_METHODS | TRAINED_METHODS
DATASET_LABELS = {
    "Deception-AILiar-completion": "AILiar",
    "Deception-ClaimsDefinitional-completion": "ClaimsDef",
    "Deception-ClaimsEvidential-completion": "ClaimsEvid",
    "Deception-ClaimsFictional-completion": "ClaimsFict",
    "Deception-ConvincingGame-completion": "ConvGame",
    "Deception-HarmPressureChoice-completion": "HarmChoice",
    "Deception-InstructedDeception-completion": "InstrDec",
    "Deception-InsiderTrading-SallyConcat-completion": "Insider",
    "Deception-Mask-completion": "Mask",
    "Deception-Roleplaying-completion": "Roleplay",
    ZERO_SHOT_SOURCE: "ZeroShot",
}


@dataclass(frozen=True)
class ResolvedLayerSpec:
    key: str
    label: str
    layer_indices: tuple[int, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "layer_spec": self.key,
            "layer_label": self.label,
            "layer_indices": list(self.layer_indices),
        }


@dataclass(frozen=True)
class CachedSplitFeatures:
    dataset: str
    split: str
    sample_ids: list[str]
    labels: np.ndarray
    features_by_spec: dict[str, np.ndarray]


def resolve_layer_specs(layer_count: int, raw_specs: list[str]) -> list[ResolvedLayerSpec]:
    resolved: list[ResolvedLayerSpec] = []
    seen: set[str] = set()
    for raw_spec in raw_specs:
        token = str(raw_spec).strip().lower()
        if token in seen:
            continue
        if token == "last4_mean":
            if layer_count < 4:
                raise ValueError(f"Need at least 4 layers for last4_mean, found {layer_count}")
            resolved.append(
                ResolvedLayerSpec(
                    key="last4_mean",
                    label="Last4Mean",
                    layer_indices=tuple(range(layer_count - 4, layer_count)),
                )
            )
            seen.add(token)
            continue

        if not token.isdigit():
            raise ValueError(f"Unsupported layer spec: {raw_spec}")
        layer_number = int(token)
        if layer_number < 1 or layer_number > layer_count:
            raise ValueError(f"Layer spec {layer_number} is out of range for {layer_count} layers")
        zero_index = layer_number - 1
        resolved.append(
            ResolvedLayerSpec(
                key=str(layer_number),
                label=f"L{layer_number}",
                layer_indices=(zero_index,),
            )
        )
        seen.add(token)
    if not resolved:
        raise ValueError("At least one layer spec is required")
    return resolved


def role_vector_for_spec(vector, spec: ResolvedLayerSpec):  # noqa: ANN001, ANN201
    import torch

    indices = list(spec.layer_indices)
    selected = vector[indices].to(torch.float32)
    if selected.ndim == 1:
        return selected
    if selected.shape[0] == 1:
        return selected[0]
    return selected.mean(dim=0)


def _normalize(vector):  # noqa: ANN001, ANN201
    import torch

    norm = torch.linalg.norm(vector)
    if float(norm) == 0.0:
        return vector.to(torch.float32)
    return (vector / norm).to(torch.float32)


def _top_components(components, count: int = 3):  # noqa: ANN001, ANN201
    import torch

    return components[: min(count, int(components.shape[0]))].to(torch.float32)


def build_role_axis_bundle(
    role_vectors: dict[str, Any],
    honest_roles: list[str],
    deceptive_roles: list[str],
    anchor_role: str,
    layer_specs: list[str],
) -> dict[str, Any]:
    import torch

    if anchor_role not in role_vectors:
        raise KeyError(f"Missing anchor role '{anchor_role}' in role vectors")
    for role_name in honest_roles + deceptive_roles:
        if role_name not in role_vectors:
            raise KeyError(f"Missing role '{role_name}' in role vectors")

    comparison_roles = list(deceptive_roles) + list(honest_roles)
    layer_count = int(next(iter(role_vectors.values())).shape[0])
    resolved_specs = resolve_layer_specs(layer_count, layer_specs)
    bundle_layers: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []

    for spec in resolved_specs:
        layer_role_vectors = {
            role_name: role_vector_for_spec(role_vectors[role_name], spec)
            for role_name in [anchor_role] + comparison_roles
        }
        honest_mean = torch.stack([layer_role_vectors[name] for name in honest_roles], dim=0).mean(dim=0)
        deceptive_mean = torch.stack([layer_role_vectors[name] for name in deceptive_roles], dim=0).mean(dim=0)
        contrast_axis = _normalize(honest_mean - deceptive_mean)
        pca = run_centered_pca(layer_role_vectors, comparison_roles)
        components = _top_components(pca["components"])
        pc_signs = [1.0] * int(components.shape[0])
        if components.shape[0] > 0 and cosine_similarity(components[0], contrast_axis) < 0.0:
            components[0] = -components[0]
            pc_signs[0] = -1.0

        centered_anchor = layer_role_vectors[anchor_role] - pca["mean_vector"]
        anchor_projection = centered_anchor @ components.T
        contrast_pc_cosines = [cosine_similarity(contrast_axis, component) for component in components]
        role_projections = {
            role_name: ((layer_role_vectors[role_name] - pca["mean_vector"]) @ components.T)
            for role_name in comparison_roles
        }

        bundle_layers[spec.key] = {
            "layer_spec": spec.key,
            "layer_label": spec.label,
            "layer_indices": list(spec.layer_indices),
            "contrast_axis": contrast_axis,
            "honest_mean": honest_mean.to(torch.float32),
            "deceptive_mean": deceptive_mean.to(torch.float32),
            "pca_mean": pca["mean_vector"].to(torch.float32),
            "pc_components": components,
            "pc_signs": pc_signs,
            "explained_variance_ratio": pca["explained_variance_ratio"][: int(components.shape[0])],
            "anchor_projection": anchor_projection.to(torch.float32),
            "contrast_pc_cosines": contrast_pc_cosines,
            "role_projections": {
                role_name: projection.to(torch.float32)
                for role_name, projection in role_projections.items()
            },
        }
        summary_rows.append(
            {
                "layer_spec": spec.key,
                "layer_label": spec.label,
                "layer_indices": ",".join(str(idx) for idx in spec.layer_indices),
                "pc1_explained_variance_ratio": pca["explained_variance_ratio"][0] if pca["explained_variance_ratio"] else None,
                "pc2_explained_variance_ratio": pca["explained_variance_ratio"][1] if len(pca["explained_variance_ratio"]) > 1 else None,
                "pc3_explained_variance_ratio": pca["explained_variance_ratio"][2] if len(pca["explained_variance_ratio"]) > 2 else None,
                "pc1_contrast_cosine": contrast_pc_cosines[0] if contrast_pc_cosines else None,
                "pc2_contrast_cosine": contrast_pc_cosines[1] if len(contrast_pc_cosines) > 1 else None,
                "pc3_contrast_cosine": contrast_pc_cosines[2] if len(contrast_pc_cosines) > 2 else None,
                "anchor_pc1_projection": float(anchor_projection[0]) if int(anchor_projection.shape[0]) > 0 else None,
                "anchor_pc2_projection": float(anchor_projection[1]) if int(anchor_projection.shape[0]) > 1 else None,
                "anchor_pc3_projection": float(anchor_projection[2]) if int(anchor_projection.shape[0]) > 2 else None,
            }
        )

    return {
        "anchor_role": anchor_role,
        "honest_roles": list(honest_roles),
        "deceptive_roles": list(deceptive_roles),
        "comparison_roles": comparison_roles,
        "layer_count": layer_count,
        "resolved_layer_specs": [spec.to_manifest() for spec in resolved_specs],
        "layers": bundle_layers,
        "summary_rows": summary_rows,
    }


def _tensor_to_serializable(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _tensor_to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tensor_to_serializable(item) for item in value]
    return value


def write_role_axis_bundle(run_root: Path, bundle: dict[str, Any]) -> dict[str, str]:
    import torch

    results_dir = ensure_dir(run_root / "results")
    bundle_path = results_dir / "axis_bundle.pt"
    torch.save(bundle, bundle_path)
    write_json(results_dir / "axis_bundle.json", _tensor_to_serializable(bundle))

    summary_path = results_dir / "layer_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "layer_spec",
            "layer_label",
            "layer_indices",
            "pc1_explained_variance_ratio",
            "pc2_explained_variance_ratio",
            "pc3_explained_variance_ratio",
            "pc1_contrast_cosine",
            "pc2_contrast_cosine",
            "pc3_contrast_cosine",
            "anchor_pc1_projection",
            "anchor_pc2_projection",
            "anchor_pc3_projection",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in bundle["summary_rows"]:
            writer.writerow(row)

    return {
        "axis_bundle": str(bundle_path),
        "bundle_json": str(results_dir / "axis_bundle.json"),
        "layer_summary": str(summary_path),
    }


def load_role_axis_bundle(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu")


def load_manifest_rows(split_dir: Path) -> dict[str, dict[str, Any]]:
    manifest_path = split_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.jsonl in {split_dir}")
    rows: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(manifest_path):
        rows[str(row["id"])] = row
    return rows


def _pool_completion_mean_for_spec(
    tensor,
    row: dict[str, Any],
    spec: ResolvedLayerSpec,
    *,
    assume_completion_only: bool = False,
) -> np.ndarray:  # noqa: ANN001
    import torch

    if tensor.dim() != 3:
        raise ValueError(f"Expected tensor shape (L,T,D), got {tuple(tensor.shape)}")
    if "user_end_idx" not in row or "completion_end_idx" not in row:
        raise ValueError("Missing completion boundary metadata in manifest row")

    start = int(row["user_end_idx"])
    end = int(row["completion_end_idx"])
    if end <= start:
        raise ValueError(f"Invalid completion bounds [{start}, {end}) for sample {row.get('id')}")
    tensor_len = int(tensor.shape[1])
    if end > tensor_len:
        original_length = int(row.get("generation_length") or 0)
        if assume_completion_only:
            start = 0
            end = tensor_len
        elif original_length <= 0:
            raise ValueError(
                f"Completion bound {end} exceeds token length {tensor_len} for sample {row.get('id')} "
                "and manifest generation_length is missing"
            )
        elif original_length < end:
            if assume_completion_only:
                start = 0
                end = tensor_len
            else:
                raise ValueError(
                    f"Completion bound {end} exceeds manifest generation_length {original_length} "
                    f"for sample {row.get('id')}"
                )
        else:
            start = int(np.floor(start / original_length * tensor_len))
            end = int(np.ceil(end / original_length * tensor_len))
            start = max(0, min(start, tensor_len - 1))
            end = max(start + 1, min(end, tensor_len))

    pooled_layers: list[torch.Tensor] = []
    for layer_index in spec.layer_indices:
        if layer_index >= int(tensor.shape[0]):
            raise ValueError(f"Layer index {layer_index} out of range for sample {row.get('id')}")
        pooled_layers.append(tensor[layer_index, start:end, :].mean(dim=0))
    pooled = torch.stack(pooled_layers, dim=0).mean(dim=0)
    return pooled.detach().cpu().numpy().astype(np.float32, copy=False)


def load_completion_mean_split(
    split_dir: Path,
    resolved_specs: list[ResolvedLayerSpec],
) -> CachedSplitFeatures:
    from safetensors.torch import load_file

    assume_completion_only = split_dir.parent.name.endswith("-completion")
    manifest_rows = load_manifest_rows(split_dir)
    shard_paths = sorted(split_dir.glob("shard_*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.safetensors found in {split_dir}")

    features_by_spec: dict[str, list[np.ndarray]] = {spec.key: [] for spec in resolved_specs}
    sample_ids: list[str] = []
    labels: list[int] = []

    for shard_path in shard_paths:
        shard = load_file(str(shard_path))
        for sample_id, tensor in shard.items():
            row = manifest_rows.get(str(sample_id))
            if row is None:
                continue
            label = int(row.get("label", -1))
            if label < 0:
                continue
            for spec in resolved_specs:
                features_by_spec[spec.key].append(
                    _pool_completion_mean_for_spec(
                        tensor,
                        row,
                        spec,
                        assume_completion_only=assume_completion_only,
                    )
                )
            sample_ids.append(str(sample_id))
            labels.append(label)

    if not labels:
        raise ValueError(f"No labeled samples loaded from {split_dir}")

    return CachedSplitFeatures(
        dataset=split_dir.parent.name,
        split=split_dir.name,
        sample_ids=sample_ids,
        labels=np.asarray(labels, dtype=np.int64),
        features_by_spec={
            spec.key: np.stack(features_by_spec[spec.key]).astype(np.float32)
            for spec in resolved_specs
        },
    )


def scores_for_layer(features: np.ndarray, layer_bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    contrast_axis = layer_bundle["contrast_axis"].detach().cpu().numpy().astype(np.float32)
    pca_mean = layer_bundle["pca_mean"].detach().cpu().numpy().astype(np.float32)
    pc_components = layer_bundle["pc_components"].detach().cpu().numpy().astype(np.float32)
    centered = features - pca_mean[None, :]
    return {
        "contrast": features @ contrast_axis,
        "pc_scores": centered @ pc_components.T,
    }


def compute_binary_metrics(
    labels: np.ndarray,
    deceptive_scores: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

    score_values = probabilities if probabilities is not None else deceptive_scores
    try:
        auroc = float(roc_auc_score(labels, score_values))
    except Exception:
        auroc = 0.5
    try:
        auprc = float(average_precision_score(labels, score_values))
    except Exception:
        auprc = 0.5
    return {
        "auroc": auroc,
        "auprc": auprc,
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "count": int(labels.shape[0]),
    }


def evaluate_zero_shot(
    labels: np.ndarray,
    honest_like_scores: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    deceptive_scores = -honest_like_scores
    predictions = (deceptive_scores > 0.0).astype(np.int64)
    probabilities = 1.0 / (1.0 + np.exp(-deceptive_scores))
    metrics = compute_binary_metrics(labels, deceptive_scores, predictions, probabilities=probabilities)
    return metrics, deceptive_scores, probabilities


def fit_logistic_scores(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    *,
    random_seed: int,
    max_iter: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(
        random_state=random_seed,
        max_iter=max_iter,
        solver="liblinear",
    )
    classifier.fit(train_features, train_labels)
    probabilities = classifier.predict_proba(eval_features)[:, 1]
    predictions = classifier.predict(eval_features)
    artifact = {
        "coef": classifier.coef_.reshape(-1).tolist(),
        "intercept": classifier.intercept_.reshape(-1).tolist(),
        "classes": classifier.classes_.tolist(),
    }
    return artifact, probabilities.astype(np.float32), predictions.astype(np.int64)


def read_completed_metric_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            (row["method"], row["layer_spec"], row["source_dataset"], row["target_dataset"])
            for row in rows
        }


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_summary_csv(path: Path, metric_rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    summary: dict[tuple[str, str], dict[str, Any]] = {}
    for row in metric_rows:
        key = (row["method"], row["layer_spec"])
        bucket = summary.setdefault(
            key,
            {
                "method": row["method"],
                "layer_spec": row["layer_spec"],
                "rows": 0,
                "diag_rows": 0,
                "cross_rows": 0,
                "auroc_sum": 0.0,
                "auprc_sum": 0.0,
                "diag_auroc_sum": 0.0,
                "cross_auroc_sum": 0.0,
            },
        )
        bucket["rows"] += 1
        bucket["auroc_sum"] += float(row["auroc"])
        bucket["auprc_sum"] += float(row["auprc"])
        if row["source_dataset"] == row["target_dataset"]:
            bucket["diag_rows"] += 1
            bucket["diag_auroc_sum"] += float(row["auroc"])
        elif row["source_dataset"] != ZERO_SHOT_SOURCE:
            bucket["cross_rows"] += 1
            bucket["cross_auroc_sum"] += float(row["auroc"])

    fieldnames = [
        "method",
        "layer_spec",
        "row_count",
        "mean_auroc",
        "mean_auprc",
        "diag_count",
        "diag_mean_auroc",
        "cross_count",
        "cross_mean_auroc",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for bucket in summary.values():
            writer.writerow(
                {
                    "method": bucket["method"],
                    "layer_spec": bucket["layer_spec"],
                    "row_count": bucket["rows"],
                    "mean_auroc": bucket["auroc_sum"] / bucket["rows"] if bucket["rows"] else None,
                    "mean_auprc": bucket["auprc_sum"] / bucket["rows"] if bucket["rows"] else None,
                    "diag_count": bucket["diag_rows"],
                    "diag_mean_auroc": (
                        bucket["diag_auroc_sum"] / bucket["diag_rows"] if bucket["diag_rows"] else None
                    ),
                    "cross_count": bucket["cross_rows"],
                    "cross_mean_auroc": (
                        bucket["cross_auroc_sum"] / bucket["cross_rows"] if bucket["cross_rows"] else None
                    ),
                }
            )


def write_fit_summary_csv(path: Path, fit_artifacts_path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "method",
        "layer_spec",
        "source_dataset",
        "target_dataset",
        "coef",
        "intercept",
        "coef_l2_norm",
        "coef_dim",
    ]
    if not fit_artifacts_path.exists():
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
        return

    rows: list[dict[str, Any]] = []
    with fit_artifacts_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            coef = np.asarray(record.get("coef", []), dtype=np.float32)
            rows.append(
                {
                    "method": record.get("method"),
                    "layer_spec": record.get("layer_spec"),
                    "source_dataset": record.get("source_dataset"),
                    "target_dataset": record.get("target_dataset"),
                    "coef": json.dumps([float(value) for value in coef.tolist()]),
                    "intercept": json.dumps(record.get("intercept", [])),
                    "coef_l2_norm": float(np.linalg.norm(coef)) if coef.size else 0.0,
                    "coef_dim": int(coef.size),
                }
            )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_metric_heatmaps(
    output_dir: Path,
    metric_rows: list[dict[str, Any]],
    methods: list[str],
    layer_specs: list[str],
    datasets: list[str],
    *,
    metrics: tuple[str, ...] = ("auroc",),
) -> list[str]:
    import matplotlib.pyplot as plt

    ensure_dir(output_dir)
    output_paths: list[str] = []
    target_index = {name: idx for idx, name in enumerate(datasets)}

    for method in methods:
        for layer_spec in layer_specs:
            rows = [row for row in metric_rows if row["method"] == method and row["layer_spec"] == layer_spec]
            if not rows:
                continue

            seen_sources = {row["source_dataset"] for row in rows}
            sources = [ZERO_SHOT_SOURCE] if ZERO_SHOT_SOURCE in seen_sources else []
            sources.extend([name for name in datasets if name in seen_sources])
            source_index = {name: idx for idx, name in enumerate(sources)}

            for metric_name in metrics:
                matrix = np.full((len(sources), len(datasets)), np.nan, dtype=np.float32)
                for row in rows:
                    value = row.get(metric_name)
                    if value in (None, ""):
                        continue
                    matrix[source_index[row["source_dataset"]], target_index[row["target_dataset"]]] = float(value)

                fig, ax = plt.subplots(figsize=(max(12, len(datasets) * 2.0), max(5, len(sources) * 1.3)))
                image = ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto", cmap="viridis")
                ax.set_xticks(
                    range(len(datasets)),
                    [DATASET_LABELS.get(name, name) for name in datasets],
                    rotation=35,
                    ha="right",
                    fontsize=12,
                )
                ax.set_yticks(
                    range(len(sources)),
                    [DATASET_LABELS.get(name, name) for name in sources],
                    fontsize=12,
                )
                ax.set_title(f"{metric_name.replace('_', ' ').title()} heatmap: {method} @ {layer_spec}")
                for row_index in range(matrix.shape[0]):
                    for col_index in range(matrix.shape[1]):
                        value = matrix[row_index, col_index]
                        if np.isnan(value):
                            continue
                        ax.text(
                            col_index,
                            row_index,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            fontsize=11,
                            color="white" if value < 0.55 else "black",
                            fontweight="bold",
                        )
                ax.set_xticks(np.arange(-0.5, len(datasets), 1), minor=True)
                ax.set_yticks(np.arange(-0.5, len(sources), 1), minor=True)
                ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
                ax.tick_params(which="minor", bottom=False, left=False)
                fig.colorbar(
                    image,
                    ax=ax,
                    label=metric_name.replace("_", " ").title(),
                    fraction=0.046,
                    pad=0.04,
                )
                fig.tight_layout()
                path = output_dir / f"{method}__{layer_spec}__{metric_name}_heatmap.png"
                fig.savefig(path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                output_paths.append(str(path))

    return output_paths


def save_transfer_lineplots(
    output_dir: Path,
    metric_rows: list[dict[str, Any]],
    layer_specs: list[str],
    datasets: list[str],
    *,
    metric_name: str = "auroc",
) -> list[str]:
    import matplotlib.pyplot as plt

    ensure_dir(output_dir)
    rows = [
        row
        for row in metric_rows
        if row["method"] in {"contrast_zero_shot", "pc1_zero_shot", "pc123_linear"}
    ]
    if not rows:
        return []

    layer_order = {layer_spec: idx for idx, layer_spec in enumerate(layer_specs)}
    method_labels = {
        "contrast_zero_shot": "Contrast Zero-Shot",
        "pc1_zero_shot": "PC1 Zero-Shot",
        "pc123_linear": "PC123 Mean Cross-Source",
    }
    method_colors = {
        "contrast_zero_shot": "#0b7285",
        "pc1_zero_shot": "#e67700",
        "pc123_linear": "#c2255c",
    }
    plot_records: dict[str, dict[str, dict[str, float]]] = {
        dataset: {
            "contrast_zero_shot": {},
            "pc1_zero_shot": {},
            "pc123_linear": {},
        }
        for dataset in datasets
    }

    for dataset in datasets:
        zero_shot_rows = [
            row
            for row in rows
            if row["target_dataset"] == dataset and row["source_dataset"] == ZERO_SHOT_SOURCE
        ]
        for row in zero_shot_rows:
            plot_records[dataset][row["method"]][row["layer_spec"]] = float(row[metric_name])

        cross_rows = [
            row
            for row in rows
            if row["method"] == "pc123_linear"
            and row["target_dataset"] == dataset
            and row["source_dataset"] not in {ZERO_SHOT_SOURCE, dataset}
        ]
        by_layer: dict[str, list[float]] = {}
        for row in cross_rows:
            by_layer.setdefault(row["layer_spec"], []).append(float(row[metric_name]))
        for layer_spec, values in by_layer.items():
            if values:
                plot_records[dataset]["pc123_linear"][layer_spec] = float(np.mean(values))

    figure_paths: list[str] = []
    ncols = 2
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15, max(4.5 * nrows, 8)),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).reshape(-1)

    for ax in axes_array:
        ax.set_facecolor("#fbfbfd")
        ax.grid(True, axis="y", color="#dfe3e6", linewidth=0.8, alpha=0.9)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#adb5bd")
        ax.spines["bottom"].set_color("#adb5bd")
        ax.set_ylim(0.0, 1.0)

    for ax, dataset in zip(axes_array, datasets, strict=False):
        for method in ("contrast_zero_shot", "pc1_zero_shot", "pc123_linear"):
            series = plot_records[dataset][method]
            x_values = [layer_order[layer_spec] for layer_spec in layer_specs if layer_spec in series]
            y_values = [series[layer_spec] for layer_spec in layer_specs if layer_spec in series]
            if not x_values:
                continue
            ax.plot(
                x_values,
                y_values,
                color=method_colors[method],
                linewidth=2.4,
                marker="o",
                markersize=5.5,
                label=method_labels[method],
            )
        ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=14, fontweight="bold", color="#1f2933")
        ax.set_xticks(range(len(layer_specs)), layer_specs, fontsize=11)
        ax.set_ylabel(metric_name.upper(), fontsize=11, color="#495057")

    for ax in axes_array[len(datasets):]:
        ax.axis("off")

    handles, labels = axes_array[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            fontsize=12,
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.suptitle(
        f"Role-Axis Transfer Across Layers ({metric_name.upper()})",
        fontsize=20,
        fontweight="bold",
        color="#111827",
    )

    path = output_dir / f"role_axis_transfer_by_layer__{metric_name}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    figure_paths.append(str(path))
    return figure_paths


def save_fit_artifact(path: Path, record: dict[str, Any]) -> None:
    append_jsonl(path, [record])


def save_score_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    append_jsonl(path, rows)
