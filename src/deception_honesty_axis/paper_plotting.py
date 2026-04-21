from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable

from deception_honesty_axis.common import slugify


DECEPTION_AXIS_ALIASES = {
    "deception-axis",
    "deception-axis-v2",
    "quantity-axis-v2",
    "quantity-v2",
    "quantity_axis_v2",
    "quantity-v2-cumulative-pc-sweep",
    "quantity-axis-v2-cumulative-pc-sweep",
}
SYCOPHANCY_AXIS_ALIASES = {
    "sycophancy-axis",
    "sycophancy-pilot-v1",
    "sycophancy-pilot-v2",
    "sycophancy_pilot_v1",
    "sycophancy_pilot_v2",
    "sycophancy-pilot-v1-cumulative-pc-sweep",
    "sycophancy-pilot-v2-cumulative-pc-sweep",
}
AXIS_DISPLAY_NAMES = {
    "deception_axis": "Deception Axis",
    "sycophancy_axis": "Sycophancy Axis",
}

DECEPTION_DATASET_ORDER = [
    "Deception-AILiar-completion",
    "Deception-ConvincingGame-completion",
    "Deception-HarmPressureChoice-completion",
    "Deception-InstructedDeception-completion",
    "Deception-Mask-completion",
    "Deception-Roleplaying-completion",
    "Deception-ClaimsDefinitional-completion",
    "Deception-ClaimsEvidential-completion",
    "Deception-ClaimsFictional-completion",
]
SYCOPHANCY_DATASET_ORDER = [
    "Sycophancy Dataset",
    "Open-Ended Sycophancy",
    "OEQ Validation",
    "OEQ Indirectness",
    "OEQ Framing",
]
DATASET_LABEL_OVERRIDES = {
    "Deception-AILiar-completion": "AI Liar",
    "Deception-ClaimsDefinitional-completion": "Claims Definitional",
    "Deception-ClaimsEvidential-completion": "Claims Evidential",
    "Deception-ClaimsFictional-completion": "Claims Fictional",
    "Deception-ConvincingGame-completion": "Convincing Game",
    "Deception-HarmPressureChoice-completion": "Harm Pressure Choice",
    "Deception-InstructedDeception-completion": "Instructed Deception",
    "Deception-Mask-completion": "Mask",
    "Deception-Roleplaying-completion": "Roleplaying",
    "Open-Ended Sycophancy": "Open-Ended Sycophancy",
    "OEQ Validation": "OEQ Validation",
    "OEQ Indirectness": "OEQ Indirectness",
    "OEQ Framing": "OEQ Framing",
    "Sycophancy Dataset": "Sycophancy Dataset",
}

ZERO_SHOT_METHOD_ORDER = [
    "contrast_zero_shot",
    "pc1_zero_shot",
    "pc2_zero_shot",
    "pc3_zero_shot",
]
ZERO_SHOT_METHOD_LABELS = {
    "contrast_zero_shot": "Contrast",
    "pc1_zero_shot": "PC1",
    "pc2_zero_shot": "PC2",
    "pc3_zero_shot": "PC3",
}
ZERO_SHOT_METHOD_COLORS = {
    "contrast_zero_shot": "#0f766e",
    "pc1_zero_shot": "#2563eb",
    "pc2_zero_shot": "#7c3aed",
    "pc3_zero_shot": "#dc2626",
}

TRANSFER_SERIES_ORDER = ["baseline", "pc1", "pc2", "pc3"]
TRANSFER_SERIES_LABELS = {
    "baseline": "Baseline",
    "pc1": "PC1 Projection",
    "pc2": "PC2 Projection",
    "pc3": "PC3 Projection",
}
TRANSFER_SERIES_COLORS = {
    "baseline": "#475569",
    "pc1": "#2563eb",
    "pc2": "#7c3aed",
    "pc3": "#dc2626",
}

HF_ARTIFACT_FILE_PATTERNS = {
    "axis_bundle": ("runs/role-axis-bundles/", "/results/axis_bundle.pt"),
    "zero_shot_eval": ("runs/behavior-axis-evaluation/", "/results/pairwise_metrics.csv"),
    "baseline_transfer": ("runs/activation-row-transfer/", "/results/pairwise_metrics.csv"),
    "pc_projection_transfer": ("runs/activation-row-transfer-pc-projection/", "/results/pairwise_metrics.csv"),
}
PAPER_FIGURE_FORMATS = ("png", "pdf")


def normalized_token(text: str) -> str:
    return slugify(text).replace("_", "-")


def canonical_axis_key(raw_name: str) -> str:
    normalized = normalized_token(raw_name)
    if normalized in DECEPTION_AXIS_ALIASES or "quantity-axis-v2" in normalized:
        return "deception_axis"
    if normalized in SYCOPHANCY_AXIS_ALIASES or "sycophancy-pilot" in normalized:
        return "sycophancy_axis"
    if "quantity" in normalized and "axis" in normalized:
        return "deception_axis"
    if "sycophancy" in normalized:
        return "sycophancy_axis"
    return normalized


def canonical_axis_display_name(raw_name: str) -> str:
    axis_key = canonical_axis_key(raw_name)
    if axis_key in AXIS_DISPLAY_NAMES:
        return AXIS_DISPLAY_NAMES[axis_key]
    return title_case_slug(raw_name)


def behavior_for_axis(raw_name: str) -> str | None:
    axis_key = canonical_axis_key(raw_name)
    if axis_key == "deception_axis":
        return "deception"
    if axis_key == "sycophancy_axis":
        return "sycophancy"
    return None


def title_case_slug(text: str) -> str:
    cleaned = re.sub(r"[-_]+", " ", str(text)).strip()
    if not cleaned:
        return "Value"
    return " ".join(part.upper() if part.isupper() else part.capitalize() for part in cleaned.split())


def canonical_dataset_label(dataset_name: str) -> str:
    if dataset_name in DATASET_LABEL_OVERRIDES:
        return DATASET_LABEL_OVERRIDES[dataset_name]
    without_suffix = re.sub(r"-completion$", "", dataset_name)
    return title_case_slug(without_suffix)


def dataset_order_for_behavior(behavior_name: str | None, observed: Iterable[str]) -> list[str]:
    observed_list = list(observed)
    if behavior_name == "deception":
        preferred = DECEPTION_DATASET_ORDER
    elif behavior_name == "sycophancy":
        preferred = SYCOPHANCY_DATASET_ORDER
    else:
        preferred = []
    ordered = [dataset for dataset in preferred if dataset in observed_list]
    ordered.extend(dataset for dataset in observed_list if dataset not in ordered)
    return ordered


def canonical_role_label(role_name: str) -> str:
    if role_name == "default":
        return "Default"
    return title_case_slug(role_name)


def canonical_role_side(role_row: dict[str, Any]) -> str | None:
    raw_side = (
        role_row.get("role_side")
        or role_row.get("side")
        or role_row.get("goal_side")
        or role_row.get("class")
    )
    if raw_side in (None, ""):
        if role_row.get("anchor_only"):
            return "anchor"
        return None
    normalized = normalized_token(str(raw_side))
    if normalized in {"anchor"}:
        return "anchor"
    if normalized in {"deceptive", "sycophantic", "harmful"}:
        return "harmful"
    if normalized in {"honest", "non-sycophantic", "non-sycophantic", "safe", "harmless"}:
        return "harmless"
    return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_pc_count_from_path(path_value: str) -> int | None:
    match = re.search(r"(?:^|[-_/])pcs[-_](\d+)(?:[-_/]|$)", path_value)
    if match is None:
        return None
    return int(match.group(1))


def infer_artifact_kind(path_value: str) -> str | None:
    normalized = path_value.replace("\\", "/")
    for kind, (required_prefix, required_suffix) in HF_ARTIFACT_FILE_PATTERNS.items():
        if required_prefix in normalized and normalized.endswith(required_suffix):
            return kind
    return None


def infer_axis_key_from_path(path_value: str) -> str | None:
    normalized = normalized_token(path_value)
    if "quantity-axis-v2" in normalized:
        return "deception_axis"
    if "sycophancy-pilot-v1" in normalized or "sycophancy-pilot-v2" in normalized:
        return "sycophancy_axis"
    if "quantity" in normalized and "sycophancy" not in normalized:
        return "deception_axis"
    if "sycophancy" in normalized:
        return "sycophancy_axis"
    return None


def resolve_run_root_from_artifact_path(path_value: str, artifact_kind: str) -> str:
    normalized = path_value.replace("\\", "/")
    if artifact_kind == "axis_bundle":
        suffix = "/results/axis_bundle.pt"
    else:
        suffix = "/results/pairwise_metrics.csv"
    if normalized.endswith(suffix):
        return normalized[: -len(suffix)]
    return normalized


def figure_output_paths(output_path: Path, formats: tuple[str, ...] = PAPER_FIGURE_FORMATS) -> dict[str, Path]:
    base_name = output_path.stem
    return {
        fmt: output_path.parent / f"{base_name}.{fmt}"
        for fmt in formats
    }
