#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deception_honesty_axis.common import ensure_dir
from deception_honesty_axis.paper_plotting import canonical_role_label
from deception_honesty_axis.role_axis_transfer import load_role_axis_bundle


@dataclass(frozen=True)
class AxisSpec:
    slug: str
    display_name: str
    bundle_path: Path
    negative_label: str
    positive_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot role separation in PCA space from saved axis bundles. "
            "Each axis writes a CSV of role PC projections plus a PNG/PDF scatter plot."
        )
    )
    parser.add_argument(
        "--axis",
        action="append",
        required=True,
        help=(
            "Axis spec in the form "
            "'slug|display_name|axis_bundle_run_dir_or_axis_bundle_pt|negative_label|positive_label'. "
            "Repeat for multiple axes."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write CSV and scatter outputs into.",
    )
    parser.add_argument(
        "--layer-spec",
        default="14",
        help="Layer spec or layer number to use. Defaults to 14.",
    )
    parser.add_argument(
        "--x-pc",
        type=int,
        default=1,
        help="1-based principal component index for the x-axis. Defaults to 1.",
    )
    parser.add_argument(
        "--y-pc",
        type=int,
        default=2,
        help="1-based principal component index for the y-axis. Defaults to 2.",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Disable role-name annotations on the scatter plots.",
    )
    return parser.parse_args()


def parse_axis_spec(raw: str) -> AxisSpec:
    parts = [part.strip() for part in raw.split("|")]
    if len(parts) != 5:
        raise ValueError(
            f"Invalid --axis spec {raw!r}. Expected "
            "'slug|display_name|axis_bundle_run_dir_or_axis_bundle_pt|negative_label|positive_label'."
        )
    slug, display_name, bundle_raw, negative_label, positive_label = parts
    return AxisSpec(
        slug=slug,
        display_name=display_name,
        bundle_path=resolve_bundle_path(Path(bundle_raw)),
        negative_label=negative_label,
        positive_label=positive_label,
    )


def resolve_bundle_path(path: Path) -> Path:
    if path.is_file():
        return path.resolve()
    direct = (path / "axis_bundle.pt").resolve()
    if direct.exists():
        return direct
    nested = (path / "results" / "axis_bundle.pt").resolve()
    if nested.exists():
        return nested
    raise FileNotFoundError(f"Could not find axis_bundle.pt at {path}")


def choose_layer_spec(axis_bundle: dict[str, Any], requested: str | None) -> str:
    layer_specs = list(axis_bundle["layers"].keys())
    if requested is None or requested == "":
        if len(layer_specs) != 1:
            raise ValueError(f"Axis bundle has multiple layers {layer_specs!r}; specify --layer-spec")
        return str(layer_specs[0])
    requested_str = str(requested)
    if requested_str in axis_bundle["layers"]:
        return requested_str
    for layer_spec in layer_specs:
        layer_numbers = [int(value) for value in (axis_bundle["layers"][layer_spec].get("layer_numbers") or [])]
        if len(layer_numbers) == 1 and requested_str == str(layer_numbers[0]):
            return str(layer_spec)
    raise ValueError(f"Requested layer_spec={requested!r} is unavailable in axis bundle layers {layer_specs!r}")


def role_rows_for_axis(
    axis_spec: AxisSpec,
    *,
    axis_bundle: dict[str, Any],
    layer_spec: str,
) -> list[dict[str, Any]]:
    layer_entry = axis_bundle["layers"][layer_spec]
    negative_roles = {str(role_name) for role_name in axis_bundle.get("honest_roles", [])}
    positive_roles = {str(role_name) for role_name in axis_bundle.get("deceptive_roles", [])}

    rows: list[dict[str, Any]] = []
    for role_name, projection in sorted(layer_entry["role_projections"].items()):
        role_name = str(role_name)
        projection_values = [float(value) for value in projection]
        if role_name in negative_roles:
            side = axis_spec.negative_label
        elif role_name in positive_roles:
            side = axis_spec.positive_label
        else:
            side = "Unknown"
        rows.append(
            {
                "axis_slug": axis_spec.slug,
                "axis_display_name": axis_spec.display_name,
                "role_name": role_name,
                "role_label": canonical_role_label(role_name),
                "side": side,
                "pc1_projection": projection_values[0] if len(projection_values) > 0 else None,
                "pc2_projection": projection_values[1] if len(projection_values) > 1 else None,
                "pc3_projection": projection_values[2] if len(projection_values) > 2 else None,
            }
        )

    anchor_projection = [float(value) for value in layer_entry.get("anchor_projection", [])]
    rows.append(
        {
            "axis_slug": axis_spec.slug,
            "axis_display_name": axis_spec.display_name,
            "role_name": "default",
            "role_label": "Default",
            "side": "Default",
            "pc1_projection": anchor_projection[0] if len(anchor_projection) > 0 else None,
            "pc2_projection": anchor_projection[1] if len(anchor_projection) > 1 else None,
            "pc3_projection": anchor_projection[2] if len(anchor_projection) > 2 else None,
        }
    )
    return rows


def write_role_projection_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fieldnames = [
        "axis_slug",
        "axis_display_name",
        "role_name",
        "role_label",
        "side",
        "pc1_projection",
        "pc2_projection",
        "pc3_projection",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def apply_plot_style(ax) -> None:  # noqa: ANN001
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color="#d9e2ec", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa5b1")
    ax.spines["bottom"].set_color("#9aa5b1")


def plot_axis_scatter(
    rows: list[dict[str, Any]],
    *,
    axis_spec: AxisSpec,
    x_pc: int,
    y_pc: int,
    output_stem: Path,
    annotate: bool,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    ensure_dir(output_stem.parent)
    x_key = f"pc{x_pc}_projection"
    y_key = f"pc{y_pc}_projection"
    if any(row.get(x_key) is None for row in rows) or any(row.get(y_key) is None for row in rows):
        raise ValueError(
            f"Axis {axis_spec.slug!r} does not have enough stored PC projections for PC{x_pc} and PC{y_pc}"
        )

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    apply_plot_style(ax)
    style_by_side = {
        axis_spec.negative_label: {"label": axis_spec.negative_label, "color": "#1d4ed8", "marker": "s", "size": 95},
        axis_spec.positive_label: {"label": axis_spec.positive_label, "color": "#c2410c", "marker": "o", "size": 95},
        "Default": {"label": "Default", "color": "#475569", "marker": "*", "size": 180},
        "Unknown": {"label": "Unknown", "color": "#6b7280", "marker": "x", "size": 85},
    }

    ordered_sides = [axis_spec.negative_label, axis_spec.positive_label, "Default", "Unknown"]
    for side_name in ordered_sides:
        side_rows = [row for row in rows if row["side"] == side_name]
        if not side_rows:
            continue
        style = style_by_side[side_name]
        ax.scatter(
            [float(row[x_key]) for row in side_rows],
            [float(row[y_key]) for row in side_rows],
            s=style["size"],
            alpha=0.92,
            color=style["color"],
            marker=style["marker"],
            edgecolors="white" if side_name != "Unknown" else None,
            linewidths=1.1 if side_name != "Unknown" else None,
            label=style["label"],
            zorder=3,
        )
        if annotate:
            for row in side_rows:
                ax.annotate(
                    str(row["role_label"]),
                    (float(row[x_key]), float(row[y_key])),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=9.2,
                    color="#243b53",
                )

    ax.axhline(0.0, color="#9aa5b1", linewidth=1.0, alpha=0.9)
    ax.axvline(0.0, color="#9aa5b1", linewidth=1.0, alpha=0.9)
    ax.set_title(f"{axis_spec.display_name} Role Separation (PC{x_pc} vs PC{y_pc})", fontsize=15, fontweight="bold")
    ax.set_xlabel(f"PC{x_pc} Projection", fontsize=12)
    ax.set_ylabel(f"PC{y_pc} Projection", fontsize=12)
    ax.legend(frameon=False)
    fig.tight_layout()

    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def main() -> None:
    args = parse_args()
    if args.x_pc < 1 or args.y_pc < 1:
        raise ValueError("--x-pc and --y-pc must be >= 1")

    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    axis_specs = [parse_axis_spec(raw) for raw in args.axis]

    for axis_spec in axis_specs:
        axis_bundle = load_role_axis_bundle(axis_spec.bundle_path)
        layer_spec = choose_layer_spec(axis_bundle, args.layer_spec)
        rows = role_rows_for_axis(axis_spec, axis_bundle=axis_bundle, layer_spec=layer_spec)

        csv_path = output_dir / f"{axis_spec.slug}_role_pc_projections.csv"
        write_role_projection_csv(rows, csv_path)

        scatter_outputs = plot_axis_scatter(
            rows,
            axis_spec=axis_spec,
            x_pc=args.x_pc,
            y_pc=args.y_pc,
            output_stem=output_dir / f"{axis_spec.slug}_roles_pc{args.x_pc}_pc{args.y_pc}",
            annotate=not args.no_annotations,
        )

        print(f"[role-axis-pc-scatter] axis={axis_spec.slug} bundle={axis_spec.bundle_path}")
        print(f"[role-axis-pc-scatter] csv={csv_path}")
        print(f"[role-axis-pc-scatter] png={scatter_outputs['png']}")
        print(f"[role-axis-pc-scatter] pdf={scatter_outputs['pdf']}")


if __name__ == "__main__":
    main()
