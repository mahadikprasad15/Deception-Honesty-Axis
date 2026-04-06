#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from deception_honesty_axis.common import ensure_dir, find_repo_root, load_jsonl, make_run_id, read_json, slugify, utc_now_iso, write_json
from deception_honesty_axis.config import ExperimentConfig, corpus_root as resolve_corpus_root, load_config
from deception_honesty_axis.indexes import load_index
from deception_honesty_axis.metadata import write_analysis_manifest, write_stage_status
from deception_honesty_axis.modeling import load_model_and_tokenizer
from deception_honesty_axis.records import load_activation_records
from deception_honesty_axis.role_axis_transfer import resolve_layer_specs, role_vector_for_spec
from deception_honesty_axis.work_units import build_work_units


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run logit-lens analysis on selected per-axis role vectors and export top upvoted tokens."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional repo root. Defaults to auto-detection from this script path.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Optional fixed-track runs root. Defaults to <repo>/artifacts/runs/fixed-track-search.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional analysis output root. Defaults to <repo>/artifacts/analysis/role-logit-lens.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Optional variant filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of upvoted tokens to export per role.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="delta",
        choices=["delta", "absolute"],
        help="Use role minus anchor ('delta') or the raw role vector ('absolute') before unembedding.",
    )
    parser.add_argument(
        "--include-anchor",
        action="store_true",
        help="Also export the anchor/default role. Off by default because delta mode yields a zero vector for the anchor.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional fixed analysis run id.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing analysis run if checkpoints are present.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute already completed axes in the current analysis run.",
    )
    return parser.parse_args()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed_axis_ids": [], "updated_at": None}
    return read_json(path)


def save_progress(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = utc_now_iso()
    write_json(path, payload)


def analysis_run_root(output_root: Path, run_id: str) -> Path:
    return output_root / run_id


def discover_search_runs(runs_root: Path, variants: set[str]) -> list[Path]:
    candidates = []
    for selection_path in sorted(runs_root.glob("**/results/final_selection.json")):
        search_run_dir = selection_path.parent.parent
        try:
            relative_parts = search_run_dir.resolve().relative_to(runs_root.resolve()).parts
        except ValueError:
            continue
        if len(relative_parts) < 5:
            continue
        variant = relative_parts[2]
        if variants and variant not in variants:
            continue
        candidates.append(search_run_dir)
    return candidates


def compact_axis_label(relative_parts: tuple[str, ...], selection: dict[str, Any]) -> str:
    variant = relative_parts[2] if len(relative_parts) > 2 else "unknown"
    method = str(selection.get("method") or "unknown")
    objective = str(selection.get("objective") or "unknown")
    probe = "contrast" if "contrast" in method else "pc1" if "pc1" in method else method
    objective_label = "thr" if "threshold" in objective else "mean"
    return f"{variant} | {probe} | {objective_label}"


def variant_from_relative_parts(relative_parts: tuple[str, ...]) -> str | None:
    return relative_parts[2] if len(relative_parts) > 2 else None


def axis_id_for_run(relative_parts: tuple[str, ...]) -> str:
    if len(relative_parts) >= 5:
        model_slug, dataset_slug, variant, track_name, run_name = relative_parts[:5]
        return slugify(f"{model_slug}__{dataset_slug}__{variant}__{track_name}__{run_name}")
    return slugify("__".join(relative_parts))


def resolve_experiment_config(search_run_dir: Path, repo_root: Path) -> ExperimentConfig:
    manifest = read_json(search_run_dir / "meta" / "run_manifest.json")
    candidates = [
        manifest.get("experiment_config_path"),
    ]
    selected_probe_path = search_run_dir / "selected" / "probe_config.json"
    if selected_probe_path.exists():
        probe_payload = read_json(selected_probe_path)
        role_axis = probe_payload.get("role_axis", {})
        candidates.append(role_axis.get("experiment_config"))

    for value in candidates:
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        if path.exists():
            return load_config(path)
    raise FileNotFoundError(
        f"Could not resolve experiment config for {search_run_dir}. "
        "If this run used a temp config, recreate it locally before rerunning."
    )


def load_selected_questions(search_run_dir: Path, selection: dict[str, Any]) -> list[str]:
    selected_path = search_run_dir / "selected" / "questions.jsonl"
    if selected_path.exists():
        rows = load_jsonl(selected_path)
        if rows:
            return [str(row["question_id"]) for row in rows if "question_id" in row]
    return [str(item) for item in selection.get("question_ids", [])]


def load_selected_role_entries(search_run_dir: Path, selection: dict[str, Any]) -> list[dict[str, Any]]:
    selected_manifest_path = search_run_dir / "selected" / "role_manifest.json"
    role_side_by_name = {
        "default": "anchor",
        **{str(name): "deceptive" for name in selection.get("deceptive_roles", [])},
        **{str(name): "honest" for name in selection.get("honest_roles", [])},
    }
    if selected_manifest_path.exists():
        payload = read_json(selected_manifest_path)
        rows = []
        for order, entry in enumerate(payload.get("roles", [])):
            role_name = str(entry["name"])
            rows.append(
                {
                    "role_name": role_name,
                    "role_side": role_side_by_name.get(role_name, "unknown"),
                    "role_order": order,
                    "source": entry.get("source"),
                    "anchor_only": bool(entry.get("anchor_only", False)),
                }
            )
        if rows:
            return rows
    rows = [{"role_name": "default", "role_side": "anchor", "role_order": 0, "source": None, "anchor_only": True}]
    role_order = 1
    for role_name in selection.get("deceptive_roles", []):
        rows.append(
            {
                "role_name": str(role_name),
                "role_side": "deceptive",
                "role_order": role_order,
                "source": None,
                "anchor_only": False,
            }
        )
        role_order += 1
    for role_name in selection.get("honest_roles", []):
        rows.append(
            {
                "role_name": str(role_name),
                "role_side": "honest",
                "role_order": role_order,
                "source": None,
                "anchor_only": False,
            }
        )
        role_order += 1
    return rows


def load_selected_role_vectors(
    search_run_dir: Path,
    repo_root: Path,
) -> tuple[ExperimentConfig, dict[str, torch.Tensor], str, dict[str, Any], str, str, str | None]:
    selection = read_json(search_run_dir / "results" / "final_selection.json")
    experiment_config = resolve_experiment_config(search_run_dir, repo_root)
    selected_question_ids = set(load_selected_questions(search_run_dir, selection))
    role_entries = load_selected_role_entries(search_run_dir, selection)
    selected_role_names = {row["role_name"] for row in role_entries}
    work_units = build_work_units(experiment_config.repo_root, experiment_config)
    target_ids = {
        unit["item_id"]
        for unit in work_units
        if unit["role"] in selected_role_names and str(unit["question_id"]) in selected_question_ids
    }
    activation_index = load_index(resolve_corpus_root(experiment_config), "activations")
    missing = sorted(target_ids - set(activation_index))
    if missing:
        preview = missing[:8]
        raise RuntimeError(
            f"Missing {len(missing)} activation records for selected roles/questions in {search_run_dir}; "
            f"first missing ids: {preview}"
        )
    selected_rows = [row for item_id, row in activation_index.items() if item_id in target_ids]
    records = load_activation_records(resolve_corpus_root(experiment_config), selected_rows)
    if not records:
        raise RuntimeError(f"No activation records matched selected roles/questions in {search_run_dir}")

    grouped: dict[str, list[torch.Tensor]] = {}
    for record in records:
        grouped.setdefault(str(record["role"]), []).append(record["pooled_activations"].to(torch.float32))
    role_vectors = {
        role_name: torch.stack(tensors, dim=0).mean(dim=0)
        for role_name, tensors in grouped.items()
    }
    layer_spec = str(selection["layer_spec"])
    relative_parts = search_run_dir.resolve().relative_to((repo_root / "artifacts" / "runs" / "fixed-track-search").resolve()).parts
    axis_id = axis_id_for_run(relative_parts)
    axis_label = compact_axis_label(relative_parts, selection)
    variant = variant_from_relative_parts(relative_parts)
    return experiment_config, role_vectors, layer_spec, selection, axis_id, axis_label, variant


def resolve_model_components(model) -> tuple[Any, Any]:
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise AttributeError("Model does not expose lm_head; cannot run logit lens.")

    final_norm = None
    base_model = getattr(model, "model", None)
    if base_model is not None and hasattr(base_model, "norm"):
        final_norm = base_model.norm
    elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        final_norm = model.transformer.ln_f
    return lm_head, final_norm


def token_repr(tokenizer, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens([token_id])[0]
    if token is None:
        return "<none>"
    return token.replace("\n", "\\n")


def top_upvoted_tokens(
    vector: torch.Tensor,
    lm_head,
    final_norm,
    tokenizer,
    top_k: int,
) -> list[dict[str, Any]]:
    with torch.no_grad():
        hidden = vector.to(next(lm_head.parameters()).device).to(next(lm_head.parameters()).dtype)
        if final_norm is not None:
            hidden = final_norm(hidden)
        logits = lm_head(hidden).to(torch.float32).cpu()
        values, indices = torch.topk(logits, k=top_k)
    rows = []
    for rank, (value, token_id) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        rows.append(
            {
                "rank": rank,
                "token_id": int(token_id),
                "token": token_repr(tokenizer, int(token_id)),
                "logit": float(value),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root(Path(__file__).resolve())
    runs_root = args.runs_root.resolve() if args.runs_root else repo_root / "artifacts" / "runs" / "fixed-track-search"
    output_root = args.output_root.resolve() if args.output_root else repo_root / "artifacts" / "analysis" / "role-logit-lens"
    run_id = args.run_id or make_run_id()
    run_root = analysis_run_root(output_root, run_id)

    for relative in ("inputs", "checkpoints", "results", "meta", "logs"):
        ensure_dir(run_root / relative)

    progress_path = run_root / "checkpoints" / "progress.json"
    progress = load_progress(progress_path) if args.resume else {"completed_axis_ids": [], "updated_at": None}
    completed_axis_ids = set(progress.get("completed_axis_ids", []))

    run_paths = discover_search_runs(runs_root, set(args.variant))
    write_json(run_root / "inputs" / "source_runs.json", {"source_runs": [str(path.resolve()) for path in run_paths]})

    # Load the model once. We use the first resolvable experiment config to get model metadata.
    first_config = None
    for search_run_dir in run_paths:
        try:
            first_config = resolve_experiment_config(search_run_dir, repo_root)
            break
        except FileNotFoundError:
            continue
    if first_config is None:
        raise FileNotFoundError("Could not resolve an experiment config from any selected fixed-track run.")

    tokenizer, model = load_model_and_tokenizer(first_config.raw["model"])
    lm_head, final_norm = resolve_model_components(model)

    write_analysis_manifest(
        run_root,
        {
            "analysis_kind": "role-logit-lens",
            "repo_root": str(repo_root),
            "runs_root": str(runs_root),
            "variant_filters": list(args.variant),
            "mode": args.mode,
            "top_k": args.top_k,
            "include_anchor": bool(args.include_anchor),
            "model_id": first_config.model_id,
            "created_at": utc_now_iso(),
        },
    )
    write_stage_status(run_root, "role_logit_lens", "running", {"source_run_count": len(run_paths)})

    result_rows: list[dict[str, Any]] = []
    for search_run_dir in run_paths:
        experiment_config, role_vectors, layer_spec, selection, axis_id, axis_label, variant = load_selected_role_vectors(search_run_dir, repo_root)
        if axis_id in completed_axis_ids and not args.force:
            existing_path = run_root / "results" / f"{axis_id}.json"
            if existing_path.exists():
                result_rows.extend(read_json(existing_path).get("rows", []))
            continue
        resolved_spec = resolve_layer_specs(int(next(iter(role_vectors.values())).shape[0]), [layer_spec])[0]
        anchor_role = "default"
        if anchor_role not in role_vectors:
            raise KeyError(f"Axis {axis_id} is missing anchor role '{anchor_role}'")
        anchor_vector = role_vector_for_spec(role_vectors[anchor_role], resolved_spec).detach().cpu().to(torch.float32)
        role_entries = load_selected_role_entries(search_run_dir, selection)

        axis_rows: list[dict[str, Any]] = []
        for role_entry in role_entries:
            role_name = str(role_entry["role_name"])
            if role_name == anchor_role and not args.include_anchor:
                continue
            if role_name not in role_vectors:
                continue
            base_vector = role_vector_for_spec(role_vectors[role_name], resolved_spec).detach().cpu().to(torch.float32)
            lens_vector = base_vector if args.mode == "absolute" else base_vector - anchor_vector
            token_rows = top_upvoted_tokens(lens_vector, lm_head, final_norm, tokenizer, args.top_k)

            row = {
                "axis_id": axis_id,
                "axis_label": axis_label,
                "variant": variant,
                "method": selection.get("method"),
                "layer_spec": layer_spec,
                "objective": selection.get("objective"),
                "threshold": selection.get("threshold"),
                "role_name": role_name,
                "role_side": role_entry["role_side"],
                "role_order": role_entry["role_order"],
                "mode": args.mode,
                "vector_norm": float(torch.linalg.norm(lens_vector)),
                "source_run_dir": str(search_run_dir.resolve()),
            }
            joined_tokens = []
            for token_row in token_rows:
                suffix = str(token_row["rank"])
                row[f"token_{suffix}"] = token_row["token"]
                row[f"token_id_{suffix}"] = token_row["token_id"]
                row[f"logit_{suffix}"] = token_row["logit"]
                joined_tokens.append(f"{token_row['token']} ({token_row['logit']:.3f})")
            row["top_tokens"] = ", ".join(joined_tokens)
            axis_rows.append(row)

        axis_rows = sorted(axis_rows, key=lambda item: (item["role_side"], item["role_order"], item["role_name"]))
        result_rows.extend(axis_rows)
        write_json(run_root / "results" / f"{axis_id}.json", {"rows": axis_rows})
        completed_axis_ids.add(axis_id)
        save_progress(progress_path, {"completed_axis_ids": sorted(completed_axis_ids)})

    result_rows = sorted(
        result_rows,
        key=lambda row: (
            str(row.get("variant") or ""),
            str(row.get("method") or ""),
            str(row.get("objective") or ""),
            int(row.get("role_order") or 0),
            str(row.get("role_name") or ""),
        ),
    )
    if result_rows:
        write_csv(run_root / "results" / "role_logit_lens.csv", list(result_rows[0].keys()), result_rows)
        markdown_lines = [
            "| axis | role side | role | top 3 upvoted tokens |",
            "| --- | --- | --- | --- |",
        ]
        for row in result_rows:
            markdown_lines.append(
                f"| {row['axis_label']} | {row['role_side']} | {row['role_name']} | {row['top_tokens']} |"
            )
        (run_root / "results" / "role_logit_lens.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    write_json(
        run_root / "results" / "summary.json",
        {
            "axis_count": len(completed_axis_ids),
            "row_count": len(result_rows),
            "mode": args.mode,
            "top_k": args.top_k,
            "generated_at": utc_now_iso(),
        },
    )
    write_stage_status(
        run_root,
        "role_logit_lens",
        "completed",
        {
            "axis_count": len(completed_axis_ids),
            "row_count": len(result_rows),
        },
    )
    print(f"[role-logit-lens] wrote analysis artifacts to {run_root}")


if __name__ == "__main__":
    main()
