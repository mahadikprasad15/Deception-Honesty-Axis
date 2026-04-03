from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from deception_honesty_axis.common import ensure_dir, read_json, sha256_file, utc_now_iso, write_json


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def _resolve_source_url(raw_base: str, path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://", "file://")):
        return path_or_url
    return f"{raw_base.rstrip('/')}/{path_or_url.lstrip('/')}"


def _fetch_text(url: str) -> str:
    with urllib.request.urlopen(url) as response:  # noqa: S310
        return response.read().decode("utf-8")


def _select_question_rows(raw_text: str, selected_ids: list[int]) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        if "id" not in row:
            continue
        question_id = int(row["id"])
        if question_id in by_id:
            raise ValueError(f"Duplicate shared question id encountered upstream: {question_id}")
        by_id[question_id] = row
    missing = [question_id for question_id in selected_ids if question_id not in by_id]
    if missing:
        raise ValueError(f"Missing shared question ids in upstream source: {missing}")
    return [by_id[question_id] for question_id in selected_ids]


def _select_instruction_variants(raw_role: dict[str, Any], selected_indices: list[int]) -> list[dict[str, Any]]:
    instruction_variants = list(raw_role.get("instruction", []))
    if not instruction_variants:
        raise ValueError("Upstream role JSON does not define an 'instruction' list")
    invalid = [index for index in selected_indices if index < 1 or index > len(instruction_variants)]
    if invalid:
        raise ValueError(
            f"Requested 1-based instruction indices {invalid} exceed upstream count {len(instruction_variants)}"
        )
    return [instruction_variants[index - 1] for index in selected_indices]


def materialize_curated_variant_sources(
    spec_path: Path,
    repo_root: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    spec = read_json(spec_path)
    raw_base = str(spec["upstream_raw_base"])
    selected_question_ids = [int(value) for value in spec["selected_question_ids"]]
    manifest_output = _resolve_repo_path(repo_root, str(spec["manifest_output"]))

    generated_paths: set[str] = set()
    entries: list[dict[str, Any]] = []

    question_output = _resolve_repo_path(repo_root, str(spec["question_output"]))
    question_source_url = _resolve_source_url(raw_base, str(spec["question_source"]))
    if question_output.exists() and not force:
        question_status = "skipped_existing"
    else:
        selected_rows = _select_question_rows(_fetch_text(question_source_url), selected_question_ids)
        ensure_dir(question_output.parent)
        with question_output.open("w", encoding="utf-8") as handle:
            for row in selected_rows:
                handle.write(json.dumps(row, ensure_ascii=True))
                handle.write("\n")
        question_status = "written"
    generated_paths.add(str(question_output))
    entries.append(
        {
            "kind": "curated_questions",
            "source_url": question_source_url,
            "output_path": str(question_output),
            "selected_ids": selected_question_ids,
            "sha256": sha256_file(question_output),
            "status": question_status,
        }
    )

    for role_spec in spec["roles"]:
        role_name = str(role_spec["role"])
        output_path = _resolve_repo_path(repo_root, str(role_spec["output_path"]))
        source_url = _resolve_source_url(raw_base, str(role_spec["source_path"]))
        selected_indices = [int(value) for value in role_spec["instruction_indices"]]
        if output_path.exists() and not force:
            status = "skipped_existing"
        else:
            raw_role = json.loads(_fetch_text(source_url))
            curated_role = dict(raw_role)
            curated_role["instruction"] = _select_instruction_variants(raw_role, selected_indices)
            write_json(output_path, curated_role)
            status = "written"
        generated_paths.add(str(output_path))
        entries.append(
            {
                "kind": "curated_role",
                "role": role_name,
                "source_url": source_url,
                "output_path": str(output_path),
                "selected_instruction_indices": selected_indices,
                "sha256": sha256_file(output_path),
                "status": status,
            }
        )

    payload = {
        "upstream_repo": spec.get("upstream_repo", "https://github.com/safety-research/assistant-axis"),
        "upstream_raw_base": raw_base,
        "generated_at": utc_now_iso(),
        "spec_path": str(spec_path),
        "files": entries,
    }
    write_json(manifest_output, payload)
    return {
        "manifest_path": str(manifest_output),
        "files": entries,
        "generated_paths": sorted(generated_paths),
    }
