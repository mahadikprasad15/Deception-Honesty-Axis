from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from deception_honesty_axis.source_curation import materialize_curated_variant_sources


class SourceCurationTest(unittest.TestCase):
    def test_materialize_curated_variant_sources_selects_requested_roles_and_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            (repo_root / ".git").mkdir()
            upstream_root = repo_root / "upstream"
            (upstream_root / "roles" / "instructions").mkdir(parents=True)

            (upstream_root / "extraction_questions.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": 0, "question": "Question zero"}),
                        json.dumps({"id": 4, "question": "Question four"}),
                        json.dumps({"id": 9, "question": "Question nine"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            (upstream_root / "roles" / "instructions" / "rogue.json").write_text(
                json.dumps(
                    {
                        "instruction": [
                            {"pos": "rogue-1"},
                            {"pos": "rogue-2"},
                            {"pos": "rogue-3"},
                            {"pos": "rogue-4"},
                            {"pos": "rogue-5"},
                        ],
                        "questions": ["q1", "q2"],
                        "eval_prompt": "eval",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            spec_path = repo_root / "spec.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "upstream_raw_base": upstream_root.as_uri(),
                        "question_source": "extraction_questions.jsonl",
                        "question_output": "data/questions/extraction_questions_v5_selected.jsonl",
                        "selected_question_ids": [9, 0],
                        "manifest_output": "data/manifests/source_variants/test_manifest.json",
                        "roles": [
                            {
                                "role": "rogue",
                                "source_path": "roles/instructions/rogue.json",
                                "output_path": "data/roles/instructions/rogue_v5.json",
                                "instruction_indices": [1, 3, 5],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            payload = materialize_curated_variant_sources(spec_path, repo_root, force=False)

            question_output = repo_root / "data" / "questions" / "extraction_questions_v5_selected.jsonl"
            question_rows = [json.loads(line) for line in question_output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["id"] for row in question_rows], [9, 0])

            rogue_output = repo_root / "data" / "roles" / "instructions" / "rogue_v5.json"
            rogue_data = json.loads(rogue_output.read_text(encoding="utf-8"))
            self.assertEqual(
                [entry["pos"] for entry in rogue_data["instruction"]],
                ["rogue-1", "rogue-3", "rogue-5"],
            )
            self.assertEqual(rogue_data["questions"], ["q1", "q2"])
            self.assertEqual(rogue_data["eval_prompt"], "eval")

            manifest_path = Path(payload["manifest_path"])
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["files"]), 2)


if __name__ == "__main__":
    unittest.main()
