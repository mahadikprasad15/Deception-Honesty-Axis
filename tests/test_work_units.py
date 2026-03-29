from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from deception_honesty_axis.config import ExperimentConfig
from deception_honesty_axis.work_units import anchor_role, build_work_units, expand_work_units, load_role_prompts, pca_roles


class WorkUnitsTest(unittest.TestCase):
    def test_expand_work_units_uses_first_n_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            (repo_root / ".git").mkdir()
            (repo_root / "data" / "questions").mkdir(parents=True)
            (repo_root / "data" / "roles" / "instructions").mkdir(parents=True)
            (repo_root / "data" / "manifests" / "role_sets").mkdir(parents=True)

            questions_path = repo_root / "data" / "questions" / "extraction_questions.jsonl"
            questions_path.write_text(
                "\n".join(
                    json.dumps({"id": f"q{idx}", "question": f"Question {idx}"})
                    for idx in range(4)
                )
                + "\n",
                encoding="utf-8",
            )

            for role_name in ("default", "spy"):
                role_path = repo_root / "data" / "roles" / "instructions" / f"{role_name}.json"
                role_path.write_text(
                    json.dumps({"instructions": [f"{role_name} prompt 0", f"{role_name} prompt 1", f"{role_name} prompt 2"]}),
                    encoding="utf-8",
                )

            manifest_path = repo_root / "data" / "manifests" / "role_sets" / "deception_v1.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "anchor_role": "default",
                        "roles": [
                            {"name": "default", "instruction_file": "data/roles/instructions/default.json", "anchor_only": True},
                            {"name": "spy", "instruction_file": "data/roles/instructions/spy.json", "anchor_only": False},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = ExperimentConfig(
                raw={
                    "experiment_name": "deception-assistant-pca",
                    "model": {"name": "meta-llama/Llama-3.2-3B-Instruct"},
                    "dataset": {
                        "name": "assistant-axis",
                        "question_file": "data/questions/extraction_questions.jsonl",
                        "question_count": 2,
                        "question_selection": "first_n",
                    },
                    "roles": {
                        "manifest": "data/manifests/role_sets/deception_v1.json",
                        "instruction_count": 2,
                        "instruction_selection": "first_n",
                    },
                    "storage": {"artifact_root": "artifacts", "save_every": 5, "hf_dataset_repo": "dummy/repo"},
                    "analysis": {"filter_name": "all_responses"},
                },
                path=repo_root / "configs.json",
                repo_root=repo_root,
            )

            units = expand_work_units(config)
            flat_units = build_work_units(repo_root, config)

            self.assertEqual(anchor_role(config), "default")
            self.assertEqual(pca_roles(config), ["spy"])
            self.assertEqual(len(units), 8)
            self.assertEqual(len(flat_units), 8)
            self.assertEqual(units[0].item_id, "default__p00__q000")
            self.assertEqual(units[-1].item_id, "spy__p01__q001")

    def test_load_role_prompts_supports_upstream_instruction_list_with_pos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            role_path = Path(tmp_dir) / "default.json"
            role_path.write_text(
                json.dumps(
                    {
                        "instruction": [
                            {"pos": ""},
                            {"pos": "You are an AI assistant."},
                            {"pos": "Respond as yourself."},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            prompts = load_role_prompts(role_path)
            self.assertEqual(prompts, ["You are an AI assistant.", "Respond as yourself."])


if __name__ == "__main__":
    unittest.main()
