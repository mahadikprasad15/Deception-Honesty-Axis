from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from deception_honesty_axis.config import corpus_root, load_config


class ConfigTest(unittest.TestCase):
    def test_corpus_root_uses_explicit_corpus_name_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            (repo_root / ".git").mkdir()
            (repo_root / "configs" / "experiments").mkdir(parents=True)
            config_path = repo_root / "configs" / "experiments" / "test.json"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "demo-exp",
                        "model": {"id": "meta-llama/Llama-3.2-3B-Instruct", "generation": {}, "dtype": "float16"},
                        "data": {
                            "question_file": "data/questions/extraction_questions.jsonl",
                            "role_manifest_file": "data/manifests/role_sets/deception_honesty_v1.json",
                        },
                        "subset": {
                            "instruction_selection": "first_n",
                            "instruction_count": 2,
                            "question_selection": "first_n",
                            "question_count": 100,
                        },
                        "artifacts": {
                            "root": "artifacts",
                            "dataset_name": "assistant-axis",
                            "analysis_variant": "center-only",
                            "corpus_name": "deception-v1",
                        },
                        "saving": {"rollout_shard_size": 100, "activation_shard_size": 32, "save_every": 25},
                        "analysis": {"filter_name": "all-responses", "pooling": "mean_response", "anchor_role": "default"},
                        "hf": {"dataset_repo_id": "owner/repo", "root_prefix": "test", "private": True},
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)
            self.assertEqual(
                corpus_root(config),
                (repo_root / "artifacts" / "corpora" / "meta-llama-llama-3-2-3b-instruct" / "assistant-axis" / "deception-v1").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
