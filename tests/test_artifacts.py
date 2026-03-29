from __future__ import annotations

import json
import unittest

from deception_honesty_axis.artifacts import corpus_paths, ensure_corpus_dirs, update_coverage
from deception_honesty_axis.config import load_config
from deception_honesty_axis.io_utils import read_json


class ArtifactsTest(unittest.TestCase):
    def test_coverage_tracks_missing_items(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp_dir:
            repo = Path(tmp_dir)
            (repo / ".git").mkdir()
            (repo / "configs" / "experiments").mkdir(parents=True)
            config = {
                "experiment_name": "test-exp",
                "model": {"id": "fake/model", "generation": {}, "dtype": "float16"},
                "data": {
                    "question_file": "data/questions/extraction_questions.jsonl",
                    "role_manifest_file": "data/manifests/role_sets/deception_v1.json",
                },
                "subset": {
                    "instruction_selection": "first_n",
                    "instruction_count": 2,
                    "question_selection": "first_n",
                    "question_count": 2,
                },
                "artifacts": {"root": "artifacts", "dataset_name": "assistant-axis", "analysis_variant": "center-only"},
                "saving": {"rollout_shard_size": 10, "activation_shard_size": 10, "save_every": 5},
                "analysis": {"pooling": "mean_response", "anchor_role": "default"},
                "hf": {"dataset_repo_id": "owner/repo", "root_prefix": "test", "private": True},
            }
            config_path = repo / "configs" / "experiments" / "deception_v1_llama32_3b.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            loaded = load_config(config_path)
            paths = corpus_paths(loaded)
            ensure_corpus_dirs(paths)

            coverage_path = update_coverage(paths, ["a", "b", "c"], {"a"}, {"a", "b"})
            coverage = read_json(coverage_path)

            self.assertEqual(coverage["target_item_count"], 3)
            self.assertEqual(coverage["rollout_count"], 1)
            self.assertEqual(coverage["activation_count"], 2)
            self.assertEqual(coverage["missing_rollouts"], ["b", "c"])
            self.assertEqual(coverage["missing_activations"], ["c"])


if __name__ == "__main__":
    unittest.main()
