from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from deception_honesty_axis.common import load_jsonl, read_json
from deception_honesty_axis.shared_unified_axis_prep import prepare_shared_unified_axis_experiment


class SharedUnifiedAxisPrepTest(unittest.TestCase):
    def test_prepare_shared_unified_axis_experiment_writes_balanced_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            (repo_root / ".git").mkdir(parents=True)
            (repo_root / "configs" / "experiments").mkdir(parents=True)
            (repo_root / "data" / "questions").mkdir(parents=True)
            (repo_root / "data" / "manifests" / "role_sets").mkdir(parents=True)

            quantity_questions = repo_root / "data" / "questions" / "quantity.jsonl"
            sycophancy_questions = repo_root / "data" / "questions" / "syco.jsonl"
            quantity_questions.write_text(
                "".join(
                    json.dumps({"question_id": f"quant{i}", "question": f"Q{i}"}) + "\n"
                    for i in range(1, 4)
                ),
                encoding="utf-8",
            )
            sycophancy_questions.write_text(
                "".join(
                    json.dumps({"question_id": f"syc{i}", "question": f"S{i}"}) + "\n"
                    for i in range(1, 5)
                ),
                encoding="utf-8",
            )

            quantity_manifest = repo_root / "data" / "manifests" / "role_sets" / "quantity.json"
            sycophancy_manifest = repo_root / "data" / "manifests" / "role_sets" / "syco.json"
            quantity_manifest.write_text(
                json.dumps(
                    {
                        "name": "quantity",
                        "anchor_role": "default",
                        "pca_roles": ["handler", "ombudsman"],
                        "roles": [
                            {"name": "default", "source": "default.json", "anchor_only": True, "role_side": "anchor"},
                            {"name": "handler", "source": "handler.json", "anchor_only": False, "role_side": "deceptive"},
                            {"name": "ombudsman", "source": "ombudsman.json", "anchor_only": False, "role_side": "honest"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            sycophancy_manifest.write_text(
                json.dumps(
                    {
                        "name": "syco",
                        "anchor_role": "default",
                        "pca_roles": ["yes_man", "critic"],
                        "roles": [
                            {"name": "default", "source": "default.json", "anchor_only": True, "side": "anchor"},
                            {"name": "yes_man", "source": "yes_man.json", "anchor_only": False, "side": "sycophantic"},
                            {"name": "critic", "source": "critic.json", "anchor_only": False, "side": "non_sycophantic"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            quantity_config = repo_root / "configs" / "experiments" / "quantity.json"
            sycophancy_config = repo_root / "configs" / "experiments" / "syco.json"
            base_model = {
                "id": "meta-llama/Llama-3.2-3B-Instruct",
                "dtype": "float16",
                "device": "auto",
                "trust_remote_code": False,
                "generation": {
                    "max_new_tokens": 128,
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "do_sample": True,
                },
            }
            quantity_config.write_text(
                json.dumps(
                    {
                        "experiment_name": "quantity-axis-v2",
                        "model": base_model,
                        "data": {
                            "question_file": "data/questions/quantity.jsonl",
                            "role_manifest_file": "data/manifests/role_sets/quantity.json",
                        },
                        "subset": {
                            "instruction_selection": "first_n",
                            "instruction_count": 2,
                            "question_selection": "first_n",
                            "question_count": 3,
                        },
                        "artifacts": {
                            "root": "artifacts",
                            "dataset_name": "assistant-axis",
                            "analysis_variant": "center-only",
                            "corpus_name": "quantity-axis-v2",
                        },
                        "saving": {
                            "rollout_shard_size": 100,
                            "activation_shard_size": 32,
                            "save_every": 25,
                        },
                        "analysis": {
                            "filter_name": "all-responses",
                            "pooling": "mean_response",
                            "anchor_role": "default",
                        },
                        "hf": {
                            "dataset_repo_id": "Prasadmahadik/deception-honesty-axis-artifacts",
                            "root_prefix": "deception-honesty-axis",
                            "private": True,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            sycophancy_config.write_text(
                json.dumps(
                    {
                        "experiment_name": "sycophancy-pilot-v1-pca",
                        "model": base_model,
                        "data": {
                            "question_file": "data/questions/syco.jsonl",
                            "role_manifest_file": "data/manifests/role_sets/syco.json",
                        },
                        "subset": {
                            "instruction_selection": "first_n",
                            "instruction_count": 2,
                            "question_selection": "first_n",
                            "question_count": 4,
                        },
                        "artifacts": {
                            "root": "artifacts",
                            "dataset_name": "assistant-axis",
                            "analysis_variant": "center-only",
                            "corpus_name": "sycophancy-pilot-v1",
                        },
                        "saving": {
                            "rollout_shard_size": 100,
                            "activation_shard_size": 32,
                            "save_every": 25,
                        },
                        "analysis": {
                            "filter_name": "all-responses",
                            "pooling": "mean_response",
                            "anchor_role": "default",
                            "activation_layers": [14],
                        },
                        "hf": {
                            "dataset_repo_id": "Prasadmahadik/deception-honesty-axis-artifacts",
                            "root_prefix": "deception-honesty-axis",
                            "private": True,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            prepared = prepare_shared_unified_axis_experiment(
                quantity_experiment_config_path=quantity_config,
                sycophancy_experiment_config_path=sycophancy_config,
                artifact_root=repo_root / "artifacts",
                questions_per_source=2,
                pc_count=64,
                layer_number=14,
                run_id="fixed-run",
            )

            question_rows = load_jsonl(prepared.question_file)
            self.assertEqual(len(question_rows), 4)
            self.assertEqual(
                [row["question_origin"] for row in question_rows],
                ["deception", "deception", "sycophancy", "sycophancy"],
            )

            merged_manifest = read_json(prepared.role_manifest_file)
            self.assertEqual(merged_manifest["anchor_role"], "default")
            self.assertEqual({role["name"] for role in merged_manifest["roles"]}, {"default", "handler", "ombudsman", "yes_man", "critic"})

            experiment_config = read_json(prepared.experiment_config_file)
            self.assertEqual(experiment_config["subset"]["question_count"], 4)
            self.assertEqual(experiment_config["analysis"]["activation_layers"], [14])

            bundle_config = read_json(prepared.bundle_config_file)
            self.assertEqual(bundle_config["axis"]["pc_count"], 64)
            self.assertEqual(bundle_config["axis"]["shared_source"]["safe_roles"], ["ombudsman", "critic"])
            self.assertEqual(bundle_config["axis"]["shared_source"]["harmful_roles"], ["handler", "yes_man"])


if __name__ == "__main__":
    unittest.main()
