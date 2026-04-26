from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from deception_honesty_axis.common import slugify


class PrepareSycophancyActivationRowTransferConfigTest(unittest.TestCase):
    def test_script_writes_config_from_completed_runs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "prepare_sycophancy_activation_row_transfer_config.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_repo = Path(tmpdir) / "repo"
            (temp_repo / ".git").mkdir(parents=True)
            model_slug = slugify("meta-llama/Meta-Llama-3-8B-Instruct")
            run_dir = (
                temp_repo
                / "artifacts"
                / "runs"
                / "external-sycophancy-activation-extraction"
                / "ariana-sycophancy-train"
                / model_slug
                / "fixed-run"
            )
            (run_dir / "results").mkdir(parents=True)
            (run_dir / "meta").mkdir(parents=True)
            (run_dir / "results" / "records.jsonl").write_text(
                json.dumps(
                    {
                        "ids": "row-1",
                        "dataset_source": "ariana-sycophancy-train",
                        "activation": [0.1, 0.2],
                        "label": 1,
                        "activation_pooling": "mean_response",
                        "activation_layer_number": 16,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "meta" / "stage_status.json").write_text(
                json.dumps(
                    {
                        "stages": {
                            "extract_external_sycophancy_activations": {
                                "status": "completed",
                            }
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            output_config = temp_repo / "configs" / "generated.json"
            env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--targets",
                    "Sycophancy Dataset=ariana-sycophancy-train",
                    "--output-config",
                    str(output_config),
                ],
                cwd=temp_repo,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(output_config.read_text(encoding="utf-8"))
            self.assertEqual(config["activation_rows"]["expected_layer_number"], 16)
            self.assertEqual(config["datasets"][0]["name"], "Sycophancy Dataset")
            self.assertEqual(config["datasets"][0]["source_kind"], "activation_rows")
            self.assertEqual(
                config["datasets"][0]["activation_jsonl"],
                "artifacts/runs/external-sycophancy-activation-extraction/"
                "ariana-sycophancy-train/meta-llama-meta-llama-3-8b-instruct/"
                "fixed-run/results/records.jsonl",
            )


if __name__ == "__main__":
    unittest.main()
