from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from deception_honesty_axis.transfer_comparison import (
    load_comparison_manifest,
    run_zero_shot_transfer_comparison,
)


class TransferComparisonTest(unittest.TestCase):
    def test_run_zero_shot_transfer_comparison_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            (repo_root / ".git").mkdir()
            artifacts_root = repo_root / "artifacts"

            variants = [
                ("baseline_old", "baseline-run"),
                ("triggered_augmented", "triggered-aug-run"),
                ("triggered_only", "triggered-only-run"),
            ]
            datasets = [
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
            methods = ["contrast_zero_shot", "pc1_zero_shot"]
            layer_specs = ["7", "14", "21", "28"]

            variant_entries = []
            for variant_idx, (label, run_name) in enumerate(variants):
                run_dir = (
                    artifacts_root
                    / "runs"
                    / "role-axis-transfer"
                    / "meta-llama-llama-3-2-3b-instruct"
                    / "assistant-axis"
                    / f"{label}-role-set"
                    / "completion_mean"
                    / run_name
                )
                results_dir = run_dir / "results"
                results_dir.mkdir(parents=True)
                variant_entries.append({"label": label, "transfer_run_dir": str(run_dir.relative_to(repo_root))})

                with (results_dir / "pairwise_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "method",
                            "layer_spec",
                            "layer_label",
                            "source_dataset",
                            "target_dataset",
                            "train_split",
                            "eval_split",
                            "auroc",
                            "auprc",
                            "balanced_accuracy",
                            "f1",
                            "accuracy",
                            "count",
                            "completed_at",
                        ],
                    )
                    writer.writeheader()
                    for method_idx, method in enumerate(methods):
                        for layer_idx, layer_spec in enumerate(layer_specs):
                            for dataset_idx, dataset in enumerate(datasets):
                                writer.writerow(
                                    {
                                        "method": method,
                                        "layer_spec": layer_spec,
                                        "layer_label": f"L{layer_spec}",
                                        "source_dataset": "__zero_shot__",
                                        "target_dataset": dataset,
                                        "train_split": "",
                                        "eval_split": "test",
                                        "auroc": f"{0.55 + 0.01 * variant_idx + 0.005 * method_idx + 0.002 * layer_idx + 0.001 * dataset_idx:.3f}",
                                        "auprc": "0.0",
                                        "balanced_accuracy": "0.0",
                                        "f1": "0.0",
                                        "accuracy": "0.0",
                                        "count": "100",
                                        "completed_at": "2026-04-02T00:00:00Z",
                                    }
                                )

            manifest_path = repo_root / "comparison_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "artifacts": {"root": "artifacts"},
                        "variants": variant_entries,
                        "dataset_order": datasets,
                        "methods": methods,
                        "layer_specs": layer_specs,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = load_comparison_manifest(manifest_path)
            outputs = run_zero_shot_transfer_comparison(manifest, run_id="role-axis-all-datasets-zeroshot-compare-v1")

            run_root = Path(outputs["run_root"])
            self.assertTrue((run_root / "results" / "zero_shot_auroc_long.csv").exists())
            self.assertTrue((run_root / "results" / "zero_shot_auroc_wide.csv").exists())
            self.assertTrue((run_root / "results" / "plots" / "all_variants_zero_shot_overview.png").exists())
            self.assertEqual(len(outputs["plots"]), 8)

            with (run_root / "results" / "zero_shot_auroc_long.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3 * 2 * 4 * 9)
            self.assertEqual(rows[0]["dataset_label"], "AILiar")
            self.assertIn("ClaimsDef", {row["dataset_label"] for row in rows})


if __name__ == "__main__":
    unittest.main()
