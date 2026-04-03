from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from deception_honesty_axis.analysis import write_pca_report_artifacts


class PcaReportsTest(unittest.TestCase):
    def test_write_pca_report_artifacts_generates_tables_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = Path(tmp_dir)
            layers = []
            for layer_idx in range(28):
                layers.append(
                    {
                        "layer": layer_idx,
                        "explained_variance_ratio": [0.5, 0.3, 0.2],
                        "role_projections": {
                            "spy": [1.0 + 0.01 * layer_idx, 2.0 - 0.02 * layer_idx, 3.0],
                            "rogue": [-1.5 + 0.01 * layer_idx, 0.5, -0.25 - 0.01 * layer_idx],
                        },
                        "anchor_projection": [0.4 + 0.01 * layer_idx, -0.2, 0.1 - 0.005 * layer_idx],
                        "contrast_pc_cosines": [0.1, -0.8 + 0.01 * layer_idx, 0.2],
                    }
                )

            outputs = write_pca_report_artifacts(run_root, layers, "default", ["spy", "rogue"])

            self.assertTrue((run_root / "results" / "tables" / "role_pc_projections.csv").exists())
            self.assertTrue((run_root / "results" / "tables" / "pc_rankings.csv").exists())
            self.assertTrue((run_root / "results" / "tables" / "layer_summary.csv").exists())
            self.assertTrue((run_root / "results" / "plots" / "pc1_pc2.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pc1_pc3.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pc2_pc3.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "default_abs_projection_heatmap.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "contrast_cosine_heatmap.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pc1_pc2__L7.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pc1_pc2__L14.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pc1_pc2__L21.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pc1_pc2__L28.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pc1_pc3__L7.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pc1_pc3__L14.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pc1_pc3__L21.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pc1_pc3__L28.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pc2_pc3__L7.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pc2_pc3__L14.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pc2_pc3__L21.png").exists())
            self.assertTrue((run_root / "results" / "plots" / "pc2_pc3__L28.png").exists())
            self.assertIn("role_projection_table", outputs)
            self.assertIn("pc1_pc2_selected_layers", outputs["plots"])
            self.assertIn("pc1_pc3_selected_layers", outputs["plots"])
            self.assertIn("pc2_pc3_selected_layers", outputs["plots"])

            with (run_root / "results" / "tables" / "pc_rankings.csv").open("r", encoding="utf-8", newline="") as handle:
                ranking_rows = list(csv.DictReader(handle))

            layer0_pc1 = next(row for row in ranking_rows if row["layer"] == "0" and row["pc_index"] == "1")
            layer0_pc2 = next(row for row in ranking_rows if row["layer"] == "0" and row["pc_index"] == "2")
            layer0_pc3 = next(row for row in ranking_rows if row["layer"] == "0" and row["pc_index"] == "3")

            self.assertEqual(layer0_pc1["default_abs_rank"], "1")
            self.assertEqual(layer0_pc2["default_abs_rank"], "2")
            self.assertEqual(layer0_pc3["default_abs_rank"], "3")
            self.assertEqual(layer0_pc2["contrast_abs_rank"], "1")


if __name__ == "__main__":
    unittest.main()
