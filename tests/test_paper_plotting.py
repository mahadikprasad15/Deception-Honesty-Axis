from __future__ import annotations

import unittest

from deception_honesty_axis.paper_plotting import (
    canonical_axis_display_name,
    canonical_dataset_label,
    canonical_role_side,
    infer_artifact_kind,
    infer_axis_key_from_path,
    parse_pc_count_from_path,
    resolve_run_root_from_artifact_path,
)


class PaperPlottingTest(unittest.TestCase):
    def test_canonical_axis_display_name_maps_quantity_to_deception(self) -> None:
        self.assertEqual(canonical_axis_display_name("quantity_axis_v2"), "Deception Axis")
        self.assertEqual(canonical_axis_display_name("quantity-axis-v2-cumulative-pc-sweep"), "Deception Axis")

    def test_canonical_axis_display_name_maps_sycophancy_aliases(self) -> None:
        self.assertEqual(canonical_axis_display_name("sycophancy_pilot_v1"), "Sycophancy Axis")
        self.assertEqual(canonical_axis_display_name("sycophancy_pilot_v2"), "Sycophancy Axis")

    def test_canonical_dataset_label_uses_paper_friendly_names(self) -> None:
        self.assertEqual(canonical_dataset_label("Deception-AILiar-completion"), "AI Liar")
        self.assertEqual(canonical_dataset_label("Deception-HarmPressureChoice-completion"), "Harm Pressure Choice")

    def test_canonical_role_side_recognizes_manifest_variants(self) -> None:
        self.assertEqual(canonical_role_side({"role_side": "deceptive"}), "harmful")
        self.assertEqual(canonical_role_side({"side": "non_sycophantic"}), "harmless")
        self.assertEqual(canonical_role_side({"anchor_only": True}), "anchor")

    def test_inventory_helpers_classify_hf_artifact_paths(self) -> None:
        path_value = (
            "runs/activation-row-transfer-pc-projection/meta-llama-llama-3-2-3b-instruct/"
            "deception/deception/quantity-axis-v2-cumulative-pc-sweep/mean-response-layer-14-pcs-3/"
            "20260421T000000Z-abcdef/results/pairwise_metrics.csv"
        )
        self.assertEqual(infer_artifact_kind(path_value), "pc_projection_transfer")
        self.assertEqual(infer_axis_key_from_path(path_value), "deception_axis")
        self.assertEqual(parse_pc_count_from_path(path_value), 3)
        self.assertEqual(
            resolve_run_root_from_artifact_path(path_value, "pc_projection_transfer"),
            path_value.removesuffix("/results/pairwise_metrics.csv"),
        )


if __name__ == "__main__":
    unittest.main()
