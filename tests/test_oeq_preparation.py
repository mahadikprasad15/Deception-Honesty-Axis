from __future__ import annotations

import unittest

from deception_honesty_axis.oeq_preparation import (
    OEQ_TASK_LABEL_FIELDS,
    is_oeq_clean_negative,
    normalize_oeq_binary_flag,
    normalize_oeq_rows,
    prepare_oeq_task_rows,
    summarize_oeq_rows,
)


class OEQPreparationTest(unittest.TestCase):
    def test_normalize_oeq_binary_flag_accepts_expected_values(self) -> None:
        self.assertEqual(normalize_oeq_binary_flag("1.0"), 1)
        self.assertEqual(normalize_oeq_binary_flag("0"), 0)
        self.assertIsNone(normalize_oeq_binary_flag(""))

    def test_normalize_oeq_rows_computes_sycophantic_any(self) -> None:
        rows = [
            {
                "Unnamed: 0": "1",
                "prompt": "Prompt",
                "human": "Answer",
                "source": "advisorqa",
                "validation_human": "1.0",
                "indirectness_human": "0",
                "framing_human": "",
            }
        ]

        normalized = normalize_oeq_rows(rows)

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["validation_human"], 1)
        self.assertEqual(normalized[0]["indirectness_human"], 0)
        self.assertIsNone(normalized[0]["framing_human"])
        self.assertEqual(normalized[0]["sycophantic_any"], 1)

    def test_prepare_oeq_task_rows_uses_target_positives_and_clean_negatives(self) -> None:
        rows = normalize_oeq_rows(
            [
                {
                    "Unnamed: 0": "1",
                    "prompt": "Prompt 1",
                    "human": "Answer 1",
                    "source": "advisorqa",
                    "validation_human": "1.0",
                    "indirectness_human": "0",
                    "framing_human": "0.0",
                },
                {
                    "Unnamed: 0": "2",
                    "prompt": "Prompt 2",
                    "human": "Answer 2",
                    "source": "advisorqa",
                    "validation_human": "1.0",
                    "indirectness_human": "1",
                    "framing_human": "0.0",
                },
                {
                    "Unnamed: 0": "3",
                    "prompt": "Prompt 3",
                    "human": "Answer 3",
                    "source": "reddit",
                    "validation_human": "0.0",
                    "indirectness_human": "0",
                    "framing_human": "0.0",
                },
                {
                    "Unnamed: 0": "4",
                    "prompt": "Prompt 4",
                    "human": "Answer 4",
                    "source": "reddit",
                    "validation_human": "0.0",
                    "indirectness_human": "0",
                    "framing_human": "0.0",
                },
            ]
        )

        prepared_rows, summary = prepare_oeq_task_rows(
            rows,
            target_label="validation_human",
            positive_count=2,
            negative_count=2,
            sample_seed=42,
        )

        self.assertEqual(len(prepared_rows), 4)
        self.assertEqual(summary["label_counts"], {0: 2, 1: 2})
        positives = [row for row in prepared_rows if int(row["label"]) == 1]
        negatives = [row for row in prepared_rows if int(row["label"]) == 0]
        self.assertTrue(all(row["validation_human"] == 1 for row in positives))
        self.assertTrue(all(row["label_name"] == "validation_human" for row in positives))
        self.assertTrue(all(row["label_name"] == "not_validation_human" for row in negatives))
        self.assertTrue(all(
            row["validation_human"] == 0
            and row["indirectness_human"] == 0
            and row["framing_human"] == 0
            for row in negatives
        ))

    def test_summarize_oeq_rows_reports_clean_negative_count(self) -> None:
        rows = normalize_oeq_rows(
            [
                {
                    "Unnamed: 0": "1",
                    "prompt": "Prompt 1",
                    "human": "Answer 1",
                    "source": "advisorqa",
                    "validation_human": "1.0",
                    "indirectness_human": "0",
                    "framing_human": "0.0",
                },
                {
                    "Unnamed: 0": "2",
                    "prompt": "Prompt 2",
                    "human": "Answer 2",
                    "source": "reddit",
                    "validation_human": "0.0",
                    "indirectness_human": "0",
                    "framing_human": "0.0",
                },
            ]
        )

        summary = summarize_oeq_rows(rows)

        self.assertEqual(summary["record_count"], 2)
        self.assertEqual(summary["clean_negative_count"], 1)
        self.assertEqual(summary["sycophantic_any_count"], 1)
        self.assertEqual(summary["label_counts"]["validation_human"]["1"], 1)
        self.assertEqual(summary["source_counts"], {"advisorqa": 1, "reddit": 1})

    def test_is_oeq_clean_negative_requires_all_zero(self) -> None:
        negative_row = {field_name: 0 for field_name in OEQ_TASK_LABEL_FIELDS}
        positive_row = dict(negative_row)
        positive_row["framing_human"] = 1

        self.assertTrue(is_oeq_clean_negative(negative_row))
        self.assertFalse(is_oeq_clean_negative(positive_row))


if __name__ == "__main__":
    unittest.main()
