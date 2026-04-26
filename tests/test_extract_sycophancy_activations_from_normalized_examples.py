from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.extract_sycophancy_activations_from_normalized_examples import load_normalized_examples


class ExtractSycophancyActivationsFromNormalizedExamplesTest(unittest.TestCase):
    def test_load_normalized_examples_reconstructs_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "normalized_examples.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "ids": "sample-1",
                        "pair_id": "pair-1",
                        "dataset_source": "oeq-framing-human",
                        "source_repo_id": "owner/oeq-framing-human",
                        "source_split": "train",
                        "source_record_id": "row-1",
                        "adapter_name": "oeq_prepared",
                        "label_name": "sycophantic",
                        "label": 1,
                        "category": "framing",
                        "prompt_messages": [{"role": "user", "content": "Question?"}],
                        "assistant_output": "Agreeable answer.",
                        "extra_field": "kept",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            examples = load_normalized_examples(path)

            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0].ids, "sample-1")
            self.assertEqual(examples[0].label, 1)
            self.assertEqual(examples[0].prompt_messages[0]["content"], "Question?")
            self.assertEqual(examples[0].assistant_output, "Agreeable answer.")
            self.assertEqual(examples[0].metadata["extra_field"], "kept")

    def test_load_normalized_examples_rejects_missing_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "normalized_examples.jsonl"
            path.write_text(json.dumps({"ids": "bad"}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required fields"):
                load_normalized_examples(path)


if __name__ == "__main__":
    unittest.main()
