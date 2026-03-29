from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deception_honesty_axis.indexes import (
    append_index_records,
    ensure_corpus_layout,
    load_index,
    next_shard_path,
)


class IndexesTest(unittest.TestCase):
    def test_next_shard_path_increments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_root = Path(tmp_dir)
            ensure_corpus_layout(corpus_root)
            first = next_shard_path(corpus_root, "rollouts", "spy", ".jsonl")
            first.touch()
            second = next_shard_path(corpus_root, "rollouts", "spy", ".jsonl")
            self.assertEqual(first.name, "part-00001.jsonl")
            self.assertEqual(second.name, "part-00002.jsonl")

    def test_append_and_load_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_root = Path(tmp_dir)
            ensure_corpus_layout(corpus_root)
            append_index_records(
                corpus_root,
                "rollouts",
                [
                    {"item_id": "spy__p00__q000", "role": "spy", "shard_path": "rollouts/role=spy/part-00001.jsonl"},
                    {"item_id": "default__p00__q000", "role": "default", "shard_path": "rollouts/role=default/part-00001.jsonl"},
                ],
            )
            loaded = load_index(corpus_root, "rollouts")
            self.assertEqual(set(loaded), {"spy__p00__q000", "default__p00__q000"})
            self.assertEqual(loaded["spy__p00__q000"]["role"], "spy")


if __name__ == "__main__":
    unittest.main()
