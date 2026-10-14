from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource
from tests.e3_horizon.cap_generate import materialize
from tests.e3_horizon.generate import SHARD_TOKENS, TOKENS, materialize as horizon_materialize
from tests.e3_reference.corpus import FINGERPRINT, VOCAB_SIZE


class CapCorpusTests(unittest.TestCase):
    def test_genuine_one_over_rows_and_first_read_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "corpus"
            first = materialize(root)
            self.assertEqual(first.name, "shard-0000000000000000.ets")
            shards, digest = load_d1_resource(
                root, expected_fingerprint=FINGERPRINT,
                expected_vocab=VOCAB_SIZE, maximum_total_tokens=16387,
            )
            self.assertEqual(tuple(map(len, shards)), (SHARD_TOKENS,) * 8 + (3,))
            self.assertEqual(shards[0][:3], TOKENS[:3])
            rows = ReferenceDataset(
                shards, ReferenceConfig(1, 2, None, 1, True),
                manifest_digest=digest, tokenizer_fingerprint=FINGERPRINT,
                vocab_size=VOCAB_SIZE,
            ).rows
            self.assertEqual(len(rows), 8193)
            self.assertEqual(sum(sum(row.loss_mask) for row in rows), 16386)
            self.assertEqual(rows[0].targets, shards[0][1:3])
            self.assertEqual(rows[-1].targets, shards[-1][1:3])

    def test_two_fresh_materializations_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            materialize(first)
            materialize(second)
            self.assertEqual(
                {path.name: path.read_bytes() for path in first.iterdir()},
                {path.name: path.read_bytes() for path in second.iterdir()},
            )

    def test_one_over_adds_exactly_one_row_to_accepted_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            within = Path(temporary) / "within"
            over = Path(temporary) / "over"
            horizon_materialize(within, 8192)
            materialize(over)
            emitted = []
            for root in (within, over):
                shards, digest = load_d1_resource(
                    root, expected_fingerprint=FINGERPRINT,
                    expected_vocab=VOCAB_SIZE,
                )
                emitted.append(ReferenceDataset(
                    shards, ReferenceConfig(1, 2, None, 1, True),
                    manifest_digest=digest, tokenizer_fingerprint=FINGERPRINT,
                    vocab_size=VOCAB_SIZE,
                ).rows)
            self.assertEqual(len(emitted[0]), 8192)
            self.assertEqual(len(emitted[1]), 8193)
            self.assertEqual(emitted[0], emitted[1][:8192])

    def test_requires_fresh_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "corpus"
            root.mkdir()
            with self.assertRaises(FileExistsError):
                materialize(root)


if __name__ == "__main__":
    unittest.main()
