from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource
from tests.e3_private.n5_corpus import RAW_TOKENS, ROWS, SHARD_TOKENS, materialize
from tests.e3_reference.corpus import FINGERPRINT, VOCAB_SIZE


class N5CorpusTest(unittest.TestCase):
    def test_two_fresh_materializations_and_independent_d2_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = materialize(Path(temporary) / "first")
            second = materialize(Path(temporary) / "second")
            first_files = {path.name: path.read_bytes() for path in first.iterdir()}
            second_files = {path.name: path.read_bytes() for path in second.iterdir()}
            self.assertEqual(first_files, second_files)
            self.assertEqual(len(first_files), 10)  # Nine shards and manifest.
            shards, digest = load_d1_resource(
                first, expected_fingerprint=FINGERPRINT,
                expected_vocab=VOCAB_SIZE, maximum_total_tokens=RAW_TOKENS,
            )
            self.assertEqual(tuple(map(len, shards)),
                             (SHARD_TOKENS,) * 8 + (3,))
            self.assertEqual(sum(map(len, shards)), RAW_TOKENS)
            reference = ReferenceDataset(
                shards, ReferenceConfig(1, 2, None, 1, True),
                manifest_digest=digest, tokenizer_fingerprint=FINGERPRINT,
                vocab_size=VOCAB_SIZE,
            )
            self.assertEqual(len(reference.rows), ROWS)
            self.assertTrue(all(row.loss_mask == (True, True)
                                for row in reference.rows))


if __name__ == "__main__":
    unittest.main()
