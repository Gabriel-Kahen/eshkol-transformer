from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource
from tests.e3_private.n5_corpus import (
    RAW_TOKENS, ROWS, SHARD_TOKENS, materialize, materialize_boundary,
)
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

    def test_independent_boundary_matches_one_real_advance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            one_over = materialize(Path(temporary) / "one-over")
            boundary = materialize_boundary(Path(temporary) / "boundary")
            full, full_digest = load_d1_resource(
                one_over, expected_fingerprint=FINGERPRINT,
                expected_vocab=VOCAB_SIZE, maximum_total_tokens=RAW_TOKENS,
            )
            sliced, sliced_digest = load_d1_resource(
                boundary, expected_fingerprint=FINGERPRINT,
                expected_vocab=VOCAB_SIZE, maximum_total_tokens=RAW_TOKENS - 2,
            )
            self.assertEqual(tuple(token for shard in full for token in shard)[2:],
                             tuple(token for shard in sliced for token in shard))
            first = ReferenceDataset(full, ReferenceConfig(1, 2, None, 1, True),
                                     manifest_digest=full_digest,
                                     tokenizer_fingerprint=FINGERPRINT,
                                     vocab_size=VOCAB_SIZE)
            second = ReferenceDataset(sliced, ReferenceConfig(1, 2, None, 1, True),
                                      manifest_digest=sliced_digest,
                                      tokenizer_fingerprint=FINGERPRINT,
                                      vocab_size=VOCAB_SIZE)
            self.assertEqual(len(second.rows), ROWS - 1)
            self.assertEqual(first.rows[1:], second.rows)


if __name__ == "__main__":
    unittest.main()
