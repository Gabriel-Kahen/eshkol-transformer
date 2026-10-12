from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource
from tests.e3_n5.generate import ROWS, SHARD_TOKENS, TOTAL_TOKENS, materialize
from tests.e3_n5.evidence import fixture_digest
from tests.e3_reference.corpus import FINGERPRINT, VOCAB_SIZE


class N5CorpusTests(unittest.TestCase):
    def test_one_over_and_repeated_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / name for name in ("a", "b")]
            records = [materialize(root) for root in roots]
            self.assertEqual(records[0], records[1])
            self.assertEqual(
                fixture_digest(roots[0]),
                "54300ebc7fed784858cf4d5749ba5313b1a4347690c4391c80bdf611dc093935",
            )
            for relative in sorted(path.relative_to(roots[0]) for path in roots[0].rglob("*")):
                if (roots[0] / relative).is_file():
                    self.assertEqual((roots[0] / relative).read_bytes(),
                                     (roots[1] / relative).read_bytes())
            one_over = records[0]["cases"][0]
            self.assertEqual(one_over["expected_rows"], ROWS)
            self.assertEqual(one_over["active_positions"], 2 * ROWS)
            self.assertEqual(one_over["tokens"], TOTAL_TOKENS)
            self.assertEqual(len(one_over["rows"]), ROWS)
            self.assertEqual(one_over["rows"][0],
                             {"inputs": [3, 197], "targets": [197, 0],
                              "mask": [True, True]})
            self.assertEqual([item["bytes"] for item in one_over["files"]][1:],
                             [80 + len(FINGERPRINT) + 8 * SHARD_TOKENS + 32] * 8
                             + [80 + len(FINGERPRINT) + 8 * 3 + 32])
            shards, digest = load_d1_resource(
                roots[0] / "one-over", expected_fingerprint=FINGERPRINT,
                expected_vocab=VOCAB_SIZE,
            )
            self.assertEqual(tuple(map(len, shards)), (SHARD_TOKENS,) * 8 + (3,))
            rows = ReferenceDataset(
                shards, ReferenceConfig(1, 2, None, 1, True),
                manifest_digest=digest, tokenizer_fingerprint=FINGERPRINT,
                vocab_size=VOCAB_SIZE,
            ).rows
            self.assertEqual(len(rows), ROWS)
            self.assertTrue(all(row.loss_mask == (True, True) for row in rows))


if __name__ == "__main__":
    unittest.main()
