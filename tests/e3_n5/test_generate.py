from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import ResourceError, load_d1_resource
from tests.e3_n5.generate import (EXACT_ROWS, EXACT_TOKENS, ROWS, SHARD_TOKENS,
                                  TOTAL_TOKENS, materialize)
from tests.e3_n5.evidence import fixture_digest
from tests.e3_reference.corpus import FINGERPRINT, VOCAB_SIZE


class N5CorpusTests(unittest.TestCase):
    def test_exact_final_target_corruption_rejects_source_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "corpus"
            materialize(root)
            original = fixture_digest(root)
            shard = sorted((root / "exact-limit").glob("shard-*.ets"))[-1]
            data = bytearray(shard.read_bytes())
            self.assertEqual(data[-40], 255)
            data[-40] ^= 1  # The last valid target; leave the D1 checksum stale.
            shard.write_bytes(data)
            self.assertNotEqual(fixture_digest(root), original)
            with self.assertRaisesRegex(ResourceError, "corrupt-data"):
                load_d1_resource(root / "exact-limit",
                                 expected_fingerprint=FINGERPRINT,
                                 expected_vocab=VOCAB_SIZE)

    def test_exact_boundary_one_over_and_repeated_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / name for name in ("a", "b")]
            records = [materialize(root) for root in roots]
            self.assertEqual(records[0], records[1])
            self.assertEqual(
                fixture_digest(roots[0]),
                "97a5359116173246a205b980842917889fbd8ac6b52676c92f525494569d541e",
            )
            for relative in sorted(path.relative_to(roots[0]) for path in roots[0].rglob("*")):
                if (roots[0] / relative).is_file():
                    self.assertEqual((roots[0] / relative).read_bytes(),
                                     (roots[1] / relative).read_bytes())
            cases = {case["name"]: case for case in records[0]["cases"]}
            self.assertEqual(set(cases), {"one-over", "exact-limit", "retry-small",
                                          "empty", "singleton"})
            one_over = cases["one-over"]
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
            exact = cases["exact-limit"]
            self.assertEqual(exact["tokens"], EXACT_TOKENS)
            self.assertEqual(exact["expected_rows"], EXACT_ROWS)
            self.assertEqual(exact["active_positions"], EXACT_TOKENS - 1)
            self.assertEqual(len(exact["rows"]), EXACT_ROWS)
            self.assertEqual(exact["rows"][-1],
                             {"inputs": [0, 0], "targets": [255, 0],
                              "mask": [True, False]})
            exact_shards, exact_digest = load_d1_resource(
                roots[0] / "exact-limit", expected_fingerprint=FINGERPRINT,
                expected_vocab=VOCAB_SIZE,
            )
            self.assertEqual(tuple(map(len, exact_shards)), (SHARD_TOKENS,) * 8)
            exact_rows = ReferenceDataset(
                exact_shards, ReferenceConfig(1, 2, None, 1, True),
                manifest_digest=exact_digest, tokenizer_fingerprint=FINGERPRINT,
                vocab_size=VOCAB_SIZE,
            ).rows
            self.assertEqual(len(exact_rows), EXACT_ROWS)
            self.assertTrue(all(row.loss_mask == (True, True)
                                for row in exact_rows[:-1]))
            self.assertEqual(exact_rows[-1].loss_mask, (True, False))

    def test_zero_row_resources_and_exhausted_nonzero_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "corpus"
            records = materialize(root)
            cases = {case["name"]: case for case in records["cases"]}
            config = ReferenceConfig(1, 2, None, 1, True)
            for name, expected_tokens, expected_shards in (
                ("empty", 0, ()), ("singleton", 1, (1,))
            ):
                case = cases[name]
                self.assertEqual((case["tokens"], case["expected_rows"],
                                  case["active_positions"], case["rows"]),
                                 (expected_tokens, 0, 0, []))
                shards, digest = load_d1_resource(
                    root / name, expected_fingerprint=FINGERPRINT,
                    expected_vocab=VOCAB_SIZE,
                )
                self.assertEqual(tuple(map(len, shards)), expected_shards)
                dataset = ReferenceDataset(
                    shards, config, manifest_digest=digest,
                    tokenizer_fingerprint=FINGERPRINT, vocab_size=VOCAB_SIZE,
                )
                initial = dataset.snapshot()
                self.assertIsNone(dataset.next_batch())
                self.assertIsNone(dataset.next_batch())
                self.assertEqual(dataset.snapshot(), initial)

            shards, digest = load_d1_resource(
                root / "retry-small", expected_fingerprint=FINGERPRINT,
                expected_vocab=VOCAB_SIZE,
            )
            dataset = ReferenceDataset(
                shards, config, manifest_digest=digest,
                tokenizer_fingerprint=FINGERPRINT, vocab_size=VOCAB_SIZE,
            )
            initial = dataset.snapshot()
            self.assertEqual(len(dataset.rows), 3)
            for _ in range(3):
                self.assertIsNotNone(dataset.next_batch())
            exhausted = dataset.snapshot()
            self.assertNotEqual(initial, exhausted)
            self.assertIsNone(dataset.next_batch())
            self.assertEqual(dataset.snapshot(), exhausted)

    def test_zero_row_resource_corruption_rejects_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "corpus"
            materialize(root)
            for name, filename in (("empty", "manifest.etm"),
                                   ("singleton", "shard-0000000000000000.ets")):
                path = root / name / filename
                data = bytearray(path.read_bytes())
                data[-1] ^= 1
                path.write_bytes(data)
                with self.assertRaisesRegex(ResourceError, "corrupt-data"):
                    load_d1_resource(root / name,
                                     expected_fingerprint=FINGERPRINT,
                                     expected_vocab=VOCAB_SIZE)


if __name__ == "__main__":
    unittest.main()
