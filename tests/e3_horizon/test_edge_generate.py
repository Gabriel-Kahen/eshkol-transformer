from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.d2.reference import ReferenceConfig, ReferenceDataset, decode_reference_cursor
from tests.d2.test_resources import load_d1_resource
from tests.e3_horizon.edge_generate import CASES, materialize
from tests.e3_reference.corpus import FINGERPRINT, VOCAB_SIZE


class EdgeCorpusTests(unittest.TestCase):
    def test_emitted_rows_cursors_and_repeat_bytes(self) -> None:
        expected = {
            "empty": (),
            "singleton": (),
            "entry-eos": (((3, 197), (197, 0), (True, True)),),
            "one-token-tail": (
                ((3, 197), (197, 0), (True, True)),
                ((0, 0), (255, 0), (True, False)),
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            for case in CASES:
                first = Path(temporary) / f"a-{case}"
                second = Path(temporary) / f"b-{case}"
                cursor, rows = materialize(first, case)
                duplicate_cursor, duplicate_rows = materialize(second, case)
                self.assertEqual(rows, expected[case])
                self.assertEqual((cursor, rows), (duplicate_cursor, duplicate_rows))
                self.assertEqual(
                    {path.name: path.read_bytes() for path in first.iterdir()},
                    {path.name: path.read_bytes() for path in second.iterdir()},
                )
                shards, digest = load_d1_resource(
                    first, expected_fingerprint=FINGERPRINT, expected_vocab=VOCAB_SIZE
                )
                reference = ReferenceDataset(
                    shards, ReferenceConfig(1, 2, None, 1, True,
                                            maximum_batch_bytes=34),
                    manifest_digest=digest, tokenizer_fingerprint=FINGERPRINT,
                    vocab_size=VOCAB_SIZE,
                )
                parsed = decode_reference_cursor(cursor)
                self.assertEqual(parsed.total_rows, len(rows))
                self.assertEqual(parsed.next_ordinal,
                                 1 if case == "entry-eos" else 0)
                self.assertEqual(parsed.manifest_digest, digest)
                if rows:
                    self.assertEqual(reference.rows[-1].loss_mask,
                                     (True, False) if case == "one-token-tail"
                                     else (True, True))

    def test_unknown_case_rejected_without_resource(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "unknown"
            with self.assertRaisesRegex(ValueError, "unsupported E3 edge case"):
                materialize(root, "unknown")
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
