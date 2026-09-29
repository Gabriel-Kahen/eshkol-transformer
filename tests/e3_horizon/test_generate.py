from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource
from tests.e3_horizon.generate import HORIZONS, SHARD_TOKENS, materialize
from tests.e3_reference.corpus import FINGERPRINT, VOCAB_SIZE


class HorizonCorpusTests(unittest.TestCase):
    def test_exact_rows_and_late_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for horizon in HORIZONS:
                root = Path(temporary) / str(horizon)
                late = materialize(root, horizon)
                self.assertEqual(late.parent, root)
                self.assertEqual(
                    late.name, f"shard-{2 * horizon // SHARD_TOKENS:016d}.ets"
                )
                shards, digest = load_d1_resource(
                    root, expected_fingerprint=FINGERPRINT,
                    expected_vocab=VOCAB_SIZE,
                )
                self.assertEqual(
                    tuple(map(len, shards)),
                    (SHARD_TOKENS,) * (2 * horizon // SHARD_TOKENS) + (1,),
                )
                rows = ReferenceDataset(
                    shards, ReferenceConfig(1, 2, None, 1, True),
                    manifest_digest=digest, tokenizer_fingerprint=FINGERPRINT,
                    vocab_size=VOCAB_SIZE,
                ).rows
                self.assertEqual(len(rows), horizon)
                self.assertTrue(all(row.loss_mask == (True, True) for row in rows))
                self.assertEqual(rows[-1].targets[-1], shards[-1][0])

    def test_repeated_generation_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for horizon in HORIZONS:
                first = Path(temporary) / f"first-{horizon}"
                second = Path(temporary) / f"second-{horizon}"
                materialize(first, horizon)
                materialize(second, horizon)
                self.assertEqual(
                    {path.name: path.read_bytes() for path in first.iterdir()},
                    {path.name: path.read_bytes() for path in second.iterdir()},
                )

    def test_reject_unadmitted_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "unsupported E3 horizon"):
                materialize(Path(temporary) / "other", 8193)


if __name__ == "__main__":
    unittest.main()
