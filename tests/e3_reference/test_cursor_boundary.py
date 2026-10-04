"""Boundary oracle tests are independent of the compiled E3 witness."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tests.d2.reference import decode_reference_cursor
from tests.e3_reference.cursor_boundary import ONE_OVER_TOKENS, materialize
from tests.e3_reference.reference import _synthetic_case
from tests.e3_reference.corpus import FINGERPRINT
from tests.m3.reference import parameters


class CursorBoundaryReferenceTests(unittest.TestCase):
    def test_authentic_empty_singleton_eos_and_last_row(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-boundary-") as temporary:
            root = Path(temporary) / "fixture"
            cases = materialize(root)
            self.assertEqual(tuple(cases),
                             ("empty", "singleton", "entry-eos", "final-suffix", "one-over"))
            for name in ("empty", "singleton", "entry-eos"):
                case = cases[name]
                self.assertEqual(case["batches"], [])
                cursor = decode_reference_cursor(case["saved_cursor"])
                self.assertEqual(cursor.next_ordinal, case["total_rows"])
            self.assertEqual(cases["empty"]["total_rows"], 0)
            self.assertEqual(cases["singleton"]["total_rows"], 0)
            self.assertEqual(cases["entry-eos"]["total_rows"], 1)
            suffix = cases["final-suffix"]
            self.assertEqual(suffix["total_rows"], 3)
            self.assertEqual(decode_reference_cursor(suffix["saved_cursor"]).next_ordinal, 2)
            self.assertEqual(suffix["batches"], [
                {"inputs": [7, 0], "targets": [7, 0], "mask": [True, False]},
            ])
            model, _ = parameters(1729)
            expected, _, _ = _synthetic_case(suffix["batches"], model)
            self.assertEqual(expected["final"]["counters"], [1, 1])
            self.assertEqual(expected["final"]["metrics_f32"],
                             ["40b1b2bd", "3f800000", "4381039a", "00000000"])
            over = cases["one-over"]
            self.assertEqual(len(ONE_OVER_TOKENS), 16_387)
            self.assertLess(len(ONE_OVER_TOKENS), 65_536)
            self.assertEqual(sum(1 for _ in range(0, len(ONE_OVER_TOKENS) - 1, 2)),
                             8_193)
            self.assertEqual(over["total_rows"], 8_193)
            self.assertEqual(decode_reference_cursor(over["saved_cursor"]).next_ordinal, 0)
            self.assertEqual(over["batches"], [])
            self.assertEqual((over["directory"] / "shard-0000000000000000.ets").stat().st_size,
                             80 + len(FINGERPRINT.encode()) + 8 * 16_387 + 32)
            self.assertLess((over["directory"] / "shard-0000000000000000.ets").stat().st_size,
                            1_048_576)

    def test_fresh_path_required(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-boundary-") as temporary:
            root = Path(temporary) / "fixture"
            materialize(root)
            with self.assertRaises(FileExistsError):
                materialize(root)


if __name__ == "__main__":
    unittest.main()
