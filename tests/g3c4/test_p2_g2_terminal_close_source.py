"""Source-only negatives; compiled same-call proof is a later gate."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-g3c4-p2g2-terminal-close.py"
spec = importlib.util.spec_from_file_location("terminal_close_checker", CHECKER)
assert spec and spec.loader
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class TerminalCloseSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = checker.SOURCE.read_text(encoding="utf-8")
        cls.closure = checker.CLOSURE.read_text(encoding="utf-8").splitlines()
        cls.base = checker.BASE.read_text(encoding="utf-8").splitlines()

    def validate(self, source: str | None = None,
                 closure: list[str] | None = None) -> None:
        checker.validate(source if source is not None else self.source,
                         closure if closure is not None else self.closure,
                         self.base)

    def test_current_source_and_exact_closure(self) -> None:
        self.validate()

    def test_wrong_first_raw_header_extent_rejected(self) -> None:
        self.assertIn("first-raw 9", self.source)
        with self.assertRaises(AssertionError):
            self.validate(self.source.replace("first-raw 9", "first-raw 8", 1))

    def test_wrong_clone_copy_extent_rejected(self) -> None:
        self.assertIn("bytes (+ 8 (* count 8))", self.source)
        with self.assertRaises(AssertionError):
            self.validate(self.source.replace("bytes (+ 8 (* count 8))",
                                              "bytes (+ 16 (* count 8))", 1))

    def test_tail_allocation_rejected(self) -> None:
        marker = "(vector-set! barrier 0 #t)"
        self.assertIn(marker, self.source)
        with self.assertRaises(AssertionError):
            self.validate(self.source.replace(marker,
                                              marker + " (make-bytevector 1 0)",
                                              1))

    def test_tail_callback_rejected(self) -> None:
        marker = "(vector-set! barrier 0 #t)"
        with self.assertRaises(AssertionError):
            self.validate(self.source.replace(marker,
                                              marker + " (t1-private-g3-decode-raw-into! tokenizer first-stage first-raw)",
                                              1))

    def test_barrier_must_precede_first_publication(self) -> None:
        marker = "(vector-set! barrier 0 #t)"
        self.assertIn(marker, self.source)
        with self.assertRaises(AssertionError):
            self.validate(self.source.replace(
                marker,
                "(vector-set! row 13 'live)\n                " + marker,
                1))

    def test_missing_second_bridge_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            self.validate(self.source.replace("g3c4-p2g2-native-second\n",
                                              "g3c4-p2g2-native-forged\n", 1))

    def test_equal_length_substitute_without_original_anchor_rejected(self) -> None:
        original = "(eq? (vector-ref record 11) (vector-ref row 3))"
        self.assertIn(original, self.source)
        with self.assertRaises(AssertionError):
            # The source checker pins this original-carrier predicate below.
            self.validate(self.source.replace(original, "#t", 1))

    def test_paired_substitution_without_local_original_rejected(self) -> None:
        original = "(eq? (vector-ref row 3) (vector-ref originals 4))"
        self.assertIn(original, self.source)
        with self.assertRaises(AssertionError):
            self.validate(self.source.replace(original, "#t", 1))

    def test_surviving_anchor_pins_logical_count_and_ids(self) -> None:
        for original in (
            "(vector-set! anchor 14 count)",
            "(vector-set! anchor 15 selected-first)",
            "(vector-set! anchor 16 selected-second)",
            "(= (vector-ref row 14) (vector-ref anchor 14))",
            "(= (vector-ref row 15) (vector-ref anchor 15))",
            "(= (vector-ref row 16) (vector-ref anchor 16))",
        ):
            with self.subTest(original=original):
                self.assertIn(original, self.source)
                with self.assertRaises(AssertionError):
                    self.validate(self.source.replace(original, "#t", 1))

    def test_closure_reordering_and_extra_member_rejected(self) -> None:
        reordered = self.closure.copy()
        reordered[-1], reordered[-2] = reordered[-2], reordered[-1]
        with self.assertRaises(AssertionError):
            self.validate(closure=reordered)
        with self.assertRaises(AssertionError):
            self.validate(closure=self.closure + ["native/extra.esk"])


if __name__ == "__main__":
    unittest.main()
