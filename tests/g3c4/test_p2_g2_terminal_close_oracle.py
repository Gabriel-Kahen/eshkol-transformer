"""Parser and near-miss negatives for the future compiled terminal transcript."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

MODULE = Path(__file__).with_name("check_p2_g2_terminal_close_oracle.py")
spec = importlib.util.spec_from_file_location("terminal_oracle", MODULE)
assert spec and spec.loader
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)

FIRST_BITS = ["c1200000"] * 256
SECOND_BITS = ["c1200000"] * 256
FIRST_BITS[3] = SECOND_BITS[5] = "41200000"


def transcript() -> list[str]:
    lines = []
    for mode, seed in oracle.MODES.items():
        for prompt in sorted(oracle.PROMPTS):
            for branch in ("first-eos", "non-eos", "second-eos"):
                if branch == "second-eos" and (mode, prompt) != (
                        "categorical", (7, 11)):
                    continue
                first_before = (1, seed, 0, 0)
                first, first_after = oracle.successor(
                    first_before, mode,
                    [oracle.reference.word(int(x, 16)) for x in FIRST_BITS])
                second, second_after = oracle.successor(
                    first_after, mode,
                    [oracle.reference.word(int(x, 16)) for x in SECOND_BITS])
                count = 1 if branch == "first-eos" else 2
                after = first_after if count == 1 else second_after
                eos = first if count == 1 else (
                    second if branch == "second-eos" else -1)
                def row(draw, selected, before, successor, bits):
                    return " ".join(map(str, ("TERM_ORACLE", mode, *prompt,
                                             draw, selected, *before,
                                             *successor, *bits)))
                lines.append(row("first", first, first_before, first_after,
                                 FIRST_BITS))
                if count == 2:
                    lines.append(row("second", second, first_after, after,
                                     SECOND_BITS))
                lines.append(" ".join(map(str, ("TERM_CACHE", mode, *prompt,
                                                 count, count + 2, first,
                                                 second if count == 2 else 0,
                                                 eos))))
    return lines


class TerminalOracleTests(unittest.TestCase):
    def test_complete_matrix(self) -> None:
        self.assertEqual(oracle.check_lines(transcript()), 9)

    def test_wrong_selected_id(self) -> None:
        rows = transcript()
        words = rows[0].split()
        words[5] = "42"
        rows[0] = " ".join(words)
        with self.assertRaises(AssertionError):
            oracle.check_lines(rows)

    def test_extra_first_eos_draw(self) -> None:
        rows = transcript()
        rows.insert(1, rows[2])
        with self.assertRaises(AssertionError):
            oracle.check_lines(rows)

    def test_wrong_rng_successor(self) -> None:
        rows = transcript()
        categorical = next(i for i, line in enumerate(rows)
                           if line.startswith("TERM_ORACLE categorical"))
        words = rows[categorical].split()
        words[12] = "12"
        rows[categorical] = " ".join(words)
        with self.assertRaises(AssertionError):
            oracle.check_lines(rows)

    def test_wrong_cache_length(self) -> None:
        rows = transcript()
        words = rows[1].split()
        words[5] = "4"
        rows[1] = " ".join(words)
        with self.assertRaises(AssertionError):
            oracle.check_lines(rows)

    def test_missing_branch_and_duplicate(self) -> None:
        rows = transcript()
        with self.assertRaises(AssertionError):
            oracle.check_lines(rows[:-3])
        with self.assertRaises(AssertionError):
            oracle.check_lines(rows + rows[:2])

    def test_false_second_eos_and_truncated_logits(self) -> None:
        rows = transcript()
        eos = next(i for i, line in enumerate(rows)
                   if line.startswith("TERM_CACHE categorical 7 11 2 4")
                   and line.endswith(" 5"))
        words = rows[eos].split()
        words[-1] = words[6]
        rows[eos] = " ".join(words)
        with self.assertRaises(AssertionError):
            oracle.check_lines(rows)
        rows = transcript()
        rows[0] = " ".join(rows[0].split()[:-1])
        with self.assertRaises(AssertionError):
            oracle.check_lines(rows)


if __name__ == "__main__":
    unittest.main()
