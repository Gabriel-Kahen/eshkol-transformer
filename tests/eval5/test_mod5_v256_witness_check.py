"""Build-free checks for the live SCORE comparator and tolerance boundary."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tests.eval5 import check_mod5_v256_witness as checker
from tests.eval5.mod5_v256_fixture import build


class Mod5V256WitnessCheckTests(unittest.TestCase):
    def test_reference_line_and_mutants(self):
        with TemporaryDirectory(prefix="eval5-v256-check-") as temp:
            root = Path(temp) / "fixture"
            record = build(root)
            score = record["score_reference"]
            metrics = [score["lm"][key] for key in
                       ("total_nll", "mean_nll", "perplexity", "bits_per_byte")]
            metrics.append(score["mc"]["accuracy"])
            values = metrics + [2, 4] + checker.expected_probes(root)
            audit = "AUDIT 0 2 0 0 0 14 0 0 0\n"
            line = audit + "SCORE " + " ".join(map(str, values)) + "\n"
            self.assertEqual(checker.check(root, line), 0.0)
            bad = values.copy()
            bad[0] += 1.0
            with self.assertRaises(ValueError):
                checker.check(root, audit + "SCORE " + " ".join(map(str, bad)) + "\n")
            bad = values.copy()
            bad[7] += 0.1
            with self.assertRaises(ValueError):
                checker.check(root, audit + "SCORE " + " ".join(map(str, bad)) + "\n")
            with self.assertRaises(ValueError):
                checker.check(root, audit + "SCORE bad\n")
            with self.assertRaises(ValueError):
                checker.check(root, audit.replace(" 14 0 ", " 14 1 ") + line.splitlines()[1] + "\n")
            with self.assertRaises(ValueError):
                checker.check(root, line.splitlines()[1] + "\n")


if __name__ == "__main__":
    unittest.main()
