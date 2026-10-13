from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.e3_horizon.edge_accuracy import CASES, expected_accuracy_word, verify_observation
from tests.e3_horizon.edge_generate import materialize


class EdgeAccuracyTests(unittest.TestCase):
    def test_independent_selection_and_fourth_output_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            oracle_lines = []
            cursor_lines = []
            for case in CASES:
                cursor, rows = materialize(Path(temporary) / case, case)
                oracle_lines.append(f"{case}\t{cursor.hex()}\t{rows!r}")
                cursor_lines.append(f"E3-EDGE-CURSOR {case} {cursor.hex()}")
            self.assertEqual(expected_accuracy_word(rows), 1065353216)
            observed = "\n".join(cursor_lines + [
                "E3-EDGE-ACCURACY one-token-tail 1065353216",
                "E3-EDGE-PRIVATE-PASS cases=4 zero-weight=3 one-token-tail=1",
            ]) + "\n"
            oracle = "\n".join(oracle_lines) + "\n"
            verify_observation(oracle, observed)
            # Change only the fourth I2 word; all cursors and other lines stay.
            mutant = observed.replace("one-token-tail 1065353216",
                                      "one-token-tail 0")
            with self.assertRaisesRegex(ValueError, "fourth I2 accuracy"):
                verify_observation(oracle, mutant)
            # A changed authentic target invalidates the selected M3 winner.
            changed = oracle.replace("(14, 0), (True, False)",
                                     "(15, 0), (True, False)")
            self.assertNotEqual(changed, oracle)
            with self.assertRaisesRegex(ValueError, "D2 rows differ"):
                verify_observation(changed, observed)


if __name__ == "__main__":
    unittest.main()
