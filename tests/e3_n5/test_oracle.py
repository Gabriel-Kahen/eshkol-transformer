"""Pinned development-only oracle and compiled-transcript comparator checks."""

from pathlib import Path
import tempfile
import unittest

from tests.e3_n5.oracle import EDGE_NAMES, build, check_run
from tests.e3_n5.prepare import materialize


class N5OracleTests(unittest.TestCase):
    def test_one_token_oracle_and_rejected_transcript_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            oracle = build(materialize(Path(temporary) / "corpus"))
        metrics = [int(word, 16) for word in oracle["metrics_f32"]] + oracle["counters"]
        lines = ["OUTPUTS " + " ".join(map(str, metrics))]
        for name in EDGE_NAMES:
            cursor = bytes.fromhex(oracle["cursor_hex"][name])
            lines.append("CURSOR " + name + " " + " ".join(map(str, cursor)))
        lines.append("E3-N5-EDGE-PASS one=1 rejected=3 outputs=6 cursors=4")
        valid = "\n".join(lines) + "\n"
        check_run(oracle, valid)
        changed = lines.copy()
        changed[0] = changed[0].replace(str(metrics[0]), str(metrics[0] + 10000), 1)
        with self.assertRaisesRegex(ValueError, "tolerance"):
            check_run(oracle, "\n".join(changed) + "\n")
        changed = lines.copy()
        prefix, last = changed[3].rsplit(" ", 1)
        changed[3] = prefix + " " + str(int(last) ^ 1)
        with self.assertRaisesRegex(ValueError, "cursor differs"):
            check_run(oracle, "\n".join(changed) + "\n")
        with self.assertRaisesRegex(ValueError, "transcript shape"):
            check_run(oracle, "\n".join(lines[:-1]) + "\n")


if __name__ == "__main__":
    unittest.main()
