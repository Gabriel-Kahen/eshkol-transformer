from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.e3_private.parse_edge_observation import parse
from tests.e3_reference.edge_corpus import materialize


def failure_line(name: str, cursor: str, *, after: str = "0,0,0,0,0,0") -> bytes:
    return (f"E3EDGE\t{name}\tinvalid-state\t0,0,0,0,0,0\t{after}"
            f"\t{cursor}\t{cursor}\nE3-EDGE-RUNTIME-PASS\n").encode()


class EdgeObservationTests(unittest.TestCase):
    def test_three_zero_weight_outcomes_and_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "corpus"
            manifest = materialize(root)
            for case in manifest["cases"][:3]:
                name, cursor = case["name"], case["start_cursor"]
                result = parse(failure_line(name, cursor), root, manifest, name)
                self.assertEqual(result["status"], "invalid-state")
                self.assertEqual(result["before"], result["after"])
                with self.subTest(name=name), self.assertRaises(ValueError):
                    parse(failure_line(name, cursor, after="0,0,0,0,1,0"),
                          root, manifest, name)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    parse(failure_line(name, "00"), root, manifest, name)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    parse(b"E3CE\t0\tbad\n" + failure_line(name, cursor),
                          root, manifest, name)

    def test_success_needs_real_numerical_trace_and_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "corpus"
            manifest = materialize(root)
            cursor = manifest["cases"][3]["start_cursor"]
            raw = (f"E3EDGE\tone-valid-final\tsuccess\t0,0,0,0,0,0\t"
                   f"0,0,0,0,3,2\t{cursor}\t{cursor}\n"
                   "E3-EDGE-RUNTIME-PASS\n").encode()
            with self.assertRaisesRegex(ValueError, "oracle"):
                parse(raw, root, manifest, "one-valid-final")
            with self.assertRaises(ValueError):
                parse(raw, root, manifest, "one-valid-final", {})


if __name__ == "__main__":
    unittest.main()
