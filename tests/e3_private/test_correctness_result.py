"""Test-only proof that a compiled result cannot be inferred from frame counters."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.e3_private.parse_correctness_observation import parse_correctness
from tests.e3_private.test_numerical_tuple import frame_lines
from tests.e3_reference.transcript import canonical


MANIFEST = {"cases": [
    {"name": "all-correct", "start_cursor": "aa", "end_cursor": "bb"},
    {"name": "all-wrong", "start_cursor": "cc", "end_cursor": "dd"},
]}
OUTPUTS = "1065353216,1073741824,1077936128,0,6,3"


def sample(outputs: str = OUTPUTS, before: str = "aa", after: str = "aa") -> bytes:
    lines = frame_lines(3)
    lines[-1:-1] = [f"E3RESULT\t{outputs}", f"E3CURSOR\t{before}\t{after}"]
    return ("\n".join(lines) + "\n").encode("ascii")


class CorrectnessResultTests(unittest.TestCase):
    def test_complete_result_and_cursor_bound_to_observed_frame(self) -> None:
        observed, roles, result = parse_correctness(sample(), MANIFEST, "all-correct")
        self.assertEqual(len(roles.splitlines()), 63)
        self.assertEqual(observed["final"]["counters"], [6, 3])
        self.assertEqual(result["outputs_u32"],
                         [0x3f800000, 0x40000000, 0x40400000, 0, 6, 3])
        self.assertEqual((result["cursor_before"], result["cursor_after"]),
                         ("aa", "aa"))

    def test_wrong_output_or_cursor_cannot_use_valid_frame(self) -> None:
        variants = [
            sample(outputs=OUTPUTS.replace("1065353216", "0", 1)),
            sample(outputs=OUTPUTS[:-1] + "2"),
            sample(before="ab", after="ab"), sample(after="bc"),
            sample(after="bb"),  # Traversal end is not the restored caller cursor.
            sample(outputs="4294967296," + OUTPUTS.split(",", 1)[1]),
            sample(outputs="01," + OUTPUTS.split(",", 1)[1]),
            sample().replace(b"E3RESULT", b"E3RESULT\t0,0,0,0,0,0\nE3RESULT", 1),
            sample().replace(b"E3CURSOR\taa", b"E3CURSOR\tAA", 1),
            sample().replace(b"E3CURSOR\taa", b"E3CURSOR\taa\n", 1),
            sample().replace(b"E3CURSOR\taa", b"E3CURSOR\tbb", 1),
        ]
        for index, raw in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                parse_correctness(raw, MANIFEST, "all-correct")
        changed = copy.deepcopy(MANIFEST)
        changed["cases"][1]["name"] = "all-correct"
        with self.assertRaises(ValueError):
            parse_correctness(sample(), changed, "all-correct")

    def test_runner_uses_real_private_call_and_separate_result_parser(self) -> None:
        root = Path(__file__).resolve().parents[2]
        runtime = (root / "tests/e3_private/correctness_runtime.esk").read_text()
        runner = (root / "scripts/test-e3-correctness-cases.sh").read_text()
        self.assertIn("(e3-private-test-run", runtime)
        self.assertIn('(display "E3RESULT\\t")', runtime)
        self.assertIn('(display "E3CURSOR\\t")', runtime)
        self.assertIn("parse_correctness_observation", runner)
        self.assertIn("verify_observation", runner)
        self.assertIn("wrong_vocab_axis_rejection.py", runner)

    def test_exact_cli_writes_complete_record_and_rejects_duplicate_manifest(self) -> None:
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="e3-result-parser-") as temporary:
            directory = Path(temporary)
            raw, manifest = directory / "raw", directory / "manifest.json"
            raw.write_bytes(sample())
            manifest.write_bytes(canonical(MANIFEST))
            command = [sys.executable, "-B", "-m",
                       "tests.e3_private.parse_correctness_observation",
                       "--stdout", str(raw), "--manifest", str(manifest),
                       "--case", "all-correct", "--observed", str(directory / "observed"),
                       "--roles", str(directory / "roles"),
                       "--result", str(directory / "result")]
            accepted = subprocess.run(command, cwd=root, capture_output=True, check=False)
            self.assertEqual((accepted.returncode, accepted.stdout, accepted.stderr),
                             (0, b"", b""))
            self.assertEqual(json.loads((directory / "result").read_bytes())["outputs_u32"],
                             [0x3f800000, 0x40000000, 0x40400000, 0, 6, 3])
            duplicate = canonical(MANIFEST).replace(b'"cases":', b'"cases":[],"cases":', 1)
            manifest.write_bytes(duplicate)
            rejected = subprocess.run(command, cwd=root, capture_output=True, check=False)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(b"duplicate correctness manifest field", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
