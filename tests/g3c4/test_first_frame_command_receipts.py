#!/usr/bin/env python3
"""Source-only corruption controls for first-frame leaf command evidence."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_first_frame_command_receipts import ALIASES, LABELS, expected_program, inspect


def fixture(evidence: Path) -> Path:
    root = evidence / "commands.test"
    root.mkdir()
    (evidence / "commands-path.txt").write_text(str(root) + "\n")
    for number, label in enumerate(LABELS, 1):
        receipt = root / f"{number:04d}-{label}"
        receipt.mkdir()
        (receipt / "label").write_text(label + "\n")
        (receipt / "cwd").write_text(str(evidence.resolve()) + "\n")
        (receipt / "argv.nul").write_bytes((expected_program(label) + "\0").encode())
        (receipt / "stdout.raw").write_bytes(("out:" + label).encode())
        stderr = ("C4 P2/G2 first frame requires pending output and token forward"
                  if label == "invalid-compile" else "err:" + label)
        (receipt / "stderr.raw").write_bytes(stderr.encode())
        if label != "receipts-audit":
            (receipt / "exit").write_text("1\n" if label == "invalid-compile" else "0\n")
            for alias, stream in zip(ALIASES.get(label, (None, None)),
                                     ("stdout.raw", "stderr.raw")):
                if alias:
                    os.link(receipt / stream, evidence / alias)
    return root


def seal(evidence: Path) -> Path:
    root = fixture(evidence)
    assert inspect(evidence, seal_current=True) == len(LABELS)
    audit = root / f"{len(LABELS):04d}-receipts-audit"
    (audit / "exit").write_text("0\n")
    for alias, stream in zip(ALIASES["receipts-audit"],
                             ("stdout.raw", "stderr.raw")):
        os.link(audit / stream, evidence / alias)
    assert inspect(evidence) == len(LABELS)
    return root


class FirstFrameReceiptTests(unittest.TestCase):
    def test_complete_receipts_and_snapshot_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            seal(Path(temporary))

    def test_missing_and_duplicate_receipts_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            root = fixture(evidence)
            missing = root / "0011-symbol-parity"
            missing.rename(root / "removed")
            with self.assertRaises(ValueError):
                inspect(evidence, seal_current=True)
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            root = fixture(evidence)
            (root / "0032-duplicate").mkdir()
            with self.assertRaises(ValueError):
                inspect(evidence, seal_current=True)

    def test_malformed_argv_and_exit_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            root = fixture(evidence)
            target = root / "0011-symbol-parity/argv.nul"
            target.write_bytes(b"cmp\0\0")
            with self.assertRaisesRegex(ValueError, "NUL argv"):
                inspect(evidence, seal_current=True)
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            root = fixture(evidence)
            (root / "0012-invalid-compile/exit").write_text("124\n")
            with self.assertRaisesRegex(ValueError, "ordinary rejection"):
                inspect(evidence, seal_current=True)

    def test_missing_exit_and_altered_raw_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            root = fixture(evidence)
            (root / "0002-static/exit").unlink()
            with self.assertRaises(ValueError):
                inspect(evidence, seal_current=True)
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            root = seal(evidence)
            (root / "0002-static/stdout.raw").write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "changed"):
                inspect(evidence)

    def test_current_diagnostic_and_alias_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            root = fixture(evidence)
            (root / "0012-invalid-compile/stderr.raw").write_bytes(b"stale")
            with self.assertRaisesRegex(ValueError, "diagnostic absent"):
                inspect(evidence, seal_current=True)
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            fixture(evidence)
            (evidence / "off.symbols").unlink()
            (evidence / "off.symbols").write_bytes(b"substituted")
            with self.assertRaisesRegex(ValueError, "raw alias differs"):
                inspect(evidence, seal_current=True)


if __name__ == "__main__":
    unittest.main()
