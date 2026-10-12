#!/usr/bin/env python3
"""Source-only corruption controls for first-frame leaf command evidence."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_first_frame_command_receipts import (ALIASES, LABELS, expected_program,
                                                expected_common, check_nested_argv,
                                                inspect, inspect_prefix)


def nested_fixture(evidence: Path) -> None:
    (evidence / "nested-profile.tsv").write_text(
        "cc_default\t0\ncxx_default\t0\ncc_program\tclang-21\n"
        "cxx_program\tclang++-21\nllvm_program\tllvm-config-21\n"
        "cc_relative\t0\ncxx_relative\t0\n")
    for family, labels in (("nested", ("host-uname-m",)),
                           ("static-nested", ("output-reservation", "p2g1-pending"))):
        directory = evidence / (family + ".test")
        directory.mkdir()
        (evidence / ("nested-path.txt" if family == "nested" else
                     "static-nested-path.txt")).write_text(str(directory) + "\n")
        for label in labels:
            receipt = directory / f"{'common' if family == 'nested' else 'static'}-{label}.test"
            receipt.mkdir()
            (receipt / "label").write_text(label + "\n")
            (receipt / "cwd").write_text(str(evidence.resolve()) + "\n")
            if family == "nested":
                argv = ["uname", "-m"]
            else:
                predecessor = ("check-g3c4-output-reservation.py" if label == "output-reservation"
                               else "check-g3c4-p2g1-pending.py")
                argv = [sys.executable, str(Path(__file__).resolve().parents[2] /
                                            "scripts" / predecessor)]
            (receipt / "argv.nul").write_bytes(
                b"".join(arg.encode() + b"\0" for arg in argv))
            (receipt / "stdout.raw").write_bytes(b"child stdout")
            (receipt / "stderr.raw").write_bytes(b"child stderr")
            (receipt / "exit").write_text("0\n")


def fixture(evidence: Path) -> Path:
    nested_fixture(evidence)
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
    def setUp(self):
        patcher = mock.patch("check_first_frame_command_receipts.expected_common",
                             return_value=Counter({"host-uname-m": 1}))
        patcher.start()
        self.addCleanup(patcher.stop)

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

    def test_nested_missing_extra_and_changed_raw_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            fixture(evidence)
            target = evidence / "nested.test/common-host-uname-m.test"
            (target / "exit").unlink()
            with self.assertRaises(ValueError):
                inspect(evidence, seal_current=True)
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            fixture(evidence)
            import shutil
            target = evidence / "nested.test/common-host-uname-m.extra"
            shutil.copytree(evidence / "nested.test/common-host-uname-m.test", target)
            with self.assertRaisesRegex(ValueError, "extra nested child"):
                inspect(evidence, seal_current=True)
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            seal(evidence)
            target = evidence / "static-nested.test/static-p2g1-pending.test/stderr.raw"
            target.write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "changed"):
                inspect(evidence)

    def test_failing_nested_checker_and_toolchain_child_rejected(self):
        for target in ("static-nested.test/static-p2g1-pending.test",
                       "nested.test/common-host-uname-m.test"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                evidence = Path(temporary)
                fixture(evidence)
                (evidence / target / "exit").write_text("1\n")
                with self.assertRaisesRegex(ValueError, "nested exit differs"):
                    inspect(evidence, seal_current=True)

    def test_nested_argv_and_profile_change_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            fixture(evidence)
            target = evidence / "nested.test/common-host-uname-m.test/argv.nul"
            target.write_bytes(b"uname\0-s\0")
            with self.assertRaisesRegex(ValueError, "uname argv differs"):
                inspect(evidence, seal_current=True)
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            seal(evidence)
            (evidence / "nested-profile.tsv").write_text(
                (evidence / "nested-profile.tsv").read_text().replace(
                    "cc_default\t0", "cc_default\t1"))
            with self.assertRaisesRegex(ValueError, "changed"):
                inspect(evidence)

    def test_relative_resolver_and_override_programs_are_source_bounded(self):
        profile = dict(cc_default="0", cxx_default="0", cc_program="custom-cc",
                       cxx_program="custom-cxx", llvm_program="custom-llvm",
                       cc_relative="1", cxx_relative="1")
        # The class's patch covers only success fixtures; exercise the real count rule.
        counts = expected_common(profile)
        self.assertEqual(counts["resolve-resolved_cc-dirname"], 1)
        self.assertEqual(counts["resolve-resolved_cxx-basename"], 1)
        check_nested_argv("resolve-resolved_cc-dirname", "common",
                          ["dirname", "--", "./custom-cc"], profile)
        check_nested_argv("llvm-version", "common", ["/opt/custom-llvm", "--version"],
                          profile)
        with self.assertRaisesRegex(ValueError, "nested executable differs"):
            check_nested_argv("cc-version", "common", ["clang-21", "--version"],
                              profile)

    def test_prefix_retains_early_and_late_failure(self):
        for stop in (1, 20):
            with self.subTest(stop=stop), tempfile.TemporaryDirectory() as temporary:
                evidence = Path(temporary)
                root = fixture(evidence)
                for number, label in enumerate(LABELS, 1):
                    if number > stop:
                        import shutil
                        shutil.rmtree(root / f"{number:04d}-{label}")
                if stop == 20:
                    (root / f"{stop:04d}-{LABELS[stop-1]}" / "exit").unlink()
                active = root / f"{stop+1:04d}-postfailure-audit"
                active.mkdir()
                (active / "label").write_text("postfailure-audit\n")
                (active / "stdout.raw").write_bytes(b"")
                (active / "stderr.raw").write_bytes(b"")
                inspect_prefix(evidence)
                result = (evidence / "postfailure-inventory.json").read_text()
                self.assertIn("HOLD/INCOMPLETE", result)
                self.assertIn("postfailure-audit", result)
                if stop == 20:
                    self.assertIn("missing exit", result)


if __name__ == "__main__":
    unittest.main()
