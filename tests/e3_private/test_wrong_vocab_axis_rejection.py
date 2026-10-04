"""Exact negative verdict for the private E3 wrong-vocabulary-axis witness."""

from __future__ import annotations

from argparse import Namespace
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.e3_reference.correctness import build_reference, materialize
from tests.e3_reference.transcript import canonical, encode
from tests.e3_private.wrong_vocab_axis_rejection import (
    ERROR, MARKER, NUMPY_WARNING, check, run,
)


ROOT = Path(__file__).resolve().parents[2]


class WrongAxisRejectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="e3-wrong-axis-negative-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        manifest = materialize(self.directory / "corpus")
        reference = build_reference(manifest)
        observed = copy.deepcopy(reference["cases"][0]["synthetic"])
        observed["kind"] = "observed-f32"
        for batch in observed["batches"]:
            batch["correct_f32"] = "00000000"
            batch["state_f32"][2] = "00000000"
        observed["final"]["metrics_f32"][3] = "00000000"
        self.reference = self.directory / "reference.json"
        self.observed = self.directory / "observed.json"
        self.roles = self.directory / "roles.tsv"
        self.transcript = self.directory / "rejected-transcript.json"
        self.traceback = self.directory / "validator.traceback"
        self.status = self.directory / "validator.exit"
        self.stdout = self.directory / "validator.stdout"
        self.stderr = self.directory / "validator.stderr"
        self.reference.write_bytes(encode(reference))
        self.observed.write_bytes(canonical(observed))
        self.roles.write_bytes(b"")
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / "tests/e3_private/wrong_vocab_axis_rejection.py"),
             "run", "--reference", str(self.reference), "--observed", str(self.observed),
             "--roles", str(self.roles), "--transcript", str(self.transcript),
             "--traceback", str(self.traceback)],
            cwd=self.directory, env=environment, capture_output=True, check=False,
        )
        self.status.write_text(f"{completed.returncode}\n")
        self.stdout.write_bytes(completed.stdout)
        self.stderr.write_bytes(completed.stderr)

    def audit(self) -> None:
        check(Namespace(status=self.status, stdout=self.stdout, stderr=self.stderr,
                        traceback=self.traceback, transcript=self.transcript))

    def test_exact_validator_exception_is_the_only_accepted_negative(self) -> None:
        self.assertEqual(self.status.read_text(), "1\n")
        self.assertEqual(self.stdout.read_bytes(), b"")
        self.assertIn(self.stderr.read_bytes(), (MARKER, NUMPY_WARNING + MARKER))
        self.assertIn(("TranscriptError: " + ERROR).encode(), self.traceback.read_bytes())
        self.assertFalse(self.transcript.exists())
        self.audit()
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        external = subprocess.run(
            [sys.executable, "-B", str(ROOT / "tests/e3_private/wrong_vocab_axis_rejection.py"),
             "check", "--status", str(self.status), "--stdout", str(self.stdout),
             "--stderr", str(self.stderr), "--traceback", str(self.traceback),
             "--transcript", str(self.transcript)],
            cwd=self.directory, env=environment, capture_output=True, check=False,
        )
        self.assertEqual((external.returncode, external.stdout, external.stderr),
                         (0, b"", b""))

    def test_status_stdout_and_diagnostic_mutations_fail_closed(self) -> None:
        self.audit()
        for bad in (b"0\n", b"2\n", b"124\n", b"130\n", b"137\n", b"143\n",
                    b"1\n2\n", b"01\n",
                    b"-11\n", b"missing\n"):
            with self.subTest(status=bad):
                self.status.write_bytes(bad)
                with self.assertRaises(ValueError):
                    self.audit()
        self.status.write_bytes(b"1\n")
        self.stdout.write_bytes(b"unexpected success\n")
        with self.assertRaisesRegex(ValueError, "stdout"):
            self.audit()
        self.stdout.write_bytes(b"")
        original = self.stderr.read_bytes()
        for bad in (b"", b"unexpected diagnostic\n", original + b"extra\n",
                    original.replace(b"correct_f32", b"ce_f32")):
            with self.subTest(stderr=bad[:40]):
                self.stderr.write_bytes(bad)
                with self.assertRaisesRegex(ValueError, "diagnostic"):
                    self.audit()
        self.stderr.write_bytes(original)
        self.audit()
        self.stderr.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.audit()

    def test_missing_tampered_trace_and_extra_transcript_fail_closed(self) -> None:
        self.audit()
        original = self.traceback.read_bytes()
        self.traceback.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.audit()
        self.traceback.write_bytes(original.replace(b"in _fail", b"in _fake"))
        with self.assertRaisesRegex(ValueError, "source context"):
            self.audit()
        self.traceback.write_bytes(original)
        self.transcript.write_bytes(b"{}\n")
        with self.assertRaisesRegex(ValueError, "published"):
            self.audit()

    def test_same_message_from_untrusted_exception_source_does_not_admit(self) -> None:
        for index, error in enumerate((ValueError(ERROR), RuntimeError("crash"))):
            other_trace = self.directory / f"other-{index}.traceback"
            args = Namespace(reference=self.reference, observed=self.observed,
                             roles=self.roles, transcript=self.transcript,
                             traceback=other_trace)
            with self.subTest(error=type(error).__name__):
                with patch("tests.e3_native_parity.verify_observation.main",
                           side_effect=error):
                    with self.assertRaises(type(error)):
                        run(args)
                self.assertTrue(other_trace.is_file())


if __name__ == "__main__":
    unittest.main()
