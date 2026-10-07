"""Owned negative controls for the exact compiled first-warmup verdict."""

import signal
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

from tests.e3_private.parse_first_eval_detection import LABEL, verify

ROOT = Path(__file__).resolve().parents[2]


class FirstEvalVerdict(unittest.TestCase):
    def test_accepts_only_exact_signal_and_label(self) -> None:
        for selector, gradient in (("absent-to-present", "absent"),
                                   ("seeded-bit", "present"),
                                   ("seeded-metadata", "present")):
            for horizon in (1024, 8192):
                valid = dict(selector=selector, gradient=gradient, horizon=horizon,
                             package="retention-first-eval-mutant",
                             compile_status=0, link_status=0,
                             runtime_status=-signal.SIGABRT,
                             runtime_signal=signal.SIGABRT, timed_out=False,
                             survivors=False, stdout=LABEL, stderr=b"")
                with self.subTest(selector=selector, horizon=horizon):
                    verify(**valid)
                for change in (
                    dict(stdout=b""), dict(stdout=LABEL[:-1]),
                    dict(stdout=LABEL + LABEL), dict(stdout=LABEL + b"extra\n"),
                    dict(stdout=b"E3-RETENTION-FAIL: mutation-readback\n"),
                    dict(gradient="present" if gradient == "absent" else "absent"),
                    dict(package="retention"), dict(compile_status=1),
                    dict(link_status=1), dict(runtime_status=134,
                                             runtime_signal=None),
                    dict(runtime_status=0, runtime_signal=None),
                    dict(runtime_status=-signal.SIGSEGV,
                         runtime_signal=signal.SIGSEGV),
                    dict(timed_out=True), dict(survivors=True),
                    dict(stderr=b"AddressSanitizer: unrelated failure\n"),
                ):
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        verify(**(valid | change))
        for invalid in ("unknown", "seeded-metadata-extra"):
            with self.assertRaises(ValueError):
                verify(selector=invalid, gradient="present", horizon=1024,
                       package="retention-first-eval-mutant", compile_status=0,
                       link_status=0, runtime_status=-signal.SIGABRT,
                       runtime_signal=signal.SIGABRT, timed_out=False,
                       survivors=False, stdout=LABEL, stderr=b"")

    def test_capture_distinguishes_signal_from_exit_134(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "first_eval_aot_runner", ROOT / "scripts/test-e3-first-eval-aot-detection.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            for name, command in (
                ("abort", "import os,signal,sys;sys.stdout.write('E3-RETENTION-FAIL: success-model-image\\n');sys.stdout.flush();os.kill(os.getpid(),signal.SIGABRT)"),
                ("exit", "import sys;sys.stdout.write('E3-RETENTION-FAIL: success-model-image\\n');sys.exit(134)"),
            ):
                result = module.capture(out, name, [sys.executable, "-c", command],
                                        {"PATH": os.environ.get("PATH", "")}, ROOT, 10)
                self.assertEqual((out / (name + ".stdout")).read_bytes(), LABEL)
                if name == "abort":
                    self.assertEqual((result["returncode"], result["signal"]),
                                     (-signal.SIGABRT, signal.SIGABRT))
                else:
                    self.assertEqual((result["returncode"], result["signal"]),
                                     (134, None))


if __name__ == "__main__":
    unittest.main()
