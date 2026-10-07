"""Owned negative controls for the exact compiled first-warmup verdict."""

import signal
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import warnings

from tests.e3_private.parse_first_eval_detection import LABEL, verify

ROOT = Path(__file__).resolve().parents[2]


def runner():
    spec = importlib.util.spec_from_file_location(
        "first_eval_aot_runner", ROOT / "scripts/test-e3-first-eval-aot-detection.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        module = runner()
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

    def test_pid_reuse_and_foreign_session_refuse_group_signal(self) -> None:
        module = runner()
        leader = (999999, 999999, 999999, 42, "R")
        for changed in ((999999, 999999, 999999, 43, "R"),
                        (999999, 999999, 999998, 42, "R")):
            with self.subTest(identity=changed), \
                 mock.patch.object(module, "process_identity", return_value=changed), \
                 mock.patch.object(module.os, "killpg") as kill:
                with self.assertRaises(ValueError):
                    module.signal_owned_session(leader, os.getuid(), signal.SIGTERM)
                kill.assert_not_called()
        current = module.process_identity(os.getpid())
        with mock.patch.object(module.os, "killpg") as kill:
            with self.assertRaises(ValueError):
                module.signal_owned_session(current, os.getuid() + 1,
                                            signal.SIGTERM)
            kill.assert_not_called()

    def test_timeout_and_descendant_writer_are_closed(self) -> None:
        module = runner()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            timed = module.capture(out, "timeout", [sys.executable, "-c",
                "import time;time.sleep(30)"], {"PATH": os.environ.get("PATH", "")},
                ROOT, 1)
            self.assertTrue(timed["timed_out"])
            self.assertTrue(timed["quiescent"])
            self.assertEqual(timed["signal"], signal.SIGTERM)
            self.assertIsNotNone(timed["stdout_sha256"])
            child = ("import subprocess,sys;"
                     "subprocess.Popen([sys.executable,'-c',"
                     "'import time;time.sleep(30)']);print('parent closed',flush=True)")
            observed = module.capture(out, "descendant", [sys.executable, "-c", child],
                                      {"PATH": os.environ.get("PATH", "")}, ROOT, 5)
            self.assertTrue(observed["survivors"])
            self.assertTrue(observed["quiescent"])
            self.assertIn("unexpected descendants: SIGTERM", observed["teardown"])
            self.assertEqual((out / "descendant.stdout").read_bytes(), b"parent closed\n")

    def test_wait_exception_cleans_owned_child_and_preserves_failure(self) -> None:
        module = runner()
        real_waitid = os.waitid
        calls = 0

        def fail_once(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("injected wait failure")
            return real_waitid(*args)

        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            with mock.patch.object(module.os, "waitid", side_effect=fail_once):
                with self.assertRaises(RuntimeError):
                    module.capture(out, "wait-failure", [sys.executable, "-c",
                        "import time;time.sleep(30)"],
                        {"PATH": os.environ.get("PATH", "")}, ROOT, 5)
            record = json.loads((out / "wait-failure.json").read_text())
            self.assertIn("injected wait failure", record["capture_error"])
            self.assertIn("exception cleanup: SIGKILL", record["teardown"])
            self.assertTrue(record["quiescent"])
            self.assertEqual(record["signal"], signal.SIGKILL)

    def test_pidfd_close_failure_retains_wait_and_rejects_hash(self) -> None:
        module = runner()
        real_open, real_close = os.pidfd_open, os.close
        owned_fds = set()

        def open_owned(pid):
            fd = real_open(pid)
            owned_fds.add(fd)
            return fd

        def close_fault(fd):
            real_close(fd)
            if fd in owned_fds:
                raise OSError("injected pidfd close failure")

        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            with mock.patch.object(module.os, "pidfd_open", side_effect=open_owned), \
                 mock.patch.object(module.os, "close", side_effect=close_fault):
                with self.assertRaises(RuntimeError):
                    module.capture(out, "close-failure", [sys.executable, "-c",
                        "print('closed')"], {"PATH": os.environ.get("PATH", "")},
                        ROOT, 5)
            record = json.loads((out / "close-failure.json").read_text())
            self.assertIn("injected pidfd close failure", record["capture_error"])
            self.assertTrue(record["quiescent"])
            self.assertIsNotNone(record["wait_status"])
            self.assertIsNone(record["stdout_sha256"])

    def test_pidfd_close_does_not_replace_primary_wait_failure(self) -> None:
        module = runner()
        real_waitid, real_open, real_close = os.waitid, os.pidfd_open, os.close
        owned_fds = set()
        calls = 0

        def fail_first_wait(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("primary wait failure")
            return real_waitid(*args)

        def open_owned(pid):
            fd = real_open(pid)
            owned_fds.add(fd)
            return fd

        def close_fault(fd):
            real_close(fd)
            if fd in owned_fds:
                raise OSError("secondary pidfd close fault")

        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            with mock.patch.object(module.os, "waitid", side_effect=fail_first_wait), \
                 mock.patch.object(module.os, "pidfd_open", side_effect=open_owned), \
                 mock.patch.object(module.os, "close", side_effect=close_fault):
                with self.assertRaises(RuntimeError):
                    module.capture(out, "primary-close", [sys.executable, "-c",
                        "import time;time.sleep(30)"],
                        {"PATH": os.environ.get("PATH", "")}, ROOT, 5)
            record = json.loads((out / "primary-close.json").read_text())
            self.assertIn("primary wait failure", record["capture_error"])
            self.assertTrue(any("secondary pidfd close fault" in item
                                for item in record["teardown"]))
            self.assertIsNone(record["stdout_sha256"])

    def test_pidfd_signal_and_unverified_wait_faults_keep_primary(self) -> None:
        module = runner()
        real_waitid, real_wait4 = os.waitid, os.wait4
        for wait4_fault in (False, True):
            calls = 0

            def fail_first_wait(*args):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("primary waitid fault")
                return real_waitid(*args)

            def fail_wait4(*args):
                raise InterruptedError("injected unverified wait4 fault")

            with self.subTest(wait4_fault=wait4_fault), \
                 tempfile.TemporaryDirectory() as temporary:
                out = Path(temporary)
                wait4 = fail_wait4 if wait4_fault else real_wait4
                with mock.patch.object(module.os, "waitid", side_effect=fail_first_wait), \
                     mock.patch.object(module, "signal_owned_session",
                                       side_effect=ValueError("group cleanup refused")), \
                     mock.patch.object(module.signal, "pidfd_send_signal",
                                       side_effect=OSError("injected pidfd signal fault")), \
                     mock.patch.object(module.os, "wait4", side_effect=wait4):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ResourceWarning)
                        with self.assertRaises(RuntimeError):
                            module.capture(out, "cleanup-fault", [sys.executable, "-c",
                                "print('done')"],
                                {"PATH": os.environ.get("PATH", "")}, ROOT, 5)
                record = json.loads((out / "cleanup-fault.json").read_text())
                self.assertIn("primary waitid fault", record["capture_error"])
                self.assertTrue(any("pidfd signal uncertainty" in item
                                    for item in record["teardown"]))
                self.assertFalse(record["quiescent"])
                self.assertIsNone(record["stdout_sha256"])
                if wait4_fault:
                    self.assertIsNone(record["wait_status"])
                    self.assertTrue(any("unverified wait4 uncertainty" in item
                                        for item in record["teardown"]))
                pid = record["leader_identity"][0]
                try:
                    os.waitpid(pid, 0)
                except ChildProcessError:
                    pass

    def test_stream_close_fault_still_records_failed_capture(self) -> None:
        module = runner()

        class FaultStack:
            def __init__(self):
                self.inner = ExitStack()

            def enter_context(self, context):
                return self.inner.enter_context(context)

            def close(self):
                self.inner.close()
                raise OSError("injected stream close fault")

        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            with mock.patch.object(module, "ExitStack", FaultStack):
                with self.assertRaises(RuntimeError):
                    module.capture(out, "stream-close", [sys.executable, "-c",
                        "print('done')"], {"PATH": os.environ.get("PATH", "")},
                        ROOT, 5)
            record = json.loads((out / "stream-close.json").read_text())
            self.assertIn("injected stream close fault", record["capture_error"])
            self.assertIsNotNone(record["wait_status"])
            self.assertIsNone(record["stdout_sha256"])


if __name__ == "__main__":
    unittest.main()
