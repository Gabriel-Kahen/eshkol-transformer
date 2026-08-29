from __future__ import annotations

import copy
import contextlib
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

B0_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = B0_DIR.parents[1]
sys.path.insert(0, str(B0_DIR))

import benchmark_format  # noqa: E402
import run_benchmark  # noqa: E402


def _definition(
    *, target: str = "build/eshkol-transformer-smoke", warmup: int = 0, repetitions: int = 1
) -> dict[str, object]:
    return {
        "argv": [target],
        "backend": {
            "acceleration": False,
            "device": "cpu",
            "evidence": "direct-process-execution",
            "name": "host-cpu",
        },
        "cwd": ".",
        "dtype": "not-applicable",
        "expected": {
            "exit_code": 0,
            "stderr": "empty",
            "stdout_sha256": hashlib.sha256(b"ok\n").hexdigest(),
        },
        "id": "smoke-v1",
        "measurement": {
            "clock": "clock_monotonic",
            "completion_source": "linux_pidfd_poll",
            "elapsed_unit": "nanosecond",
            "memory_source": "linux_wait4_ru_maxrss",
            "peak_rss_unit": "kibibyte",
            "scope": "direct-child-process",
        },
        "repetitions": repetitions,
        "shapes": [],
        "timeout_ms": 1000,
        "warmup": warmup,
        "work": {"count": 1, "unit": "run"},
    }


def _metrics(**replacements: object) -> str:
    fields = {
        "format_version": "1",
        "elapsed_ns": "1000",
        "peak_rss_kib": "64",
        "exit_code": "0",
        "term_signal": "0",
        "timed_out": "false",
    }
    fields.update({key: str(value) for key, value in replacements.items()})
    return "".join(f"{key}\t{value}\n" for key, value in fields.items())


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _write_fake_helper(
    path: Path,
    *,
    metrics: str | None = None,
    captured_stdout: bytes = b"ok\n",
    captured_stderr: bytes = b"",
    helper_stderr: str = "",
    returncode: int = 0,
) -> None:
    source = f"""#!{sys.executable}
import pathlib
import sys

pathlib.Path(sys.argv[2]).write_bytes({captured_stdout!r})
pathlib.Path(sys.argv[3]).write_bytes({captured_stderr!r})
sys.stdout.write({(metrics if metrics is not None else _metrics())!r})
sys.stderr.write({helper_stderr!r})
raise SystemExit({returncode})
"""
    _write_executable(path, source)


class MetricProtocolTests(unittest.TestCase):
    def test_valid_metrics_are_typed(self) -> None:
        self.assertEqual(
            run_benchmark.parse_metrics(_metrics()),
            {
                "elapsed_ns": 1000,
                "peak_rss_kib": 64,
                "exit_code": 0,
                "term_signal": 0,
                "timed_out": False,
            },
        )

    def test_malformed_missing_duplicate_and_unknown_metrics_are_rejected(self) -> None:
        cases = (
            ("not-tab-separated\n", "malformed metrics"),
            (_metrics() + "elapsed_ns\t2\n", "malformed metrics"),
            (_metrics().replace("elapsed_ns\t1000\n", ""), "unsupported metric fields"),
            (_metrics() + "unknown\t1\n", "unsupported metric fields"),
            (_metrics(format_version=2), "unsupported metric fields"),
        )
        for raw, message in cases:
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(run_benchmark.BenchmarkRunError, message):
                    run_benchmark.parse_metrics(raw)

    def test_invalid_numbers_flags_and_nonpositive_measurements_are_rejected(self) -> None:
        cases = (
            (_metrics(elapsed_ns="one"), "non-integer metric"),
            (_metrics(timed_out="yes"), "invalid timeout flag"),
            (_metrics(elapsed_ns=0), "must both be positive"),
            (_metrics(peak_rss_kib=-1), "must both be positive"),
        )
        for raw, message in cases:
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(run_benchmark.BenchmarkRunError, message):
                    run_benchmark.parse_metrics(raw)


class MeasureOnceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.helper = self.root / "helper"
        self.target = self.root / "target"
        _write_executable(self.target, f"#!{sys.executable}\nraise SystemExit(0)\n")
        self.expected = {
            "exit_code": 0,
            "stderr": "empty",
            "stdout_sha256": hashlib.sha256(b"ok\n").hexdigest(),
        }

    def measure(self) -> dict[str, object]:
        return run_benchmark.measure_once(
            self.helper,
            self.target,
            [],
            1000,
            self.expected,
            self.root,
            "case",
        )

    def test_success_returns_captured_hashes(self) -> None:
        _write_fake_helper(self.helper)
        measured = self.measure()
        self.assertEqual(measured["stdout_sha256"], self.expected["stdout_sha256"])
        self.assertEqual(
            measured["stderr_sha256"], benchmark_format.EMPTY_SHA256
        )

    def test_unavailable_or_failing_helper_is_reported(self) -> None:
        with self.assertRaisesRegex(
            run_benchmark.BenchmarkRunError, "measurement helper is unavailable"
        ):
            self.measure()

        _write_fake_helper(
            self.helper, helper_stderr="deliberate helper failure\n", returncode=7
        )
        with self.assertRaisesRegex(
            run_benchmark.BenchmarkRunError,
            "measurement helper failed with 7: deliberate helper failure",
        ):
            self.measure()

        _write_fake_helper(self.helper, helper_stderr="fallback warning\n")
        with self.assertRaisesRegex(
            run_benchmark.BenchmarkRunError,
            "unexpected stderr.*degraded or fallback measurement",
        ):
            self.measure()

    def test_timeout_signal_and_wrong_exit_are_not_hidden(self) -> None:
        cases = (
            ({"timed_out": "true"}, "exceeded 1000 ms"),
            ({"term_signal": 15}, "terminated by signal 15"),
            ({"exit_code": 9}, "exited 9; expected 0"),
        )
        for replacements, message in cases:
            with self.subTest(replacements=replacements):
                _write_fake_helper(self.helper, metrics=_metrics(**replacements))
                with self.assertRaisesRegex(run_benchmark.BenchmarkRunError, message):
                    self.measure()

    def test_stdout_mismatch_and_unexpected_stderr_are_rejected(self) -> None:
        _write_fake_helper(self.helper, captured_stdout=b"wrong\n")
        with self.assertRaisesRegex(
            run_benchmark.BenchmarkRunError, "stdout checksum mismatch"
        ):
            self.measure()

        _write_fake_helper(self.helper, captured_stderr=b"warning\n")
        with self.assertRaisesRegex(
            run_benchmark.BenchmarkRunError, "wrote unexpected stderr"
        ):
            self.measure()


class BuildReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.definition_path = self.root / "definition.json"
        self.definition_path.write_bytes(benchmark_format.encode_definition(_definition()))
        self.target = self.root / "build" / "eshkol-transformer-smoke"
        self.target.parent.mkdir()
        self.helper = self.root / "helper"
        self.provenance = self.root / "provenance.tsv"
        self.manifest = self.root / "toolchain-manifest.tsv"
        benchmark_sources = self.root / "benchmarks"
        benchmark_sources.mkdir()
        shutil.copyfile(
            PROJECT_ROOT / "benchmarks" / "measure_linux.c",
            benchmark_sources / "measure_linux.c",
        )
        _write_executable(self.target, f"#!{sys.executable}\nprint('ok')\n")
        _write_fake_helper(self.helper)
        self.provenance.write_text(
            "cc_path\t/usr/bin/cc\n"
            "cc_version\tcc test\n"
            f"eshkol_binary_sha256\t{'a' * 64}\n"
            f"eshkol_commit\t{benchmark_format.CANONICAL_UPSTREAM_COMMIT}\n"
            f"eshkol_repository\t{benchmark_format.CANONICAL_UPSTREAM_REPOSITORY}\n",
            encoding="utf-8",
        )
        self.manifest.write_text(
            "host_supported\ttrue\neshkol_version\t1.3.4\n", encoding="utf-8"
        )

    def patches(self) -> object:
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(run_benchmark, "PROJECT_ROOT", self.root))
        stack.enter_context(mock.patch.object(run_benchmark.platform, "system", return_value="Linux"))
        stack.enter_context(mock.patch.object(run_benchmark.platform, "machine", return_value="x86_64"))
        stack.enter_context(mock.patch.object(run_benchmark.platform, "release", return_value="test-kernel"))
        stack.enter_context(mock.patch.object(run_benchmark, "cpu_model", return_value="test-cpu"))
        stack.enter_context(mock.patch.object(run_benchmark, "os_release", return_value="test-os"))
        stack.enter_context(mock.patch.object(run_benchmark.os, "cpu_count", return_value=4))
        stack.enter_context(
            mock.patch.object(
                run_benchmark,
                "git_output",
                side_effect=lambda *args: "1" * 40 if args == ("rev-parse", "HEAD") else "",
            )
        )
        return stack

    def build(self) -> dict[str, object]:
        return run_benchmark.build_report(
            self.definition_path,
            self.helper,
            self.provenance,
            self.manifest,
            "supported",
        )

    def test_unsupported_platform_is_explicit(self) -> None:
        with mock.patch.object(run_benchmark.platform, "system", return_value="Darwin"):
            with self.assertRaisesRegex(
                run_benchmark.BenchmarkRunError, "unsupported outside Linux"
            ):
                self.build()

    def test_missing_and_nonexecutable_target_or_helper_are_rejected(self) -> None:
        with mock.patch.object(run_benchmark, "PROJECT_ROOT", self.root):
            self.target.unlink()
            with self.assertRaisesRegex(
                run_benchmark.BenchmarkRunError,
                "target is missing or not executable",
            ):
                self.build()

            _write_executable(self.target, f"#!{sys.executable}\nprint('ok')\n")
            self.target.chmod(0o644)
            with self.assertRaisesRegex(
                run_benchmark.BenchmarkRunError,
                "target is missing or not executable",
            ):
                self.build()

            self.target.chmod(0o755)
            self.helper.chmod(0o644)
            with self.assertRaisesRegex(
                run_benchmark.BenchmarkRunError,
                "helper is missing or not executable",
            ):
                self.build()

            self.helper.unlink()
            with self.assertRaisesRegex(
                run_benchmark.BenchmarkRunError,
                "helper is missing or not executable",
            ):
                self.build()

    def test_malformed_duplicate_and_invalid_support_provenance_are_rejected(self) -> None:
        self.patches()
        for path, contents, message in (
            (self.provenance, "broken\n", "malformed TSV"),
            (self.provenance, "key\tone\nkey\ttwo\n", "duplicate TSV key"),
        ):
            with self.subTest(path=path, contents=contents):
                original = path.read_text(encoding="utf-8")
                path.write_text(contents, encoding="utf-8")
                try:
                    with self.assertRaisesRegex(run_benchmark.BenchmarkRunError, message):
                        self.build()
                finally:
                    path.write_text(original, encoding="utf-8")
        with self.assertRaisesRegex(
            run_benchmark.BenchmarkRunError, "support status is unavailable"
        ):
            run_benchmark.build_report(
                self.definition_path,
                self.helper,
                self.provenance,
                self.manifest,
                "unknown",
            )

    def test_missing_metadata_and_noncanonical_upstream_are_rejected(self) -> None:
        self.patches()
        with mock.patch.object(run_benchmark.os, "cpu_count", return_value=None):
            with self.assertRaisesRegex(
                run_benchmark.BenchmarkRunError, "logical CPU count is unavailable"
            ):
                self.build()

        original = self.provenance.read_text(encoding="utf-8")
        self.provenance.write_text(
            "\n".join(
                line
                for line in original.splitlines()
                if not line.startswith("cc_path\t")
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(run_benchmark.BenchmarkRunError, "missing 'cc_path'"):
            self.build()

        self.provenance.write_text(
            original.replace(
                benchmark_format.CANONICAL_UPSTREAM_REPOSITORY,
                "https://github.com/example/fork.git",
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            benchmark_format.BenchmarkFormatError, "unsupported upstream repository"
        ):
            self.build()

    def test_compatibility_only_status_is_preserved(self) -> None:
        self.patches()
        report = run_benchmark.build_report(
            self.definition_path,
            self.helper,
            self.provenance,
            self.manifest,
            "compatibility-only",
        )
        self.assertEqual(report["volatile"]["support_status"], "compatibility-only")

    def test_corrupt_report_and_mismatched_definition_fail_verification(self) -> None:
        self.patches()
        report = self.build()
        report_path = self.root / "report.json"
        encoded = benchmark_format.encode_report(report)
        report_path.write_bytes(encoded.replace(b'"dirty":false', b'"dirty":true'))
        arguments = SimpleNamespace(
            definition=str(self.definition_path), report=str(report_path)
        )
        with self.assertRaisesRegex(
            benchmark_format.BenchmarkFormatError, "checksum mismatch"
        ):
            run_benchmark.command_verify(arguments)

        report_path.write_bytes(encoded)
        changed_definition = copy.deepcopy(_definition())
        changed_definition["warmup"] = 1
        self.definition_path.write_bytes(
            benchmark_format.encode_definition(changed_definition)
        )
        with self.assertRaisesRegex(
            benchmark_format.BenchmarkFormatError, "report does not match"
        ):
            run_benchmark.command_verify(arguments)


class AtomicWriteTests(unittest.TestCase):
    def test_replace_failure_preserves_previous_report_and_removes_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.json"
            report.write_bytes(b"previous")
            with mock.patch.object(
                run_benchmark.os, "replace", side_effect=OSError("denied")
            ):
                with self.assertRaisesRegex(
                    run_benchmark.BenchmarkRunError, "cannot write report atomically"
                ):
                    run_benchmark.write_atomic(report, b"replacement")
            self.assertEqual(report.read_bytes(), b"previous")
            self.assertEqual(list(root.iterdir()), [report])

    def test_failed_measurement_does_not_clobber_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            report.write_bytes(b"previous")
            arguments = SimpleNamespace(
                definition="definition",
                report=str(report),
                helper="helper",
                provenance="provenance",
                toolchain_manifest="manifest",
                support_status="supported",
            )
            with mock.patch.object(
                run_benchmark,
                "build_report",
                side_effect=run_benchmark.BenchmarkRunError("measurement failed"),
            ):
                with self.assertRaisesRegex(
                    run_benchmark.BenchmarkRunError, "measurement failed"
                ):
                    run_benchmark.command_run(arguments)
            self.assertEqual(report.read_bytes(), b"previous")


@unittest.skipUnless(platform.system() == "Linux", "Linux measurement helper only")
class NativeMeasurementHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            raise unittest.SkipTest("C compiler is unavailable")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.helper = cls.root / "measure-linux"
        source = run_benchmark.PROJECT_ROOT / "benchmarks" / "measure_linux.c"
        completed = subprocess.run(
            [
                compiler,
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(source),
                "-o",
                str(cls.helper),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            raise AssertionError(f"measure_linux.c did not compile:\n{completed.stderr}")

    def invoke(self, timeout_ms: int, command: list[str]) -> tuple[subprocess.CompletedProcess[str], dict[str, object], Path, Path]:
        stdout_path = self.root / "captured.stdout"
        stderr_path = self.root / "captured.stderr"
        completed = subprocess.run(
            [
                str(self.helper),
                str(timeout_ms),
                str(stdout_path),
                str(stderr_path),
                "--",
                *command,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        metrics = run_benchmark.parse_metrics(completed.stdout)
        return completed, metrics, stdout_path, stderr_path

    def test_success_reports_elapsed_time_peak_rss_and_captured_output(self) -> None:
        completed, metrics, stdout_path, stderr_path = self.invoke(
            1000, ["/bin/sh", "-c", "printf 'native-ok\\n'"]
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertGreater(metrics["elapsed_ns"], 0)
        self.assertGreater(metrics["peak_rss_kib"], 0)
        self.assertEqual(metrics["exit_code"], 0)
        self.assertEqual(metrics["term_signal"], 0)
        self.assertFalse(metrics["timed_out"])
        self.assertEqual(stdout_path.read_bytes(), b"native-ok\n")
        self.assertEqual(stderr_path.read_bytes(), b"")

    def test_nonzero_exit_and_signal_are_reported_exactly(self) -> None:
        _, exited, _, _ = self.invoke(1000, ["/bin/sh", "-c", "exit 7"])
        self.assertEqual(exited["exit_code"], 7)
        self.assertEqual(exited["term_signal"], 0)

        _, signaled, _, _ = self.invoke(
            1000, ["/bin/sh", "-c", "kill -TERM $$"]
        )
        self.assertEqual(signaled["exit_code"], -1)
        self.assertEqual(signaled["term_signal"], 15)

    def test_timeout_kills_and_reaps_the_child(self) -> None:
        _, metrics, _, _ = self.invoke(20, ["/bin/sh", "-c", "sleep 2"])
        self.assertTrue(metrics["timed_out"])
        self.assertEqual(metrics["exit_code"], -1)
        self.assertEqual(metrics["term_signal"], 9)
        self.assertLess(metrics["elapsed_ns"], 1_000_000_000)

    def test_invalid_timeout_and_missing_command_are_rejected(self) -> None:
        for arguments, message in (
            (["0", "out", "err", "--", "/bin/true"], "timeout must be"),
            (["10", "out", "err", "--"], "usage:"),
        ):
            with self.subTest(arguments=arguments):
                completed = subprocess.run(
                    [str(self.helper), *arguments],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(message, completed.stderr)


if __name__ == "__main__":
    unittest.main()
