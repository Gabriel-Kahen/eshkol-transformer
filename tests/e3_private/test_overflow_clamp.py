"""Q0 controls for the separately admitted E3 clamp mutant."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.e3_private.parse_overflow_observation import parse
from tests.e3_private.capture_abort import capture
from tests.e3_private.test_overflow_tuple import transcript


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "e3_overflow_clamp_generator", ROOT / "scripts/generate-e3-overflow-clamp-overlay.py")
assert SPEC and SPEC.loader
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)


def mutant_transcript(gradient: str) -> bytes:
    ordinary = transcript("overflow", gradient)
    second = ordinary.find(b"E3ROW\t0", len(b"E3ROW\t0"))
    assert second > 0
    first = ordinary[:second]
    return (first +
            b"E3FINAL\t42b1ffc6,3f800000,7f7fffff,00000000\t1,1\n"
            b"E3-RETENTION-FAIL: overflow-stage12-error\n")


class ClampOverlayTests(unittest.TestCase):
    def test_source_hash_framing_and_exact_two_line_overlay(self) -> None:
        source = GEN.SOURCE.read_bytes()
        derived, receipt = GEN.derive(source)
        start = source.index(GEN.START)
        end = source.index(GEN.END, start)
        self.assertEqual(GEN.digest(source[start:end]), GEN.FUNCTION_SHA)
        self.assertEqual(GEN.digest(source[start:end + 1]), GEN.WITH_NEWLINE_SHA)
        self.assertNotEqual(GEN.FUNCTION_SHA, GEN.WITH_NEWLINE_SHA)
        self.assertEqual(GEN.digest(derived), GEN.DERIVED_SHA)
        self.assertEqual(receipt["function_end_exclusive"], end)
        self.assertEqual(source[:start], derived[:start])
        self.assertEqual(source[end:], derived[-len(source[end:]):])
        for damaged in (source + GEN.OLD, source.replace(GEN.OLD, GEN.OLD + b" ", 1),
                        source.replace(GEN.START, GEN.START + b" ", 1)):
            with self.assertRaises(ValueError):
                GEN.derive(damaged)

    def test_owned_stage_is_fresh_and_check_detects_tamper(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-clamp-q0-") as base:
            stage = Path(base) / ".e3-overflow-generated"
            stage.mkdir()
            output = stage / "native/e3_evaluation_metrics_provider.c"
            GEN.run(output)
            self.assertEqual(GEN.run(output, check=True)["derived_sha256"], GEN.DERIVED_SHA)
            with self.assertRaises(ValueError):
                GEN.run(output)
            output.write_bytes(output.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                GEN.run(output, check=True)
            output.unlink()
            output.symlink_to(GEN.SOURCE)
            with self.assertRaises(ValueError):
                GEN.run(output, check=True)
            other_stage = Path(base) / "second/.e3-overflow-generated"
            other_stage.mkdir(parents=True)
            (other_stage / "native").symlink_to(stage / "native", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "owned overflow native stage"):
                GEN.run(other_stage / "native/e3_evaluation_metrics_provider.c",
                        check=True)

    def test_selector_and_exact_native_dependency(self) -> None:
        policy = ROOT / "scripts/e3-overflow-clamp-policy.sh"
        body = r'''
die() { printf '%s\n' "$*" >&2; exit 1; }
PROJECT_ROOT=$1
package_policy=$2
e3_tuple_kind=$3
e3_private_test_sanitize=$4
source "$PROJECT_ROOT/scripts/e3-overflow-clamp-policy.sh"
printf '%s\n' "$e3_overflow_mutant"
'''
        def selected(policy_name: str, tuple_name: str, san: str, value: str | None):
            env = __import__("os").environ.copy()
            env.pop("E3_OVERFLOW_MUTANT", None)
            if value is not None:
                env["E3_OVERFLOW_MUTANT"] = value
            return subprocess.run(["bash", "-c", body, "bash", str(ROOT),
                                   policy_name, tuple_name, san], env=env,
                                  capture_output=True, text=True, check=False)
        self.assertTrue(policy.is_file())
        self.assertEqual(selected("e3-private-aggregate", "retention-overflow", "0", None).stdout,
                         "normal\n")
        self.assertEqual(selected("e3-private-aggregate", "retention-overflow", "0", "clamp").stdout,
                         "clamp\n")
        for args in (("e3-private-aggregate", "retention", "0", "clamp"),
                     ("e3-private-aggregate", "retention-overflow", "1", "clamp"),
                     ("e3-private-aggregate", "retention-overflow", "0", "normal"),
                     ("other", "retention-overflow", "0", "clamp")):
            self.assertEqual(selected(*args).returncode, 1)
        ordinary = (ROOT / "tests/e3_private/overflow_package_native_source_closure.txt").read_text()
        mutant = (ROOT / "tests/e3_private/overflow_package_clamp_native_source_closure.txt").read_text()
        self.assertEqual(mutant, ordinary.replace(
            "native/e3_evaluation_metrics_provider.c\n",
            "@BUILD@/native/e3_evaluation_metrics_provider.c\n"))
        self.assertEqual(mutant.count("@BUILD@/native/"), 1)
        with tempfile.TemporaryDirectory(prefix="e3-clamp-deps-") as base:
            source = Path(base) / "provider.c"
            source.write_bytes(b"exact-test-placeholder")
            extra = Path(base) / "extra.c"
            extra.write_bytes(b"not admitted")
            dependency = r'''
die() { exit 1; }
PROJECT_ROOT=$1
package_policy=e3-private-aggregate
e3_tuple_kind=retention-overflow
e3_private_test_sanitize=0
E3_OVERFLOW_MUTANT=clamp
source "$PROJECT_ROOT/scripts/e3-overflow-clamp-policy.sh"
e3_overflow_clamp_source=$2
e3_overflow_clamp_dependency "$3"
'''
            def admitted(path: Path):
                return subprocess.run(["bash", "-c", dependency, "bash", str(ROOT),
                                       str(source), str(path)], capture_output=True,
                                      text=True, check=False)
            self.assertEqual(admitted(source).stdout,
                             str(ROOT) + "/@BUILD@/native/e3_evaluation_metrics_provider.c\n")
            self.assertNotEqual(admitted(extra).returncode, 0)
            source.unlink()
            source.symlink_to(extra)
            self.assertNotEqual(admitted(source).returncode, 0)

    def test_builder_rejects_ambiguous_mutant_cli_before_toolchain(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-clamp-cli-") as base:
            script = ROOT / "scripts/build-e3-overflow-test.sh"
            for mode, variant, expected in (("normal", "unknown", "unknown E3 overflow mutation"),
                                            ("sanitize", "clamp", "cannot combine with sanitizer")):
                result = subprocess.run(["bash", str(script), str(Path(base) / "unused"),
                                         mode, variant], capture_output=True, text=True,
                                        check=False)
                self.assertEqual(result.returncode, 1)
                self.assertIn(expected, result.stderr)
            env = __import__("os").environ.copy()
            env["E3_OVERFLOW_MUTANT"] = "clamp"
            result = subprocess.run(["bash", str(script), str(Path(base) / "unused"),
                                     "normal", "normal"], env=env,
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 1)
            self.assertIn("selector must use", result.stderr)

    def test_full_clamp_trace_requires_final_marker_and_rejects_near_misses(self) -> None:
        for gradient in ("absent", "present"):
            good = mutant_transcript(gradient)
            self.assertEqual(parse(good, "overflow", gradient, mutant="clamp"), 1)
            for bad in (good.replace(b"7f7fffff", b"7f800000", 1),
                        good.replace(b"E3FINAL", b"E3FAULT", 1),
                        good.replace(b"E3-RETENTION-FAIL: overflow-stage12-error",
                                     b"E3-FINITE-OVERFLOW-PASS gradient=absent", 1),
                        good + b"unexpected\n", good[:-1]):
                with self.assertRaises(ValueError):
                    parse(bad, "overflow", gradient, mutant="clamp")

    def test_abort_capture_preserves_raw_and_canonical_status(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-clamp-abort-") as base:
            prefix = Path(base) / "signal"
            self.assertEqual(capture(prefix, [sys.executable, "-u", "-c",
                'import os; print("E3-SIGNAL-CONTROL"); os.abort()']), 134)
            self.assertEqual((Path(base) / "signal.stdout").read_bytes(),
                             b"E3-SIGNAL-CONTROL\n")
            self.assertEqual((Path(base) / "signal.exit").read_text(), "134\n")
            self.assertEqual((Path(base) / "signal.raw-exit").read_text(), "-6\n")
            with self.assertRaises(ValueError):
                capture(prefix, [sys.executable, "-c", "pass"])
            normal = Path(base) / "normal"
            self.assertEqual(capture(normal, [sys.executable, "-c", "pass"]), 0)
            forged = Path(base) / "forged"
            with self.assertRaisesRegex(ValueError, "without a SIGABRT"):
                capture(forged, [sys.executable, "-c", "raise SystemExit(134)"])

    def test_abort_wrapper_streams_must_both_be_empty(self) -> None:
        policy = ROOT / "scripts/e3-overflow-capture-policy.sh"
        runner = 'source "$1"; e3_overflow_capture_streams_empty "$2"'
        with tempfile.TemporaryDirectory(prefix="e3-clamp-capture-policy-") as base:
            prefix = Path(base) / "capture"
            stdout = Path(f"{prefix}.stdout")
            stderr = Path(f"{prefix}.stderr")
            stdout.write_bytes(b"")
            stderr.write_bytes(b"")

            def admitted() -> bool:
                result = subprocess.run(["bash", "-c", runner, "bash", str(policy),
                                         str(prefix)], capture_output=True, check=False)
                self.assertEqual(result.stdout, b"")
                self.assertEqual(result.stderr, b"")
                return result.returncode == 0

            self.assertTrue(admitted())
            stdout.write_bytes(b"unexpected wrapper stdout\n")
            self.assertFalse(admitted())
            stdout.write_bytes(b"")
            stderr.write_bytes(b"unexpected wrapper stderr\n")
            self.assertFalse(admitted())
            stderr.unlink()
            self.assertFalse(admitted())
            stderr.symlink_to(stdout)
            self.assertFalse(admitted())


if __name__ == "__main__":
    unittest.main()
