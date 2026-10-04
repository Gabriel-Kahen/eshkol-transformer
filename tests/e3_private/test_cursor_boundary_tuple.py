"""Exact source-only checks for the isolated E3 boundary tuple."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BOUNDARY = ROOT / "tests/e3_private/cursor_boundary_package"
RETENTION = ROOT / "tests/e3_private/retention_package"


def lines(prefix: Path, suffix: str) -> list[str]:
    return Path(f"{prefix}_{suffix}.txt").read_text().splitlines()


def policy(root: Path, bridge: Path, prefix: Path) -> subprocess.CompletedProcess[str]:
    script = '''
PROJECT_ROOT=$1
raw_private_root=$2
raw_package_bridge=$3
raw_package_renames=$4
raw_public_exports=$5
raw_include_dirs=("${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/internal/d2/lib" "${PROJECT_ROOT}/internal/e3/lib")
die() { printf '%s\\n' "$*" >&2; exit 77; }
source "$6"
printf '%s\\t%s\\n' "$e3_tuple_kind" "$e3_boundary_tuple"
'''
    return subprocess.run(
        ["bash", "-c", script, "bash", str(ROOT), str(root), str(bridge),
         f"{prefix}_private_renames.txt", f"{prefix}_public_exports.txt",
         str(ROOT / "scripts/e3-private-package-policy.sh")],
        capture_output=True, text=True, check=False,
    )


class CursorBoundaryTupleTests(unittest.TestCase):
    def test_exact_inventory_delta_is_test_only(self) -> None:
        for suffix in ("native_objects", "undefined_symbols", "archive_members"):
            self.assertEqual(lines(BOUNDARY, suffix), lines(RETENTION, suffix))
        for suffix in ("defined_symbols", "public_exports", "public_strings"):
            self.assertEqual(lines(BOUNDARY, suffix),
                             sorted(lines(RETENTION, suffix)
                                    + ["et_e3_cursor_boundary_run_v1"]))
        self.assertEqual(
            lines(BOUNDARY, "private_renames"),
            lines(RETENTION, "private_renames")
            + ["e3-test-cursor-boundary-dispatch "
               "et_e3_private_cursor_boundary_dispatch_cabi_v1"],
        )
        self.assertEqual(
            lines(BOUNDARY, "source_closure"),
            ["tests/e3_private/cursor_boundary_driver_root.esk"]
            + lines(RETENTION, "source_closure")
            + ["tests/e3_private/cursor_boundary_driver.esk"],
        )
        native = lines(RETENTION, "native_source_closure")
        native.insert(native.index("tests/e3_private/retention_bridge.c"),
                      "tests/e3_private/cursor_boundary_bridge.c")
        self.assertEqual(lines(BOUNDARY, "native_source_closure"), native)
        self.assertNotIn("cursor_boundary", (ROOT / "native/e3_private_driver_root.esk").read_text())

    def test_policy_admits_only_complete_exact_tuple(self) -> None:
        root = ROOT / "tests/e3_private/cursor_boundary_driver_root.esk"
        bridge = ROOT / "tests/e3_private/cursor_boundary_bridge.c"
        exact = policy(root, bridge, BOUNDARY)
        self.assertEqual((exact.returncode, exact.stdout),
                         (0, "retention\t1\n"), exact.stderr)
        near_misses = (
            (root, ROOT / "tests/e3_private/retention_bridge.c", BOUNDARY),
            (ROOT / "tests/e3_private/retention_driver_root.esk", bridge, BOUNDARY),
            (root, bridge, RETENTION),
            (ROOT / "native/e3_private_driver_root.esk", bridge, BOUNDARY),
        )
        for wrong_root, wrong_bridge, wrong_prefix in near_misses:
            with self.subTest(root=wrong_root, bridge=wrong_bridge, prefix=wrong_prefix):
                rejected = policy(wrong_root, wrong_bridge, wrong_prefix)
                self.assertEqual(rejected.returncode, 77)
        with tempfile.TemporaryDirectory(prefix="e3-boundary-alias-") as temporary:
            alias = Path(temporary) / "root.esk"
            alias.symlink_to(root)
            self.assertEqual(policy(alias, bridge, BOUNDARY).returncode, 77)

    def test_gate_and_restoration_observers_are_explicit(self) -> None:
        driver = (ROOT / "tests/e3_private/cursor_boundary_driver.esk").read_text()
        gate = (ROOT / "scripts/test-e3-cursor-boundary.sh").read_text()
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        for contract in (
            "d2-core-bytes=?", "e3-test-retention-modes=?",
            "e3-test-retention-model=?", "e3-boundary-probes=?",
            "e3-boundary-output=?", "transformer-error-message",
            "transformer-error-category", "transformer-error-operation",
            "e3-test-seed-gradient", "e3-evaluate-into!",
        ):
            self.assertIn(contract, driver)
        self.assertIn("et_e3_private_cursor_boundary_dispatch_cabi_v1", builder)
        self.assertIn('"${e3_boundary_tuple:-0}" == 0', builder)
        self.assertIn("--strict-types --no-stdlib -O 2", gate)
        self.assertIn("diff -r \"${out}/corpus-a\" \"${out}/corpus-b\"", gate)
        self.assertIn("ESHKOL_ARENA_POISON=1", gate)
        self.assertIn('"${out}/corpus-a/one-over"', gate)
        self.assertIn("(= (vector-length paths) 5)", driver)
        self.assertIn("'one-over-stage-two-unsupported", driver)
        self.assertIn("'one-over-no-acquire-or-traversal", driver)
        self.assertIn("'source-code 2", driver)
        self.assertIn("(= (vector-ref core 14) 8193)", driver)
        self.assertIn("(= (vector-ref core 15) 0)", driver)
        self.assertEqual(
            (ROOT / "tests/e3_private/cursor_boundary_driver_root.esk").read_text().splitlines()[-2:],
            ['(load "retention_driver_root.esk")', '(load "cursor_boundary_driver.esk")'],
        )
        self.assertIn('#include "retention_bridge.c"',
                      (ROOT / "tests/e3_private/cursor_boundary_bridge.c").read_text())

    def test_sanitized_boundary_imports_are_exact_and_fail_closed(self) -> None:
        self.assertEqual(lines(BOUNDARY, "undefined_symbols"),
                         lines(RETENTION, "undefined_symbols"))
        non = ROOT / "tests/e3_private/retention_package_sanitized_noninstrumentation_undefined_symbols.txt"
        instrumentation = ROOT / "tests/e3_private/retention_package_sanitized_instrumentation_undefined_symbols.txt"
        self.assertEqual((len(non.read_text().splitlines()),
                          len(instrumentation.read_text().splitlines())), (175, 51))
        check = '''
PROJECT_ROOT=$1
die() { printf '%s\\n' "$*" >&2; exit 77; }
source "$PROJECT_ROOT/scripts/e3-retention-sanitizer-import-policy.sh"
e3_retention_verify_sanitizer_imports "$2" "$3" "$4"
'''
        normal = Path(f"{BOUNDARY}_undefined_symbols.txt")

        def verify(non_path: Path, instrumentation_path: Path) -> int:
            return subprocess.run(
                ["bash", "-c", check, "bash", str(ROOT), str(normal),
                 str(non_path), str(instrumentation_path)],
                capture_output=True, text=True, check=False,
            ).returncode

        self.assertEqual(verify(non, instrumentation), 0)
        with tempfile.TemporaryDirectory(prefix="e3-boundary-imports-") as temporary:
            damaged = Path(temporary)
            extra = damaged / "non-extra.txt"
            extra.write_text(non.read_text() + "unexpected_boundary_import\n")
            absent = damaged / "instrumentation-absent.txt"
            absent.write_text("\n".join(instrumentation.read_text().splitlines()[:-1]) + "\n")
            self.assertEqual(verify(extra, instrumentation), 77)
            self.assertEqual(verify(non, absent), 77)

    def test_sanitizer_is_bound_to_private_tuple_and_all_five_cases(self) -> None:
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        package = (ROOT / "scripts/build-e3-cursor-boundary-test.sh").read_text()
        gate = (ROOT / "scripts/test-e3-cursor-boundary.sh").read_text()
        self.assertIn('"${e3_tuple_kind}" == boundary', builder)
        self.assertIn('"${e3_private_test_sanitize}" == 0', builder)
        self.assertIn('E3_CURSOR_BOUNDARY_SANITIZE="${sanitize}"', package)
        self.assertIn('"${sanitized}" sanitize', gate)
        self.assertIn('e3_retention_verify_sanitizer_imports', gate)
        self.assertIn('"${evidence}/native-depfiles"', gate)
        self.assertIn('"${out}/runtime-sanitized"', gate)
        self.assertIn('ASAN_OPTIONS=detect_leaks=1:halt_on_error=1', gate)
        self.assertIn('UBSAN_OPTIONS=halt_on_error=1', gate)
        self.assertIn('"${out}/corpus-a/one-over"', gate)
        self.assertIn('"\\$@"', gate)


if __name__ == "__main__":
    unittest.main()
