"""Source-only N5 tuple admission; no compiled E3 claim."""

from __future__ import annotations

from pathlib import Path
import subprocess
import unittest
from tests.c2.test_handler_order import parse as parse_source

ROOT = Path(__file__).resolve().parents[2]
N5 = ROOT / "tests/e3_n5"
BASE = ROOT / "native/e3_private_package"


def lines(prefix: Path, suffix: str) -> list[str]:
    return Path(f"{prefix}_{suffix}.txt").read_text().splitlines()


class N5TupleSourceTests(unittest.TestCase):
    def test_exact_private_tuple_manifests(self) -> None:
        for suffix in ("defined_symbols", "public_exports", "public_strings"):
            self.assertEqual(lines(N5 / "package", suffix),
                             sorted(lines(BASE, suffix) + ["et_e3_n5_run_v1"]))
        for suffix in ("native_objects", "undefined_symbols"):
            self.assertEqual(lines(N5 / "package", suffix), lines(BASE, suffix))
        self.assertEqual(lines(N5 / "package", "archive_members"),
                         ["e3_n5_package.o"])
        self.assertEqual(lines(N5 / "package", "private_renames"),
                         lines(BASE, "private_renames") +
                         ["e3-n5-dispatch et_e3_private_n5_dispatch_cabi_v1"])
        self.assertEqual(lines(N5 / "package", "source_closure"),
                         ["tests/e3_n5/driver_root.esk"] +
                         lines(BASE, "source_closure") +
                         ["tests/e3_n5/driver.esk"])
        native = lines(BASE, "native_source_closure")
        native.insert(native.index("native/e3_private_bridge.c"),
                      "tests/e3_n5/bridge.c")
        self.assertEqual(lines(N5 / "package", "native_source_closure"), native)

    def test_policy_rejects_mixed_tuple(self) -> None:
        policy = ROOT / "scripts/e3-private-package-policy.sh"
        script = '''set -euo pipefail
die() { printf '%s\\n' "$*" >&2; exit 77; }
PROJECT_ROOT=$1
raw_private_root=$2
raw_package_bridge=$3
raw_package_renames=$4
raw_public_exports=$5
raw_include_dirs=("$PROJECT_ROOT/internal/p1/lib" "$PROJECT_ROOT/internal/c1/lib"
  "$PROJECT_ROOT/internal/t1/lib" "$PROJECT_ROOT/src"
  "$PROJECT_ROOT/internal/d2/lib" "$PROJECT_ROOT/internal/e3/lib")
source "$PROJECT_ROOT/scripts/e3-private-package-policy.sh"
printf '%s\\n' "$e3_tuple_kind"
'''
        args = [str(ROOT), str(N5 / "driver_root.esk"),
                str(N5 / "bridge.c"), str(N5 / "package_private_renames.txt"),
                str(N5 / "package_public_exports.txt")]

        def select(inputs: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.run(["/usr/bin/bash", "-c", script, "policy", *inputs],
                                  capture_output=True, text=True)

        valid = select(args)
        self.assertEqual((valid.returncode, valid.stdout), (0, "n5\n"), valid.stderr)
        for index, foreign in ((1, ROOT / "native/e3_private_driver_root.esk"),
                               (2, ROOT / "native/e3_private_bridge.c"),
                               (3, ROOT / "native/e3_private_package_private_renames.txt")):
            wrong = args.copy()
            wrong[index] = str(foreign)
            self.assertNotEqual(select(wrong).returncode, 0)

    def test_real_evaluator_and_atomic_checks(self) -> None:
        driver = (N5 / "driver.esk").read_text()
        self.assertEqual(len(parse_source(driver)), 4)
        self.assertGreater(len(parse_source((N5 / "runtime.esk").read_text())), 0)
        self.assertEqual(driver.count("(e3-evaluate-into! model"), 3)
        self.assertLess(driver.index("'e3-n5-seed"), driver.index("'e3-n5-one-over"))
        self.assertLess(driver.index("'e3-n5-cap-control"), driver.index("'e3-n5-one-over"))
        self.assertIn("'control-batches (= (e3-counter-ref frame 0) 8192)", driver)
        self.assertIn("'control-tokens (= (e3-counter-ref frame 1) 16384)", driver)
        self.assertIn("(d2-token-dataset-open", driver)
        self.assertIn("(e3-error-table-ref 2 'unsupported)", driver)
        self.assertIn("'diagnostic-stage2", driver)
        self.assertIn("'cursor-atomic", driver)
        self.assertIn("'six-outputs-atomic", driver)
        self.assertIn("'modes-atomic", driver)
        self.assertIn("'no-owner-after-rejection", driver)
        self.assertIn("(d2-core-bytes=? before (d2-token-dataset-cursor dataset))", driver)
        self.assertNotIn("et_e3_n5_run_v1", (ROOT / "native/e3_private_bridge.c").read_text())
        self.assertNotIn("e3-n5-dispatch", (ROOT / "native/e3_private_driver_root.esk").read_text())

    def test_gate_binds_the_successful_control_and_refuses_unmeasured_sanitizer(self) -> None:
        gate = (ROOT / "scripts/test-e3-n5-private.sh").read_text()
        runtime = (N5 / "runtime.esk").read_text()
        self.assertIn("cap-8192", gate)
        self.assertIn("control=8192 stage=2 rows=8193", gate)
        self.assertIn("control=8192 stage=2 rows=8193", runtime)
        self.assertIn("N5 sanitizer import inventory is not pinned", gate)
        self.assertIn("$4 <= 524288", gate)

    def test_builder_and_gate_are_closed(self) -> None:
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        self.assertIn("E3_N5_SANITIZE", builder)
        self.assertIn("et_e3_private_n5_dispatch_cabi_v1", builder)
        self.assertIn("e3_n5_generated_stage", builder)
        gate = (ROOT / "scripts/test-e3-n5-private.sh").read_text()
        for required in ("build-e3-private.sh", "build-e3-n5-test.sh",
                         "tests.e3_n5.prepare", "cap-8193", "ASAN_OPTIONS",
                         "UBSAN_OPTIONS", "for variant in a b poison sanitized",
                         '"${out}/runtime-${variant}"'):
            self.assertIn(required, gate)


if __name__ == "__main__":
    unittest.main()
