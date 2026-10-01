"""Static admission and isolation checks for the separate E3 test tuple."""

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "native/e3_private_package"
RETENTION = ROOT / "tests/e3_private/retention_package"


def lines(prefix: Path, suffix: str) -> list[str]:
    return Path(f"{prefix}_{suffix}.txt").read_text().splitlines()


class RetentionTupleTest(unittest.TestCase):
    def test_exact_successor_inventories(self) -> None:
        successor_runtime = {
            "eshkol_format_float32_bits", "eshkol_runtime_fatal",
        }
        self.assertTrue(successor_runtime.issubset(
            set(lines(BASE, "undefined_symbols"))
        ))
        for suffix in ("native_objects",):
            self.assertEqual(lines(RETENTION, suffix), lines(BASE, suffix))
        self.assertEqual(
            lines(RETENTION, "undefined_symbols"),
            sorted(lines(BASE, "undefined_symbols")
                   + ["et_e3_retention_region_used_v1"]),
        )
        for suffix in ("defined_symbols", "public_exports", "public_strings"):
            expected = sorted(lines(BASE, suffix) + ["et_e3_retention_run_v1"])
            self.assertEqual(lines(RETENTION, suffix), expected)
        self.assertEqual(
            lines(RETENTION, "private_renames"),
            lines(BASE, "private_renames")
            + ["e3-test-retention-dispatch et_e3_private_retention_dispatch_cabi_v1"],
        )
        self.assertEqual(
            lines(RETENTION, "source_closure"),
            ["tests/e3_private/retention_driver_root.esk"]
            + lines(BASE, "source_closure")
            + ["tests/e3_private/retention_driver.esk"],
        )
        native = lines(BASE, "native_source_closure")
        native.insert(native.index("native/e3_private_bridge.c"),
                      "tests/e3_private/retention_bridge.c")
        # The retention-only E3 observer includes D2's allocation controls
        # before m3_model.c; preserve the compiler's first-seen dependency order.
        native.remove("native/d2_native.h")
        native.insert(native.index("src/eshkol_transformer/m3_model.c"),
                      "native/d2_native.h")
        self.assertEqual(lines(RETENTION, "native_source_closure"), native)
        self.assertEqual(lines(RETENTION, "archive_members"),
                         ["e3_retention_package.o"])

    def test_canonical_driver_has_no_retention_entry(self) -> None:
        canonical = (ROOT / "tests/e3_private/driver.esk").read_text()
        for name in ("e3-test-retention", "et_e3_test_retention_probe_v1",
                     "et_d2_exact_read_test_fail_stage_v1"):
            self.assertNotIn(name, canonical)
        self.assertNotIn("retention_driver.esk",
                         (ROOT / "native/e3_private_driver_root.esk").read_text())

    def test_policy_admits_only_the_exact_tuple(self) -> None:
        policy = ROOT / "scripts/e3-private-package-policy.sh"
        root = ROOT / "tests/e3_private/retention_driver_root.esk"
        bridge = ROOT / "tests/e3_private/retention_bridge.c"
        renames = Path(f"{RETENTION}_private_renames.txt")
        exports = Path(f"{RETENTION}_public_exports.txt")
        script = '''
PROJECT_ROOT=$1
raw_private_root=$2
raw_package_bridge=$3
raw_package_renames=$4
raw_public_exports=$5
raw_include_dirs=("${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/internal/d2/lib" "${PROJECT_ROOT}/internal/e3/lib")
die() { printf '%s\\n' "$*" >&2; exit 77; }
source "$6"
printf '%s\\n' "$e3_tuple_kind"
'''
        exact = subprocess.run(
            ["bash", "-c", script, "bash", str(ROOT), str(root), str(bridge),
             str(renames), str(exports), str(policy)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual((exact.returncode, exact.stdout), (0, "retention\n"),
                         exact.stderr)
        mixed = subprocess.run(
            ["bash", "-c", script, "bash", str(ROOT), str(root),
             str(ROOT / "native/e3_private_bridge.c"), str(renames),
             str(exports), str(policy)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(mixed.returncode, 77)
        self.assertIn("mixed", mixed.stderr)


if __name__ == "__main__":
    unittest.main()
