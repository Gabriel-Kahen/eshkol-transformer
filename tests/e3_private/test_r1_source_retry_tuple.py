"""Static, provisional inventory and isolation controls for the R1 test tuple.

Only emitted fe9 package and depfile evidence can admit these predictions.
"""

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / "tests/e3_private"


def rows(kind: str, name: str) -> list[str]:
    return (PREFIX / f"{kind}_package_{name}.txt").read_text().splitlines()


class R1SourceRetryTuple(unittest.TestCase):
    def test_provisional_delta_from_retention(self) -> None:
        old, new = "retention", "r1_source_retry"
        for name in ("undefined_symbols", "native_objects", "archive_members",
                     "sanitized_instrumentation_undefined_symbols",
                     "sanitized_noninstrumentation_undefined_symbols"):
            self.assertEqual(rows(new, name), rows(old, name), name)
        for name in ("defined_symbols", "public_exports", "public_strings"):
            self.assertEqual(rows(new, name), sorted(rows(old, name) +
                             ["et_e3_r1_source_retry_run_v1"]), name)
        self.assertEqual(rows(new, "source_closure"),
                         ["tests/e3_private/r1_source_retry_driver_root.esk"] +
                         rows(old, "source_closure") +
                         ["tests/e3_private/r1_source_retry_driver.esk"])
        native = rows(old, "native_source_closure")
        native.insert(native.index("tests/e3_private/retention_bridge.c"),
                      "tests/e3_private/r1_source_retry_bridge.c")
        self.assertEqual(rows(new, "native_source_closure"), native)
        self.assertEqual(rows(new, "private_renames"),
                         rows(old, "private_renames") + [
                             "e3-r1-source-retry-run "
                             "et_e3_private_r1_source_retry_run_cabi_v1"])

    def test_exact_tuple_and_test_only_native_flag(self) -> None:
        root = PREFIX / "r1_source_retry_driver_root.esk"
        bridge = PREFIX / "r1_source_retry_bridge.c"
        policy = ROOT / "scripts/e3-private-package-policy.sh"
        script = r'''
PROJECT_ROOT=$1
raw_private_root=$2
raw_package_bridge=$3
raw_package_renames=$4
raw_public_exports=$5
raw_include_dirs=("$1/internal/p1/lib" "$1/internal/c1/lib" "$1/internal/t1/lib" "$1/src" "$1/internal/d2/lib" "$1/internal/e3/lib")
die() { exit 77; }
source "$6"
printf '%s\n' "$e3_tuple_kind"
'''
        inputs = [str(root), str(bridge),
                  str(PREFIX / "r1_source_retry_package_private_renames.txt"),
                  str(PREFIX / "r1_source_retry_package_public_exports.txt")]
        exact = subprocess.run(["bash", "-c", script, "bash", str(ROOT),
                                *inputs, str(policy)], capture_output=True, text=True)
        self.assertEqual((exact.returncode, exact.stdout),
                         (0, "r1-source-retry\n"), exact.stderr)
        mixed = subprocess.run(["bash", "-c", script, "bash", str(ROOT),
                                str(PREFIX / "retention_driver_root.esk"),
                                *inputs[1:], str(policy)], capture_output=True)
        self.assertEqual(mixed.returncode, 77)
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        self.assertIn('"${e3_tuple_kind}" == r1-source-retry', builder)
        self.assertIn('package_source_flags+=(-DET_E3_R1_PREFLIGHT_TESTING)', builder)
        self.assertIn('et_e3_test_preinvoke_arm_v1', builder)
        self.assertNotIn("r1_source_retry_driver_root.esk",
                         (ROOT / "native/e3_private_driver_root.esk").read_text())


if __name__ == "__main__":
    unittest.main()
