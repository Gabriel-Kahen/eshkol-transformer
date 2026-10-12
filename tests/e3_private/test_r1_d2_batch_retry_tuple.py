"""Source-only admission and predicted inventories for the separate D2 tuple."""

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "tests/e3_private"


def rows(kind: str, name: str) -> list[str]:
    return (DIR / f"{kind}_package_{name}.txt").read_text().splitlines()


def runtime_matrix_errors(source: str) -> list[str]:
    required = (
        "for mode in normal-a normal-b normal-repeat poison sanitized; do",
        "normal-a|normal-repeat) binary=a; poison=0 ;;",
        "normal-b) binary=b; poison=0 ;;",
        "poison) binary=a; poison=1 ;;",
        "sanitized) binary=sanitized; poison=1 ;;",
        '*) die "unknown R1 runtime mode: $mode" ;;',
        'capture "runtime-$mode" 900 env ESHKOL_ARENA_POISON="$poison"',
        '[[ ! -s "$out/runtime-$mode.stderr" ]]',
        "for mode in normal-b normal-repeat poison sanitized; do",
        'cmp "$out/runtime-normal-a.stdout" "$out/runtime-$mode.stdout"',
        'cmp "$out/runtime-normal-a.stderr" "$out/runtime-$mode.stderr"',
        'cmp "$out/runtime-normal-a.exit" "$out/runtime-$mode.exit"',
    )
    return [item for item in required if source.count(item) != 1]


class R1D2BatchTuple(unittest.TestCase):
    def test_predicted_exact_delta(self) -> None:
        old, new = "r1_source_retry", "r1_d2_batch_retry"
        for name in ("undefined_symbols", "native_objects", "archive_members",
                     "sanitized_instrumentation_undefined_symbols",
                     "sanitized_noninstrumentation_undefined_symbols"):
            self.assertEqual(rows(new, name), rows(old, name), name)
        export = "et_e3_r1_d2_batch_retry_run_v1"
        for name in ("defined_symbols", "public_exports", "public_strings"):
            self.assertEqual(rows(new, name), sorted(rows(old, name) + [export]), name)
        self.assertEqual(rows(new, "source_closure"),
                         ["tests/e3_private/r1_d2_batch_retry_driver_root.esk"] +
                         rows(old, "source_closure") +
                         ["tests/e3_private/r1_d2_batch_retry_driver.esk"])
        native = rows(old, "native_source_closure")
        native.insert(native.index("tests/e3_private/r1_source_retry_bridge.c"),
                      "tests/e3_private/r1_d2_batch_retry_bridge.c")
        self.assertEqual(rows(new, "native_source_closure"), native)
        self.assertEqual(rows(new, "private_renames"), rows(old, "private_renames") + [
            "e3-r1-d2-batch-retry-run "
            "et_e3_private_r1_d2_batch_retry_run_cabi_v1"])

    def test_exact_policy_rejects_mixed_tuple(self) -> None:
        inputs = [DIR / "r1_d2_batch_retry_driver_root.esk",
                  DIR / "r1_d2_batch_retry_bridge.c",
                  DIR / "r1_d2_batch_retry_package_private_renames.txt",
                  DIR / "r1_d2_batch_retry_package_public_exports.txt"]
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
        policy = ROOT / "scripts/e3-private-package-policy.sh"
        def invoke(values: list[Path]) -> subprocess.CompletedProcess[str]:
            return subprocess.run(["bash", "-c", script, "bash", str(ROOT),
                                   *(str(x) for x in values), str(policy)],
                                  capture_output=True, text=True)
        exact = invoke(inputs)
        self.assertEqual((exact.returncode, exact.stdout),
                         (0, "r1-d2-batch-retry\n"), exact.stderr)
        for index, old_name in ((0, "r1_source_retry_driver_root.esk"),
                                (1, "r1_source_retry_bridge.c"),
                                (2, "r1_source_retry_package_private_renames.txt"),
                                (3, "r1_source_retry_package_public_exports.txt")):
            mixed = inputs.copy()
            mixed[index] = DIR / old_name
            self.assertEqual(invoke(mixed).returncode, 77, old_name)

    def test_test_only_macro_and_noninstalled_entry(self) -> None:
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        frame_flag = builder.split('"${PROJECT_ROOT}/src/eshkol_transformer/e3_frame.c")', 1)[-1]
        self.assertIn('"${e3_tuple_kind}" == r1-d2-batch-retry', frame_flag)
        self.assertIn("package_source_flags+=(-DET_E3_R1_PREFLIGHT_TESTING)", frame_flag)
        self.assertIn('"${PROJECT_ROOT}/native/e3_d2_native.c")\n'
                      '          package_source_flags+=(-DET_D2_NATIVE_TESTING)', frame_flag)
        self.assertIn("et_d2_batch_test_fail_stage_v1", builder)
        self.assertIn("r1-d2-batch-retry)",
                      (ROOT / "scripts/e3-retention-sanitizer-import-policy.sh").read_text())
        bridge = (DIR / "r1_d2_batch_retry_bridge.c").read_text()
        self.assertEqual(bridge.count('#include "r1_source_retry_bridge.c"'), 1)
        self.assertEqual(bridge.count("void et_e3_r1_d2_batch_retry_run_v1("), 1)
        self.assertNotIn("r1_d2_batch_retry",
                         (ROOT / "native/e3_private_driver_root.esk").read_text())
        script = (ROOT / "scripts/test-e3-r1-d2-batch-retry.sh").read_text()
        self.assertIn("et_d2_batch_test_fail_stage_v1$", script)
        self.assertIn("E3-R1-D2-BATCH-RETRY-PASS", script)

    def test_only_declared_retention_supplement_is_unresolved(self) -> None:
        observer = "et_e3_retention_region_used_v1"
        self.assertEqual(rows("r1_d2_batch_retry", "undefined_symbols").count(observer), 1)
        policy = ROOT / "scripts/e3-private-unresolved-policy.sh"
        command = '''
set -euo pipefail
die() { exit 77; }
source "$1"
e3_reject_unresolved_private "$2" "$3" "$4"
'''
        with tempfile.TemporaryDirectory() as temporary:
            actual = Path(temporary) / "actual"
            expected = Path(temporary) / "expected"
            def check(kind: str, symbols: list[str], declared: list[str]) -> int:
                actual.write_text("\n".join(symbols) + "\n")
                expected.write_text("\n".join(declared) + "\n")
                return subprocess.run(["bash", "-c", command, "bash", str(policy),
                                       kind, str(actual), str(expected)],
                                      capture_output=True).returncode
            self.assertEqual(check("r1-d2-batch-retry", [observer], [observer]), 0)
            self.assertEqual(check("r1-d2-batch-retry", [observer], []), 77)
            self.assertEqual(check("r1-d2-batch-retry",
                                   [observer, "et_d2_private_cursor_v1"],
                                   [observer, "et_d2_private_cursor_v1"]), 77)
            self.assertEqual(check("base", [observer], [observer]), 77)

    def test_normal_repeat_poison_sanitizer_matrix(self) -> None:
        source = (ROOT / "scripts/test-e3-r1-d2-batch-retry.sh").read_text()
        self.assertEqual(runtime_matrix_errors(source), [])
        for old, new in (
            ("normal-a normal-b normal-repeat poison sanitized", "normal-a normal-b normal-repeat sanitized"),
            ("normal-a|normal-repeat) binary=a; poison=0", "normal-a|normal-repeat) binary=a; poison=1"),
            ("normal-b) binary=b; poison=0", "normal-b) binary=b; poison=1"),
            ("poison) binary=a; poison=1", "poison) binary=a; poison=0"),
            ("sanitized) binary=sanitized; poison=1", "sanitized) binary=sanitized; poison=0"),
            ('cmp "$out/runtime-normal-a.exit" "$out/runtime-$mode.exit"', ":"),
            ('[[ ! -s "$out/runtime-$mode.stderr" ]]', ":"),
        ):
            self.assertTrue(runtime_matrix_errors(source.replace(old, new, 1)), old)


if __name__ == "__main__":
    unittest.main()
