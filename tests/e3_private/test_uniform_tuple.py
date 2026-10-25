"""Noninstalled uniform-tie tuple and private fault-hook boundaries."""

from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from tests.e3_private.parse_uniform_observation import MARKER, parse_uniform
from tests.e3_private.test_numerical_tuple import frame_bytes, frame_lines


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests/e3_private/numerical_package"
UNIFORM = ROOT / "tests/e3_private/uniform_package"


def lines(prefix: Path, suffix: str) -> list[str]:
    return Path(f"{prefix}_{suffix}.txt").read_text().splitlines()


class UniformTupleTest(unittest.TestCase):
    def test_zero_state_load_uses_first_model_initializer_successor(self) -> None:
        from tests.m3 import reference as m3

        m3.setup()
        _, successor = m3.parameters(1729)
        self.assertEqual(successor, (1, 1729, 290, 0))
        source = (ROOT / "tests/e3_private/uniform_driver.esk").read_text()
        fixture = source.split("(define (e3-test-fixture)", 1)[1].split(
            "(define (e3-test-scenario", 1)[0]
        self.assertEqual(fixture.count("(g3t-checked-m3t-model-create config initial)"), 1)
        self.assertLess(fixture.index("(e3-uniform-load-before-frame! model)"),
                        fixture.index("(e3-frame-bind! model destinations)"))
        match = re.search(r"'seed1729-successor\s*\(equal\? words '\(([^)]*)\)\)", source)
        self.assertIsNotNone(match)
        self.assertEqual(tuple(map(int, match.group(1).split())), successor)

    def test_completed_state_marker_and_closed_frame_records(self) -> None:
        frame = frame_bytes(frame_lines(3))
        complete = MARKER + frame[:-len(b"E3-NUMERICAL-RUNTIME-PASS\n")] + (
            b"E3CURSOR\t00\t00\nE3-NUMERICAL-RUNTIME-PASS\n"
        )
        observed, roles = parse_uniform(complete, 3, "00")
        self.assertEqual(len(observed["batches"]), 3)
        self.assertEqual(len(roles.splitlines()), 63)
        for malformed in (frame, MARKER + complete, complete + b"extra\n",
                          complete[:-1], complete.replace(b"\t00\t00", b"\t00\t01")):
            with self.subTest(malformed=malformed[-20:]):
                with self.assertRaises(ValueError):
                    parse_uniform(malformed, 3, "00")

    def test_exact_closed_manifest_topology(self) -> None:
        for suffix in (
            "native_objects", "defined_symbols", "public_exports",
            "public_strings", "private_renames",
            "sanitized_added_undefined_symbols", "sanitized_absent_undefined_symbols",
        ):
            self.assertEqual(lines(UNIFORM, suffix), lines(BASE, suffix), suffix)
        imports = set(lines(BASE, "undefined_symbols"))
        uniform_imports = set(lines(UNIFORM, "undefined_symbols"))
        self.assertEqual(uniform_imports - imports, {
            "eshkol_display_value", "eshkol_runtime_current_output_fp",
            "eshkol_tail_transfer_slot", "fputc",
        })
        self.assertEqual(imports - uniform_imports, set())
        self.assertEqual(len(uniform_imports), 177)
        source = lines(BASE, "source_closure")
        source[0] = "tests/e3_private/uniform_driver_root.esk"
        source.remove("native/e3_private_driver_root.esk")
        source[-1] = "tests/e3_private/uniform_driver.esk"
        self.assertEqual(lines(UNIFORM, "source_closure"), source)
        native = lines(BASE, "native_source_closure")
        native[native.index("tests/e3_private/numerical_bridge.c")] = (
            "tests/e3_private/uniform_bridge.c"
        )
        self.assertEqual(lines(UNIFORM, "native_source_closure"), native)
        self.assertEqual(lines(UNIFORM, "archive_members"),
                         ["e3_uniform_package.o"])
        self.assertIn("native/i2_wave2_root.esk", source)
        self.assertIn("native/i2_wave2_package_bridge.c", native)

    def test_uniform_import_policy_rejects_manifest_drift(self) -> None:
        suffixes = ("undefined_symbols", "sanitized_added_undefined_symbols",
                    "sanitized_absent_undefined_symbols")
        policy = ROOT / "scripts/e3-uniform-import-policy.sh"

        def verify(change=None, tuple_kind="uniform"):
            with tempfile.TemporaryDirectory(prefix="e3-uniform-imports-") as temporary:
                files = []
                for prefix in (BASE, UNIFORM):
                    for suffix in suffixes:
                        target = Path(temporary) / f"{prefix.name}_{suffix}.txt"
                        target.write_text("\n".join(lines(prefix, suffix)) + "\n")
                        files.append(target)
                if change:
                    index, old, new = change
                    rows = files[index].read_text().splitlines()
                    if old is None:
                        rows.append(new)
                    elif new is None:
                        rows.remove(old)
                    else:
                        rows[rows.index(old)] = new
                    files[index].write_text("\n".join(sorted(rows)) + "\n")
                command = ('set -euo pipefail; die() { echo "$*" >&2; exit 77; }; '
                           'e3_tuple_kind=$1; source "$2"; '
                           'e3_uniform_verify_import_manifests "$3" "$6" '
                           '"$4" "$7" "$5" "$8"')
                return subprocess.run(
                    ["bash", "-c", command, "bash", tuple_kind, str(policy),
                     *(str(path) for path in files)],
                    capture_output=True, text=True, check=False,
                )

        self.assertEqual(verify().returncode, 0)
        self.assertEqual(verify(tuple_kind="numerical").returncode, 77)
        for name in ("eshkol_display_value", "eshkol_runtime_current_output_fp",
                     "eshkol_tail_transfer_slot", "fputc"):
            with self.subTest(missing=name):
                self.assertEqual(verify((3, name, None)).returncode, 77)
        for change in ((3, None, "zz_unreviewed_import"),
                       (3, "fputc", "zz_unreviewed_import"),
                       (4, None, "zz_unreviewed_import"),
                       (5, None, "zz_unreviewed_import")):
            with self.subTest(change=change):
                self.assertEqual(verify(change).returncode, 77)

    def test_exact_tuple_and_mixed_rejection(self) -> None:
        policy = ROOT / "scripts/e3-private-package-policy.sh"
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
        root = ROOT / "tests/e3_private/uniform_driver_root.esk"
        bridge = ROOT / "tests/e3_private/uniform_bridge.c"
        renames = Path(f"{UNIFORM}_private_renames.txt")
        exports = Path(f"{UNIFORM}_public_exports.txt")

        def choose(*, bridge_path: Path = bridge, export_path: Path = exports):
            return subprocess.run(
                ["bash", "-c", script, "bash", str(ROOT), str(root),
                 str(bridge_path), str(renames), str(export_path), str(policy)],
                capture_output=True, text=True, check=False,
            )

        accepted = choose()
        self.assertEqual((accepted.returncode, accepted.stdout), (0, "uniform\n"),
                         accepted.stderr)
        for mismatch in (choose(bridge_path=ROOT / "native/e3_private_bridge.c"),
                         choose(export_path=Path(f"{BASE}_public_exports.txt"))):
            self.assertEqual(mismatch.returncode, 77)
            self.assertIn("mixed", mismatch.stderr)

    def test_stage_and_test_hook_scope(self) -> None:
        helper = ROOT / "scripts/e3-generated-source-stage.sh"
        with tempfile.TemporaryDirectory(prefix="e3-uniform-gate-") as temporary:
            gate = Path(temporary)
            stage = gate / ".e3-uniform.ABC123"
            stage.mkdir()

            def select(path: Path):
                return subprocess.run(
                    ["bash", "-c", 'source "$1"; e3_uniform_generated_stage "$2"',
                     "bash", str(helper), str(path)], capture_output=True,
                    text=True, check=False,
                )

            good = select(stage / "e3_uniform_package.o")
            self.assertEqual((good.returncode, good.stdout),
                             (0, str(gate / ".e3-uniform-generated") + "\n"),
                             good.stderr)
            self.assertNotEqual(select(stage / "e3_numerical_package.o").returncode, 0)
            (gate / ".e3-uniform-generated").symlink_to(stage, target_is_directory=True)
            self.assertNotEqual(select(stage / "e3_uniform_package.o").returncode, 0)
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        metrics = (ROOT / "native/e3_evaluation_metrics_provider.c").read_text()
        bridge = (ROOT / "tests/e3_private/uniform_bridge.c").read_text()
        self.assertIn('"${e3_tuple_kind}" == uniform', builder)
        self.assertIn("-DET_F32_TENSOR_TESTING", builder)
        self.assertIn("-DET_E3_UNIFORM_LAST_TIE_TEST", builder)
        self.assertIn("#ifdef ET_E3_UNIFORM_LAST_TIE_TEST", metrics)
        for hook in ("et_i2_test_live_builder_counts_v1",
                     "et_i2_test_retired_builder_counts_v1",
                     "et_i2_test_decode_builder_control_bytes_v1"):
            self.assertIn(hook, bridge)
        for hook in ("et_i2_test_fail_alloc_after_v1",
                     "et_i2_test_reset_allocator_v1"):
            self.assertIn(hook, (ROOT / "tests/e3_private/uniform_driver.esk").read_text())


if __name__ == "__main__":
    unittest.main()
