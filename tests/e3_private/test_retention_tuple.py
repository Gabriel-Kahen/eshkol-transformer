"""Static admission and isolation checks for the separate E3 test tuple."""

from pathlib import Path
import subprocess
import tempfile
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
                   + ["eshkol_builtin_arena_used",
                      "eshkol_display_value",
                      "eshkol_runtime_current_output_fp",
                      "eshkol_tail_transfer_slot",
                      "et_e3_retention_region_used_v1", "fputc"]),
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
            + ["@BUILD@/source/transformer/module.esk"
               if item == "internal/p1/lib/transformer/module.esk" else item
               for item in lines(BASE, "source_closure")]
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

    def test_retention_uses_only_authenticated_p1_token(self) -> None:
        source = (ROOT / "tests/e3_private/retention_driver.esk").read_text()
        start = source.index("(define (e3-test-retention-mode-token frame)")
        end = source.index("(define (e3-test-retention-model-snapshot model)", start)
        seam = source[start:end]
        self.assertIn("(e3-frame-entry-live frame 'e3-test-retention-modes) 6", seam)
        self.assertIn("p1-e3-test-fixed-mixed!", seam)
        self.assertIn("p1-e3-test-mode-mask", seam)
        self.assertIn("(= mask 43690)", seam)
        for forbidden in ("m3t-entry", "module-mode-internal", "module-train!",
                          "module-eval!", "make-vector"):
            self.assertNotIn(forbidden, seam)
        self.assertNotIn("(e3-test-retention-modes=? model", source)
        self.assertNotIn("(e3-test-retention-set-mixed! model", source)

    def test_generated_stage_rejects_foreign_or_claimed_paths(self) -> None:
        helper = ROOT / "scripts/e3-mode-graph-stage-cleanup.sh"
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        self.assertIn('"${e3_tuple_kind}" == retention', builder)
        self.assertIn('e3_retention_generated_stage "${output_object}"', builder)
        with tempfile.TemporaryDirectory() as temp:
            gate = Path(temp)
            stage = gate / ".e3-retention.A1b2C3"
            stage.mkdir()
            object_path = stage / "e3_retention_package.o"
            def select(path: Path) -> subprocess.CompletedProcess:
                return subprocess.run(
                    ["bash", "-c", 'source "$1"; e3_retention_generated_stage "$2"',
                     "bash", str(helper), str(path)], capture_output=True, text=True)
            expected = gate / ".e3-retention-generated"
            selected = select(object_path)
            self.assertEqual((selected.returncode, selected.stdout),
                             (0, f"{expected}\n"), selected.stderr)
            self.assertFalse(expected.exists())  # selection cannot claim or erase it
            self.assertNotEqual(select(stage / "foreign.o").returncode, 0)
            fake_stage = gate / ".e3-retention.Z9y8X7"
            fake_stage.symlink_to(stage, target_is_directory=True)
            self.assertNotEqual(select(fake_stage / object_path.name).returncode, 0)
            expected.symlink_to(gate / "absent")
            self.assertNotEqual(select(object_path).returncode, 0)
            expected.unlink()
            expected.mkdir()
            marker = expected / "unrelated"
            marker.write_text("preserve\n")
            self.assertNotEqual(select(object_path).returncode, 0)
            self.assertEqual(marker.read_text(), "preserve\n")

    def test_retention_generated_module_normalization_is_scoped(self) -> None:
        script = r'''
PROJECT_ROOT=$1
raw_private_root=$1/tests/e3_private/retention_driver_root.esk
raw_package_bridge=$1/tests/e3_private/retention_bridge.c
raw_package_renames=$1/tests/e3_private/retention_package_private_renames.txt
raw_public_exports=$1/tests/e3_private/retention_package_public_exports.txt
raw_include_dirs=("$1/internal/p1/lib" "$1/internal/c1/lib" "$1/internal/t1/lib" "$1/src" "$1/internal/d2/lib" "$1/internal/e3/lib")
die() { printf '%s\n' "$*" >&2; exit 77; }
source "$1/scripts/e3-private-package-policy.sh"
e3_generated_source_root=/outside-private-test/source
e3_tuple_kind=$2
printf '%s\n' /outside-private-test/source/transformer/module.esk | e3_normalize_source_dependencies
'''
        for kind in ("retention", "mode-graph"):
            result = subprocess.run(["bash", "-c", script, "bash", str(ROOT), kind],
                                    capture_output=True, text=True)
            self.assertEqual((result.returncode, result.stdout),
                             (0, "@BUILD@/source/transformer/module.esk\n"), result.stderr)
        for kind in ("canonical", "numerical", "diagnostic"):
            result = subprocess.run(["bash", "-c", script, "bash", str(ROOT), kind],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 77, result.stderr)
            self.assertNotIn("@BUILD@/source/transformer/module.esk", result.stdout)

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
