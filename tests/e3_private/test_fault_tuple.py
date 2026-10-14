"""Source-only admission controls for the noninstalled E3 fault tuple."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "tests/e3_private"
POLICY = ROOT / "scripts/e3-private-package-policy.sh"
BASE = PRIVATE / "retention_package"
FAULT = PRIVATE / "fault_package"
INPUTS = (
    PRIVATE / "fault_driver_root.esk",
    PRIVATE / "fault_bridge.c",
    PRIVATE / "fault_package_private_renames.txt",
    PRIVATE / "fault_package_public_exports.txt",
)
INCLUDES = tuple(ROOT / part for part in (
    "internal/p1/lib", "internal/c1/lib", "internal/t1/lib", "src",
    "internal/d2/lib", "internal/e3/lib",
))


def inventory(prefix: Path, name: str) -> list[str]:
    return Path(f"{prefix}_{name}.txt").read_text().splitlines()


def select(paths=INPUTS, includes=INCLUDES, extra="") -> subprocess.CompletedProcess:
    script = r'''
set -euo pipefail
PROJECT_ROOT=$1
shift
raw_private_root=$1; raw_package_bridge=$2
raw_package_renames=$3; raw_public_exports=$4
shift 4
raw_include_dirs=("$@")
die() { printf '%s\n' "$*" >&2; exit 77; }
source "$PROJECT_ROOT/scripts/e3-private-package-policy.sh"
''' + extra + '\nprintf "%s\\n" "$e3_tuple_kind"\n'
    return subprocess.run(
        ["bash", "-c", script, "tuple", str(ROOT), *(str(p) for p in paths),
         *(str(p) for p in includes)], capture_output=True, text=True,
        check=False,
    )


class FaultTupleTests(unittest.TestCase):
    def test_exact_lexical_tuple_and_near_misses(self) -> None:
        exact = select()
        self.assertEqual((exact.returncode, exact.stdout), (0, "fault\n"),
                         exact.stderr)
        for position, replacement in (
            (0, PRIVATE / "retention_driver_root.esk"),
            (1, PRIVATE / "retention_bridge.c"),
            (2, PRIVATE / "retention_package_private_renames.txt"),
            (3, PRIVATE / "retention_package_public_exports.txt"),
            (1, ROOT / "native/e3_private_bridge.c"),
        ):
            with self.subTest(position=position, replacement=replacement):
                altered = list(INPUTS)
                altered[position] = replacement
                result = select(altered)
                self.assertEqual(result.returncode, 77)
                self.assertIn("mixed", result.stderr)
        self.assertEqual(select(includes=INCLUDES[:-1]).returncode, 77)
        self.assertEqual(select(includes=INCLUDES[::-1]).returncode, 77)
        alias = select(paths=(str(PRIVATE) + "/./fault_driver_root.esk", *INPUTS[1:]))
        self.assertEqual(alias.returncode, 77)
        self.assertIn("lexical", alias.stderr)

    def test_symlinked_input_and_ambient_override_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=PRIVATE, prefix=".e3-fault-path-") as name:
            link = Path(name) / "alias"
            link.symlink_to(INPUTS[0])
            result = select(paths=(link, *INPUTS[1:]))
            self.assertEqual(result.returncode, 77)
            script = ('PROJECT_ROOT=$1; die() { exit 77; }; '
                      'source "$PROJECT_ROOT/scripts/e3-private-package-policy.sh"; '
                      'e3_check_repository_path "$2"')
            direct = subprocess.run(
                ["bash", "-c", script, "path", str(ROOT), str(link)],
                capture_output=True, text=True,
            )
            self.assertEqual(direct.returncode, 77)
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        self.assertIn('[[ -z "${E1B_PACKAGE_POLICY+x}" ]]', builder)
        self.assertIn('E1B_PACKAGE_POLICY overrides are forbidden', builder)
        with tempfile.TemporaryDirectory() as name:
            denied = subprocess.run(
                ["bash", str(ROOT / "scripts/build-e1b-consumer.sh"),
                 *(str(p) for p in INPUTS), str(Path(name) / "denied.o"),
                 *(str(p) for p in INCLUDES)],
                env={**os.environ, "E1B_PACKAGE_POLICY": "foreign"},
                capture_output=True, text=True,
            )
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("E1B_PACKAGE_POLICY overrides are forbidden", denied.stderr)

    def test_exact_manifests_and_localization(self) -> None:
        for name in ("defined_symbols", "public_exports", "public_strings"):
            self.assertEqual(inventory(FAULT, name),
                             sorted(inventory(BASE, name) + ["et_e3_fault_run_v1"]))
        for name in ("undefined_symbols", "native_objects",
                     "sanitized_noninstrumentation_undefined_symbols",
                     "sanitized_instrumentation_undefined_symbols"):
            self.assertEqual(inventory(FAULT, name), inventory(BASE, name))
        self.assertEqual(inventory(FAULT, "archive_members"), ["e3_fault_package.o"])
        self.assertEqual(inventory(FAULT, "private_renames"),
                         inventory(BASE, "private_renames") +
                         ["e3-test-fault-dispatch et_e3_private_fault_dispatch_cabi_v1"])
        self.assertEqual(inventory(FAULT, "source_closure"),
                         ["tests/e3_private/fault_driver_root.esk"] +
                         inventory(BASE, "source_closure") +
                         ["tests/e3_private/fault_driver.esk"])
        native = inventory(BASE, "native_source_closure")
        native.insert(native.index("tests/e3_private/retention_bridge.c"),
                      "tests/e3_private/fault_bridge.c")
        self.assertEqual(inventory(FAULT, "native_source_closure"), native)
        self.assertNotIn("et_e3_fault_run_v1",
                         inventory(ROOT / "native/e3_private_package", "public_exports"))
        self.assertNotIn("et_e3_fault_run_v1",
                         inventory(ROOT / "native/e3_diagnostic_public_package", "public_exports"))

    def test_fault_sanitizer_inventory_rejects_mutation(self) -> None:
        policy = ROOT / "scripts/e3-retention-sanitizer-import-policy.sh"
        with tempfile.TemporaryDirectory() as name:
            paths = []
            for suffix in (
                "undefined_symbols",
                "sanitized_noninstrumentation_undefined_symbols",
                "sanitized_instrumentation_undefined_symbols",
            ):
                path = Path(name) / suffix
                path.write_text("\n".join(inventory(FAULT, suffix)) + "\n")
                paths.append(path)
            script = ('set -euo pipefail; PROJECT_ROOT=$1; e3_tuple_kind=fault; '
                      'die() { printf "%s\\n" "$*" >&2; exit 77; }; '
                      'source "$2"; e3_retention_verify_sanitizer_imports "$3" "$4" "$5"')
            def verify() -> subprocess.CompletedProcess:
                return subprocess.run(
                    ["bash", "-c", script, "sanitizer", str(ROOT), str(policy),
                     *(str(p) for p in paths)], capture_output=True, text=True,
                )
            self.assertEqual(verify().returncode, 0)
            for index in (0, 1, 2):
                original = paths[index].read_bytes()
                lines = original.decode().splitlines()
                lines[-1] = "zz_unreviewed_import"
                paths[index].write_text("\n".join(sorted(lines)) + "\n")
                self.assertEqual(verify().returncode, 77)
                paths[index].write_bytes(original)

    def test_build_selection_and_feature_off_scope(self) -> None:
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        native = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        wrapper = (ROOT / "scripts/build-e3-fault-test.sh").read_text()
        self.assertIn('"${e3_tuple_kind}" == fault', builder)
        self.assertIn('package_native_cflags+=(-DET_E3_FAULT_TESTING)', builder)
        self.assertIn('E3_FAULT_SANITIZE', builder)
        self.assertIn('e3_fault_generated_stage "${output_object}"', builder)
        self.assertIn('et_e3_private_fault_dispatch_cabi_v1', builder)
        self.assertIn('#ifdef ET_E3_FAULT_TESTING', native)
        self.assertIn('E3_FAULT_SANITIZE="${sanitize}"', wrapper)
        self.assertIn('"${staging}/e3_fault_package.o"', wrapper)
        self.assertNotIn('fault_driver_root.esk',
                         (ROOT / "native/e3_private_driver_root.esk").read_text())
        self.assertNotIn('fault_driver_root.esk',
                         (ROOT / "native/e3_diagnostic_public_root.esk").read_text())
        with tempfile.TemporaryDirectory() as name:
            retention = (PRIVATE / "retention_driver_root.esk",
                         PRIVATE / "retention_bridge.c",
                         PRIVATE / "retention_package_private_renames.txt",
                         PRIVATE / "retention_package_public_exports.txt")
            denied = subprocess.run(
                ["bash", str(ROOT / "scripts/build-e1b-consumer.sh"),
                 *(str(p) for p in retention), str(Path(name) / "denied.o"),
                 *(str(p) for p in INCLUDES)],
                env={**os.environ, "E3_FAULT_SANITIZE": "1"},
                capture_output=True, text=True,
            )
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("fault sanitizer is allowed only for the exact test tuple",
                          denied.stderr)

    def test_stage_is_fresh_and_private(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-fault-stage-") as name:
            gate = Path(name)
            stage = gate / ".e3-fault.Abc123"
            stage.mkdir()
            helper = ROOT / "scripts/e3-mode-graph-stage-cleanup.sh"
            script = 'source "$1"; e3_fault_generated_stage "$2"'
            def probe(path: Path) -> subprocess.CompletedProcess:
                return subprocess.run(["bash", "-c", script, "stage", str(helper), str(path)],
                                      capture_output=True, text=True)
            target = stage / "e3_fault_package.o"
            self.assertEqual(probe(target).stdout,
                             str(gate / ".e3-fault-generated") + "\n")
            self.assertNotEqual(probe(stage / "foreign.o").returncode, 0)
            alias = gate / ".e3-fault.Xyz987"
            alias.symlink_to(stage, target_is_directory=True)
            self.assertNotEqual(probe(alias / target.name).returncode, 0)
            (gate / ".e3-fault-generated").symlink_to(gate / "absent")
            self.assertNotEqual(probe(target).returncode, 0)


if __name__ == "__main__":
    unittest.main()
