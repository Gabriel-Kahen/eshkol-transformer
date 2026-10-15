"""Fail-closed source-level checks; emitted Eshkol closure needs pinned build."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "native/e3_private_package"
FENV = ROOT / "tests/e3_private/fenv_package"


def lines(prefix, suffix):
    return Path(f"{prefix}_{suffix}.txt").read_text().splitlines()


class FenvTupleTest(unittest.TestCase):
    def test_manifest_candidates_are_derived_from_exact_base(self):
        for suffix in ("defined_symbols", "public_exports", "public_strings"):
            self.assertEqual(lines(FENV, suffix),
                             sorted(lines(BASE, suffix) + ["et_e3_fenv_run_v1"]))
        for suffix in ("undefined_symbols", "native_objects"):
            self.assertEqual(lines(FENV, suffix), lines(BASE, suffix))
        self.assertEqual(lines(FENV, "private_renames"),
                         lines(BASE, "private_renames") +
                         ["e3-fenv-run et_e3_private_fenv_run_cabi_v1"])
        self.assertEqual(lines(FENV, "source_closure"),
                         ["tests/e3_private/fenv_driver_root.esk"] +
                         lines(BASE, "source_closure") +
                         ["tests/e3_private/fenv_driver.esk"])
        self.assertEqual(lines(FENV, "native_source_closure"),
                         ["tests/e3_private/fenv_bridge.c"] +
                         lines(BASE, "native_source_closure") +
                         ["tests/e3_private/fenv_observer.c"])
        self.assertEqual(lines(FENV, "archive_members"), ["e3_fenv_package.o"])
        bridge = (ROOT / "tests/e3_private/fenv_bridge.c").read_text()
        self.assertLess(bridge.index('"../../native/e3_private_bridge.c"'),
                        bridge.index('"fenv_observer.c"'))

    def test_policy_requires_all_four_exact_inputs(self):
        inputs = [ROOT / "tests/e3_private/fenv_driver_root.esk",
                  ROOT / "tests/e3_private/fenv_bridge.c",
                  Path(f"{FENV}_private_renames.txt"),
                  Path(f"{FENV}_public_exports.txt")]
        policy = ROOT / "scripts/e3-private-package-policy.sh"
        script = r'''
PROJECT_ROOT=$1
raw_private_root=$2
raw_package_bridge=$3
raw_package_renames=$4
raw_public_exports=$5
raw_include_dirs=("$1/internal/p1/lib" "$1/internal/c1/lib" "$1/internal/t1/lib" "$1/src" "$1/internal/d2/lib" "$1/internal/e3/lib")
if [[ "${7:-}" == ambient ]]; then raw_include_dirs+=("$1/native"); fi
die() { printf '%s\n' "$*" >&2; exit 77; }
source "$6"
printf '%s\n' "$e3_tuple_kind"
'''
        def select(paths, extra=""):
            return subprocess.run(
                ["bash", "-c", script, "bash", str(ROOT),
                 *(str(path) for path in paths), str(policy), extra],
                capture_output=True, text=True)
        valid = select(inputs)
        self.assertEqual((valid.returncode, valid.stdout), (0, "fenv\n"),
                         valid.stderr)
        for index, replacement in enumerate((
            ROOT / "native/e3_private_driver_root.esk",
            ROOT / "native/e3_private_bridge.c",
            ROOT / "native/e3_private_package_private_renames.txt",
            ROOT / "native/e3_private_package_public_exports.txt",
        )):
            mixed = inputs.copy()
            mixed[index] = replacement
            self.assertEqual(select(mixed).returncode, 77)
        self.assertEqual(select(inputs, "ambient").returncode, 77)
        with tempfile.TemporaryDirectory() as temp:
            alias = Path(temp) / "fenv_driver_root.esk"
            alias.symlink_to(inputs[0])
            self.assertEqual(select([alias, *inputs[1:]]).returncode, 77)

    def test_stage_and_build_mode_are_closed(self):
        helper = ROOT / "scripts/e3-generated-source-stage.sh"
        with tempfile.TemporaryDirectory() as temp:
            gate = Path(temp)
            stage = gate / ".e3-fenv.ABC123"
            stage.mkdir()
            def select(path):
                return subprocess.run(
                    ["bash", "-c", 'source "$1"; e3_fenv_generated_stage "$2"',
                     "bash", str(helper), str(path)],
                    capture_output=True, text=True)
            self.assertEqual(select(stage / "e3_fenv_package.o").stdout,
                             str(gate / ".e3-fenv-generated") + "\n")
            self.assertNotEqual(select(stage / "other.o").returncode, 0)
            alias = gate / ".e3-fenv.ALIAS1"
            alias.symlink_to(stage, target_is_directory=True)
            self.assertNotEqual(select(alias / "e3_fenv_package.o").returncode, 0)
        builder = ROOT / "scripts/build-e3-fenv-test.sh"
        invalid = subprocess.run(["bash", str(builder), "/tmp/unused", "sanitize"],
                                 capture_output=True, text=True)
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("usage:", invalid.stderr)

    def test_e1b_has_fenv_only_fault_source_and_private_bindings(self):
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        self.assertIn('"${e3_tuple_kind}" == fenv', builder)
        self.assertIn('package_source_flags+=("-DET_E3_FAULT_TESTING")', builder)
        self.assertIn("et_e3_private_fenv_run_cabi_v1", builder)
        for symbol in ("et_e3_test_fenv_begin_v1", "et_e3_test_fenv_end_v1",
                       "et_e3_test_fenv_ref_v1", "et_e3_test_fenv_env_byte_v1",
                       "et_e3_test_fault_arm_v1", "et_e3_test_fault_seed_publication_v1"):
            self.assertIn(symbol, builder)
        self.assertIn("E3 fenv sanitizer and private mutant selectors need pinned emitted inventories", builder)

    def test_unverified_private_selectors_fail_before_toolchain(self):
        prefix = ROOT / "tests/e3_private/fenv_package"
        command = ["bash", str(ROOT / "scripts/build-e1b-consumer.sh"),
                   str(ROOT / "tests/e3_private/fenv_driver_root.esk"),
                   str(ROOT / "tests/e3_private/fenv_bridge.c"),
                   str(prefix) + "_private_renames.txt",
                   str(prefix) + "_public_exports.txt",
                   "/tmp/e3-fenv-selector-denial.o",
                   *(str(ROOT / path) for path in
                     ("internal/p1/lib", "internal/c1/lib", "internal/t1/lib",
                      "src", "internal/d2/lib", "internal/e3/lib"))]
        for selector in ("E3_FENV_SANITIZE", "E3_FENV_MUTANT"):
            env = dict(os.environ, **{selector: "1"})
            denied = subprocess.run(command, env=env, capture_output=True,
                                    text=True)
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("need pinned emitted inventories", denied.stderr)


if __name__ == "__main__":
    unittest.main()
