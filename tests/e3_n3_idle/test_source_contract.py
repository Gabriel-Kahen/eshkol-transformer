"""Source-only N3 control checks; package admission and AOT are still held."""

from pathlib import Path
import os
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "tests/e3_n3_idle/driver.esk"


class IdleModelSourceTest(unittest.TestCase):
    def test_separate_entry_and_unchanged_base_abi(self) -> None:
        source = DRIVER.read_text()
        original = (ROOT / "tests/e3_private/driver.esk").read_text()
        root = (ROOT / "tests/e3_n3_idle/driver_root.esk").read_text()
        bridge = (ROOT / "tests/e3_n3_idle/bridge.c").read_text()
        self.assertIn("(define (e3-test-run scenario corpus-path)", original)
        self.assertIn("(= (vector-length scenario) 2)", original)
        self.assertNotIn("e3-n3-", original)
        self.assertIn("(define (e3-n3-zero-test-run scenario corpus-path)", source)
        self.assertIn("(= (vector-length scenario) 3)", source)
        self.assertEqual(root.count('(load "../e3_private/driver.esk")'), 1)
        self.assertEqual(root.count('(load "driver.esk")'), 1)
        self.assertIn("et_e3_n3_zero_test_run_cabi_v1", bridge)
        self.assertIn("et_e3_n3_zero_test_run_v1", bridge)

    def test_load_order_and_authentic_sources(self) -> None:
        source = DRIVER.read_text()
        constructor = source[source.index("(define (e3-n3-zero-test-run") :]
        self.assertLess(constructor.index("g3t-checked-m3t-model-create"),
                        constructor.index("(vector-set! e3-n3-model-roots"))
        self.assertLess(constructor.index("(e3-n3-zero-load! model handles)"),
                        constructor.index("(e3-frame-bind! model destinations)"))
        self.assertIn("(parameter-tree-paths tree)", source)
        self.assertIn("(parameter-tree-tie-groups tree)", source)
        self.assertIn("(parameter-handle-tensor-internal handle)", source)
        self.assertIn("(i2-checkpoint-decode", source)
        self.assertIn("(vector-ref decoded 7)", source)
        self.assertIn("(i2-prepare-load (vector rows e3-n3-operation))", source)
        self.assertRegex(source, re.compile(
            r"\(i2-prepare-load \(vector rows e3-n3-operation\)\)\)+\s*"
            r"\(i2-commit-load! \(vector-ref plan 0\)\)"))
        self.assertIn("(i2-release-owned!", source)
        self.assertIn("(e3-n3-seed-gradient", source)
        self.assertIn("(e3-n3-gradient-image before)", source)
        self.assertIn("(e3-n3-zero-image? loaded)", source)
        self.assertIn("(if frame\n                (begin (e3-frame-unbind! frame)", source)
        self.assertLess(source.rindex("(e3-frame-unbind! frame)"),
                        source.rindex("(e3-test-destinations-destroy! destinations)"))

    def test_zero_load_release_is_not_external_tail_call(self) -> None:
        source = DRIVER.read_text()
        release = source.split("(define (e3-n3-release-sources! sources)", 1)[1].split(
            "(define (e3-n3-zero-load!", 1)[0]
        load = source.split("(define (e3-n3-zero-load! model handles)", 1)[1].split(
            "(define (e3-n3-scenario", 1)[0]
        self.assertIn("(if (= index 14) #t", release)
        self.assertRegex(load, re.compile(
            r"\(i2-commit-load! \(vector-ref plan 0\)\)\)\)\s*"
            r"\(e3-n3-release-sources! sources\)\s*#t\)\)\s*$"))

    def test_exact_tuple_policy_and_pending_import_gate(self) -> None:
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        self.assertIn('"${e3_tuple_kind}" == n3-idle', builder)
        self.assertIn("exact compiled undefined-symbol inventory is not reviewed",
                      builder)
        self.assertIn("-DET_E3_RETENTION_TESTING", builder)
        self.assertIn("-DET_F32_TENSOR_TESTING", builder)
        self.assertIn("-DET_D2_NATIVE_TESTING", builder)
        self.assertIn("et_e3_n3_zero_test_run_v1", builder)
        for manifest in ("e3_private_package_public_exports.txt",
                         "e3_diagnostic_public_package_public_exports.txt"):
            self.assertNotIn("et_e3_n3_zero_test_run_v1",
                             (ROOT / "native" / manifest).read_text().splitlines())

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
        prefix = ROOT / "tests/e3_n3_idle/package"
        inputs = [str(ROOT), str(ROOT / "tests/e3_n3_idle/driver_root.esk"),
                  str(ROOT / "tests/e3_n3_idle/bridge.c"),
                  f"{prefix}_private_renames.txt",
                  f"{prefix}_public_exports.txt", str(ROOT / "scripts/e3-private-package-policy.sh")]
        def check(values: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.run(["bash", "-c", script, "bash", *values],
                                  capture_output=True, text=True)
        exact = check(inputs)
        self.assertEqual((exact.returncode, exact.stdout), (0, "n3-idle\n"),
                         exact.stderr)
        for index, replacement in ((2, str(ROOT / "native/e3_private_bridge.c")),
                                   (3, str(ROOT / "native/e3_private_package_private_renames.txt")),
                                   (4, str(ROOT / "tests/e3_mode_graph/package_public_exports.txt"))):
            mixed = inputs.copy()
            mixed[index] = replacement
            result = check(mixed)
            self.assertEqual(result.returncode, 77)
            self.assertIn("mixed", result.stderr)

        with tempfile.TemporaryDirectory(prefix="e3-n3-held-") as directory:
            artifact = Path(directory) / "n3.o"
            env = os.environ.copy()
            env["ESHKOL_BUILD_DIR"] = str(Path(directory) / "unavailable-compiler")
            build = subprocess.run([
                str(ROOT / "scripts/build-e1b-consumer.sh"), *inputs[1:5],
                str(artifact), *(str(ROOT / part) for part in (
                    "internal/p1/lib", "internal/c1/lib", "internal/t1/lib",
                    "src", "internal/d2/lib", "internal/e3/lib"))],
                env=env, capture_output=True, text=True,
            )
            self.assertNotEqual(build.returncode, 0)
            self.assertIn("E3 N3 idle tuple HOLD: exact compiled undefined-symbol "
                          "inventory is not reviewed", build.stderr)
            self.assertNotIn("pinned Eshkol compiler not built", build.stderr)
            self.assertFalse(artifact.exists())
            self.assertFalse(Path(f"{artifact}.evidence").exists())

    def test_source_derived_manifest_proposals(self) -> None:
        prefix = ROOT / "tests/e3_n3_idle/package"
        base = ROOT / "native/e3_private_package"
        def lines(path: Path, suffix: str) -> list[str]:
            return Path(f"{path}_{suffix}.txt").read_text().splitlines()
        for suffix in ("defined_symbols", "public_exports", "public_strings"):
            self.assertEqual(lines(prefix, suffix), sorted(lines(base, suffix) +
                             ["et_e3_n3_zero_test_run_v1"]))
        self.assertEqual(lines(prefix, "native_objects"),
                         lines(base, "native_objects"))
        self.assertEqual(lines(prefix, "archive_members"),
                         ["e3_n3_idle_package.o"])
        source = lines(prefix, "source_closure")
        self.assertEqual(source[:2], ["tests/e3_n3_idle/driver_root.esk",
                                      "native/e3_private_root.esk"])
        self.assertEqual(source[-2:], ["tests/e3_private/driver.esk",
                                       "tests/e3_n3_idle/driver.esk"])
        self.assertIn("tests/e3_n3_idle/bridge.c",
                      lines(prefix, "native_source_closure"))
        self.assertFalse(Path(f"{prefix}_undefined_symbols.txt").exists())


if __name__ == "__main__":
    unittest.main()
