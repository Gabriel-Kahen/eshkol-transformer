"""Noninstalled uniform-tie tuple and private fault-hook boundaries."""

from pathlib import Path
import math
import re
import shutil
import subprocess
import tempfile
import unittest

from tests.e3_reference.uniform_tie import SHAPES
from tests.e3_private.parse_uniform_observation import MARKER, parse_uniform
from tests.e3_private.test_numerical_tuple import frame_bytes, frame_lines


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests/e3_private/numerical_package"
UNIFORM = ROOT / "tests/e3_private/uniform_package"


def lines(prefix: Path, suffix: str) -> list[str]:
    return Path(f"{prefix}_{suffix}.txt").read_text().splitlines()


def scheme_form(source: str, name: str) -> str:
    """Extract one balanced top-level definition; this is a source check only."""
    start = source.index(f"(define ({name}")
    depth = 0
    quoted = escaped = commented = False
    for end in range(start, len(source)):
        char = source[end]
        if commented:
            if char == "\n":
                commented = False
        elif escaped:
            escaped = False
        elif char == "\\" and quoted:
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif char == ";" and not quoted:
            commented = True
        elif not quoted:
            depth += (char == "(") - (char == ")")
            if depth == 0:
                return source[start:end + 1]
            if depth < 0:
                break
    raise ValueError(f"unbalanced {name}")


class UniformTupleTest(unittest.TestCase):
    def test_uniform_only_gradient_accessor_object_closure(self) -> None:
        cc, nm = shutil.which("cc"), shutil.which("nm")
        if not cc or not nm:
            self.skipTest("host C compiler and nm required for object-symbol check")
        includes = ["-I", str(ROOT / "include"), "-I", str(ROOT / "src"),
                    "-I", str(ROOT / "native")]
        source = ROOT / "src/eshkol_transformer/m3t_f32_integration.c"
        with tempfile.TemporaryDirectory(prefix="e3-uniform-symbol-") as temporary:
            plain, enabled, frame = (Path(temporary) / name
                                     for name in ("plain.o", "enabled.o", "frame.o"))
            for flags, output in (([], plain), (["-DET_F32_TENSOR_TESTING"], enabled)):
                subprocess.run([cc, "-std=c11", *flags, *includes, "-c",
                                str(source), "-o", str(output)], check=True,
                               capture_output=True, text=True)
            frame_flags = ("-DET_E3_RETENTION_TESTING", "-DET_F32_TENSOR_TESTING",
                           "-DET_D2_NATIVE_TESTING", "-DET_E3_NUMERICAL_TESTING")
            subprocess.run([cc, "-std=c11", *frame_flags, *includes, "-c",
                            str(ROOT / "src/eshkol_transformer/e3_frame.c"),
                            "-o", str(frame)], check=True, capture_output=True,
                           text=True)
            symbol = "et_m3t_test_parameter_gradient"
            def symbols(path: Path, option: str) -> set[str]:
                output = subprocess.run([nm, option, str(path)], check=True,
                                        capture_output=True, text=True).stdout
                return {line.split()[-1] for line in output.splitlines() if line.split()}
            self.assertNotIn(symbol, symbols(plain, "--defined-only"))
            self.assertIn(symbol, symbols(enabled, "--defined-only"))
            self.assertIn(symbol, symbols(frame, "--undefined-only"))

    def test_zero_state_roster_and_transaction_guards(self) -> None:
        driver = (ROOT / "tests/e3_private/uniform_driver.esk").read_text()
        bridge = (ROOT / "tests/e3_private/uniform_bridge.c").read_text()

        def roster(source: str) -> tuple:
            paths = source.split("(define e3-uniform-paths", 1)[1].split(
                "(define e3-uniform-shapes", 1)[0]
            shapes = source.split("(define e3-uniform-shapes", 1)[1].split(
                "(define e3-uniform-unique-counts", 1)[0]
            counts = source.split("(define e3-uniform-unique-counts", 1)[1].split(
                "(define e3-uniform-files", 1)[0]
            return (
                tuple("/".join(re.findall(r'"([^"]+)"', line))
                      for line in paths.splitlines() if '"' in line),
                tuple(tuple(map(int, match.split()))
                      for match in re.findall(r"\(([0-9 ]+)\)", shapes)),
                tuple(map(int, re.findall(r"\d+", counts))),
            )

        paths, shapes, counts = roster(driver)
        self.assertEqual(tuple(zip(paths, shapes)), SHAPES)
        self.assertEqual(counts, tuple(math.prod(shape) for _, shape in SHAPES[:-1]))
        self.assertEqual((len(counts), sum(counts), 4 * sum(counts)), (14, 1184, 4736))
        for mutant in (driver.replace('("blocks" "0" "attention" "key" "weight")',
                                      '("blocks" "0" "attention" "keq" "weight")', 1),
                       driver.replace('(4 8)', '(8 8)', 1)):
            self.assertNotEqual(roster(mutant), roster(driver))
        for witness in ("'parameter-roster", "'tie-graph", "'unique-owner-handles",
                        "'owner-contract", "'owner-payload-length",
                        "'owner-payload-total", "'owner-zero-payload",
                        "'original-state-hash", "'loaded-state-hash",
                        "'no-active-native-leases"):
            self.assertIn(witness, driver)
        self.assertEqual(driver.count("(e3-uniform-assert-original model words handles before before-hash"), 6)
        self.assertLess(driver.index("(e3-uniform-load-before-frame! model)"),
                        driver.index("(e3-frame-bind! model destinations)"))
        self.assertIn("et_e3_uniform_live_leases_v1", bridge)
        build = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        self.assertIn('"${e3_tuple_kind}" == uniform &&\n'
                      '          ( "${package_native_source}" == "${PROJECT_ROOT}/native/f32_tensor.c" ||\n'
                      '            "${package_native_source}" == "${PROJECT_ROOT}/src/eshkol_transformer/m3t_f32_integration.c" ) ]]; then\n'
                      '      package_source_flags+=(-DET_F32_TENSOR_TESTING)',
                      build)
        for field in ("borrows", "copy_plans", "gradient_plans", "reset_plans"):
            self.assertIn(f"counts.{field}", bridge)

    def test_prebind_and_postcall_contract_source(self) -> None:
        driver = (ROOT / "tests/e3_private/uniform_driver.esk").read_text()
        build = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        owner = scheme_form(driver, "e3-uniform-check-owners")
        original = scheme_form(driver, "e3-uniform-assert-original")
        load = scheme_form(driver, "e3-uniform-load-before-frame!")
        post = scheme_form(driver, "e3-uniform-assert-post")
        call = scheme_form(driver, "e3-test-run")
        gradients = scheme_form(driver, "e3-uniform-gradient-image")
        modes = scheme_form(driver, "e3-uniform-modes")
        self.assertIn("(i2-describe", owner)
        self.assertIn("'native-owner-geometry", owner)
        for witness in ("e3-uniform-gradient-image", "e3-uniform-modes",
                        "e3-uniform-assert-leases"):
            self.assertIn(witness, original)
            self.assertIn(witness, post)
        self.assertEqual(load.count("(e3-uniform-assert-original"), 6)
        self.assertIn("'zero-gradients", load)
        self.assertIn("'zero-modes", load)
        self.assertIn("'gradient-word-readable", gradients)
        self.assertIn("(<= bits 4294967295)", gradients)
        self.assertIn("(vector-ref (m3t-entry model 'model 'e3-uniform-roster #t) 0)", modes)
        self.assertIn("(module-mode-internal current)", modes)
        self.assertIn("'mode-child-parent", modes)
        self.assertIn("'distinct-mode-modules", modes)
        self.assertIn("'mode-root-node-count", modes)
        self.assertNotIn("(vector-ref (m3t-entry model 'model 'e3-uniform-roster #t) 4)", modes)
        self.assertLess(call.index("(e3-evaluate-into! model dataset frame)"),
                        call.index("(e3-uniform-assert-post model"))
        self.assertIn('"${e3_tuple_kind}" == uniform ]]; then\n'
                      '    package_native_cflags+=(-DET_E3_NUMERICAL_TESTING\n'
                      '                            -DET_E3_RETENTION_TESTING', build)
        for symbol in ("et_e3_test_model_word_count_v1",
                       "et_e3_test_model_word_v1",
                       "et_e3_test_gradient_metadata_v1"):
            self.assertIn(symbol, build)
            self.assertIn(symbol, (ROOT / "src/eshkol_transformer/e3_frame.c").read_text())
        self.assertIn("et_m3t_test_parameter_gradient",
                      (ROOT / "src/eshkol_transformer/m3t_f32_integration.c").read_text())
        self.assertIn('"${e3_tuple_kind}" == uniform ) ]]; then', build)
        # Causal source mutations: an equal-byte wrong geometry or dropped
        # postcall guard must fail the extracted contract before any runtime claim.
        def source_contract(owner_form: str, call_form: str) -> bool:
            return "(i2-describe" in owner_form and "(e3-uniform-assert-post model" in call_form

        self.assertTrue(source_contract(owner, call))
        self.assertFalse(source_contract(owner.replace("(i2-describe", "(i2-describx", 1), call))
        self.assertFalse(source_contract(owner, call.replace(
            "(e3-uniform-assert-post model", "(e3-uniform-assert-posx model", 1)))

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
            "public_strings", "private_renames", "undefined_symbols",
            "sanitized_added_undefined_symbols", "sanitized_absent_undefined_symbols",
        ):
            self.assertEqual(lines(UNIFORM, suffix), lines(BASE, suffix), suffix)
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
