"""Closed private C2 SHA stage topology checks; no runtime claim."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from tests.c2.test_handler_order import parse

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "native/g3r_c2_sha_stage_root.esk"
MANIFEST = ROOT / "native/g3r_c2_sha_stage_source_closure.txt"
BASE = ROOT / "native/c2_checkpoint_load_source_closure.txt"
RUNNER = ROOT / "scripts/test-g3r-c2-sha-stage.sh"
NATIVE_RUNNER = ROOT / "scripts/test-g3r-c2-stage-sha.sh"
EXTENSION = ROOT / "native/g3r_c2_sha_stage_extension.esk"

LOADS = ["d2_wave2_root.esk", "c2_d2_cursor_pair_extension.esk",
         "o2_wave2_extension.esk", "c2_o2_reconstruct_extension.esk",
         "c2_x1_canonical_extension.esk", "c2_training_state_extension.esk",
         "c2_persistence_policy_extension.esk", "c2_checkpoint_load_extension.esk",
         "g3r_c2_sha_stage_extension.esk"]


def verify_root_and_manifest(root: str, manifest: str) -> None:
    forms = parse(root)
    actual_loads = [form[1].strip('"') for form in forms
                    if isinstance(form, list) and len(form) == 2
                    and form[0] == "load"]
    if len(forms) != len(LOADS) or actual_loads != LOADS:
        raise ValueError("private owning load order differs")
    expected = ["native/g3r_c2_sha_stage_root.esk",
                *BASE.read_text().splitlines(),
                "native/g3r_c2_sha_stage_extension.esk"]
    if manifest.splitlines() != expected or len(set(expected)) != len(expected):
        raise ValueError("private source closure differs")


def verify_runner(runner: str) -> None:
    if (runner.count("-DET_G3R_C2_STAGE_SHA_PRIVATE") != 1
            or '"$directory/load-off.o"' not in runner
            or '"$directory/objects/load.o"' not in runner
            or '"$directory/objects"/*.o' not in runner
            or 'build_mode normal' not in runner
            or 'build_mode repeat' not in runner
            or 'build_mode sanitize' not in runner
            or 'ASAN_OPTIONS=detect_leaks=1:halt_on_error=1' not in runner
            or 'UBSAN_OPTIONS=halt_on_error=1' not in runner
            or 'LSAN_OPTIONS=exitcode=23:report_objects=1' not in runner
            or '--dump-ir' not in runner
            or runner.count('--emit-object') != 1
            or runner.count('--emit-depfile') != 1
            or runner.count('"${eshkol_inputs[@]}"') != 2
            or not (runner.index('--dump-ir') < runner.index('--emit-object')
                    < runner.index('--emit-depfile')
                    < runner.index('test -s "$directory/stage.d"'))
            or '"$directory/stage-test.ll"' not in runner
            or '"$directory/stage-test.o"' not in runner
            or '"$directory/stage.d"' not in runner
            or runner.count('"$directory/stage-dep.o"') != 4
            or runner.count('"$directory/object-build.stderr"') != 2
            or 'test -s "$directory/stage-dep.o"' not in runner
            or 'test ! -s "$directory/object-build.stderr"' not in runner
            or '"$directory/dep-object-undefined-symbols.txt"' not in runner
            or '"$directory/dep-object-defined-symbols.txt"' not in runner
            or "build-e1b-consumer.sh" in runner):
        raise ValueError("private native feature build differs")


def verify_compiler_handoff(runner: str, native: str) -> None:
    call = '"$PROJECT_ROOT/scripts/test-g3r-c2-stage-sha.sh" "$cc"'
    if (call not in runner
            or runner.index("resolve_provenance_compilers") >= runner.index(call)
            or 'clang_cc="$1"' not in native
            or native.count('"$clang_cc"') < 7
            or 'gcc "${cflags[@]}" "${sources[@]}"' not in native
            or 'for compiler in clang gcc' in native
            or '\nclang ' in native):
        raise ValueError("native compiler handoff differs")


def verify_stage_allocator_binding(extension: str) -> None:
    forms = parse(extension)
    definitions = [form for form in forms if isinstance(form, list)
                   and len(form) == 3 and form[0] == "define"
                   and form[1] == "g3r-c2-sha-stage-allocations"]
    if (len(definitions) != 1 or not isinstance(definitions[0][2], list)
            or definitions[0][2][0] != "lambda"
            or definitions[0][2][1] != ["measure"]):
        raise ValueError("stage allocator needs a mutable binding")


def verify_publication_recovery(extension: str, runner: str) -> None:
    checkpoints = (
        "(c2-runtime-reserve-exception-handlers 2)",
        "(guard (cleanup-defect",
        "(guard (caught",
        "(set! shell\n                            (c2-load-build-stage-entry",
        "(g3r-c2-sha-after-factory-hook shell digest)",
        "(vector-set! sidecar 0 shell)",
        "(vector-set! sidecar 1 'live)",
        "(set! g3r-c2-sha-stages next)",
    )
    positions = [extension.find(checkpoints[0])]
    for token in checkpoints[1:]:
        positions.append(extension.find(token, positions[-1] + 1))
    if (-1 in positions
            or "(c2-checkpoint-load-staging-release-internal!\n"
               "                                         shell)" not in extension
            or "(g3r-c2-sha-zero! digest)" not in extension
            or "(g3r-c2-sha-fail-stop 134)" not in extension):
        raise ValueError("stage publication recovery differs")
    if (">\"$directory/ir-object-build.stdout\"" not in runner
            or "2>\"$directory/ir-object-build.stderr\"" not in runner
            or "expected = warning * 4 + \"4 warnings generated.\\n\""
               not in runner
            or "unexpected LLVM IR object compiler diagnostics" not in runner):
        raise ValueError("IR diagnostics policy differs")


class C2ShaStageStaticTest(unittest.TestCase):
    def test_exact_private_ownership_and_symbols(self) -> None:
        verify_root_and_manifest(PRIVATE.read_text(), MANIFEST.read_text())
        verify_runner(RUNNER.read_text())
        verify_compiler_handoff(RUNNER.read_text(), NATIVE_RUNNER.read_text())
        extension = EXTENSION.read_text()
        parse(extension)
        verify_stage_allocator_binding(extension)
        verify_publication_recovery(extension, RUNNER.read_text())
        fixture = (ROOT / "tests/g3r/c2_sha_stage_test.esk").read_text()
        parse(fixture)
        self.assertIn("(borrow g3r-c2-sha-stages", fixture)
        self.assertIn(":real et_c2_private_checkpoint_load_stage_with_sha_v1",
                      extension)
        self.assertNotIn("c2-checkpoint-load-reconstruct-internal", extension)
        self.assertNotIn("g3r-c2-load-with-sha-internal", extension)

    def test_near_misses_reject(self) -> None:
        root = PRIVATE.read_text()
        manifest = MANIFEST.read_text()
        runner = RUNNER.read_text()
        with self.assertRaisesRegex(ValueError, "load order"):
            verify_root_and_manifest(root.replace(
                '(load "c2_checkpoint_load_extension.esk")',
                '(load "c2_public_extension.esk")'), manifest)
        with self.assertRaisesRegex(ValueError, "closure"):
            verify_root_and_manifest(root, manifest.replace(
                "native/g3r_c2_sha_stage_extension.esk",
                "native/c2_public_extension.esk"))
        with self.assertRaisesRegex(ValueError, "feature"):
            verify_runner(runner.replace("-DET_G3R_C2_STAGE_SHA_PRIVATE", ""))
        with self.assertRaisesRegex(ValueError, "feature"):
            verify_runner(runner.replace('"$directory/load-off.o"',
                                         '"$directory/objects/load-off.o"'))
        with self.assertRaisesRegex(ValueError, "feature"):
            verify_runner(runner.replace('build_mode sanitize',
                                         'build_mode normal'))
        for removed in ('--emit-object', '--emit-depfile',
                        '"${eshkol_inputs[@]}"',
                        'test -s "$directory/stage-dep.o"',
                        'test ! -s "$directory/object-build.stderr"'):
            with self.subTest(removed=removed):
                with self.assertRaisesRegex(ValueError, "feature"):
                    verify_runner(runner.replace(removed, '', 1))
        native = NATIVE_RUNNER.read_text()
        with self.assertRaisesRegex(ValueError, "compiler handoff"):
            verify_compiler_handoff(runner.replace('"$cc" \\', '"$cxx" \\'),
                                    native)
        with self.assertRaisesRegex(ValueError, "compiler handoff"):
            verify_compiler_handoff(runner, native.replace('clang_cc="$1"',
                                                          'clang_cc=clang'))
        extension = EXTENSION.read_text()
        with self.assertRaisesRegex(ValueError, "mutable binding"):
            verify_stage_allocator_binding(extension.replace(
                "(define g3r-c2-sha-stage-allocations",
                "(define (g3r-c2-sha-stage-allocations measure)"))
        for arguments in ((), ("clang",), ("/usr/bin/gcc",),
                          ("/no/such/clang",)):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    ["bash", str(NATIVE_RUNNER), *arguments],
                    capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(result.stdout, "")
        for removed in ("(g3r-c2-sha-after-factory-hook shell digest)",
                        "(c2-checkpoint-load-staging-release-internal!\n"
                        "                                         shell)",
                        "(g3r-c2-sha-fail-stop 134)"):
            with self.subTest(removed=removed):
                with self.assertRaisesRegex(ValueError, "publication recovery"):
                    verify_publication_recovery(extension.replace(removed, "", 1),
                                                runner)
        with self.assertRaisesRegex(ValueError, "diagnostics policy"):
            verify_publication_recovery(extension, runner.replace(
                "expected = warning * 4", "expected = warning * 5"))

    def test_ir_diagnostics_exact_match(self) -> None:
        runner = RUNNER.read_text()
        script = runner.split("  python3 - \"$directory/ir-object-build.stderr\" <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        warning = ("warning: <unknown>:0:0: loop not vectorized: the optimizer was "
                   "unable to perform the requested transformation; the transformation "
                   "might be disabled or specified as part of an unsupported "
                   "transformation ordering [-Wpass-failed=transform-warning]\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics"
            for value, accepted in ((warning * 4 + "4 warnings generated.\n", True),
                                    (warning * 3 + "3 warnings generated.\n", False),
                                    (warning * 4 + "4 warnings generated.\nerror: bad\n", False),
                                    ("", False)):
                path.write_text(value)
                result = subprocess.run(["python3", "-", str(path)], input=script,
                                        text=True, capture_output=True, check=False)
                self.assertEqual(result.returncode == 0, accepted)


if __name__ == "__main__":
    unittest.main()
