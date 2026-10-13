"""Source admission for the pending compiled held-holder gate."""
from pathlib import Path
import json
import os
import re
import subprocess
import tempfile
import textwrap
import unittest

from tests.c2.test_handler_order import parse


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "tests/g3c4/p2_g2_held_holder_compiled_test.esk"
SCRIPT = ROOT / "scripts/test-g3c4-p2g2-held-holder-compiled.sh"
COMPILE_HELPER = ROOT / "scripts/g3c4-held-holder-eshkol-compile.sh"
SHIM = ROOT / "tests/g3c4/p2_g2_held_holder_compiled_native.c"
DEPS = ROOT / "native/g3c4_p2g2_held_holder_compiled_eshkol_deps.txt"


class HeldHolderCompiledSource(unittest.TestCase):
    def test_driver_parses_and_reaches_exact_private_entry(self):
        source = DRIVER.read_text()
        self.assertTrue(parse(source))
        self.assertIn('(load "g3c4_p2g2_terminal_held_holder_root.esk")', source)
        self.assertIn("(g3c4-p2g2-terminal-generate-held-internal", source)
        self.assertIn("(normal-case 1 2 first-id 1)", source)
        self.assertIn("(normal-case 2 2 first-id 1)", source)
        self.assertIn("(p3-first-eos-reject first-id)", source)
        for kind, phase in ((1, 2), (2, 2), (1, 3), (2, 3)):
            self.assertIn(f"(retry-case {kind} {phase})", source)

    def test_fault_cuts_check_retry_and_original_roots(self):
        source = DRIVER.read_text()
        for fragment in (
            "wrong generation rejected before transfer",
            "exact first typed busy and one end/retry",
            "native context rolled back",
            "original ledgers and holder root restored",
            "holder disposed before terminal zero",
            '"repeat-busy"', '"postzero"',
        ):
            self.assertIn(fragment, source)
        shim = SHIM.read_text()
        extension = (ROOT / "native/g3c4_p2g2_terminal_held_holder_extension.esk").read_text()
        begin = extension[extension.index("(define (g3c4-p2g2-holder-begin!"):
                          extension.index("(define (g3c4-p2g2-holder-close!")]
        close = extension[extension.index("(define (g3c4-p2g2-holder-close!"):
                          extension.index("(define (g3c4-p2g2-held-protected-postzero!")]
        self.assertIn("(vector-set! shell 5 'held)", begin)
        self.assertIn("(g3c4-p2g2-holder-witness", close)
        self.assertLess(extension.index("(vector-set! shell 5 'held)"),
                        extension.index("(g3c4-p2g2-holder-witness", extension.index("(define (g3c4-p2g2-holder-close!")))
        self.assertIn("__real_et_g3c4_private_p2g2_holder_begin_capture_v1", shim)
        self.assertIn("__real_et_g3c4_private_p2g2_holder_witness_v1", shim)
        self.assertIn("status == 0 && (held_cut == 1 || held_cut == 2)", shim)
        begin_wrap = shim[shim.index("int64_t __wrap_et_g3c4_private_p2g2_holder_begin_capture_v1("):
                          shim.index("int64_t __real_et_g3c4_private_p2g2_holder_witness_v1(")]
        self.assertNotIn("held_cut == 1", begin_wrap)
        self.assertIn("__real_et_g3c4_private_p2g2_holder_abort_attempt_v1", shim)
        self.assertIn("ET_I64_TENSOR_CODE_ACTIVE_BORROW", shim)
        self.assertIn("ET_KERNEL_CODE_PROVIDER_REJECTED", shim)
        self.assertIn("et_g3c4_holder_tokens[et_g3c4_holder_token_count - 1].live", shim)

    def test_script_is_source_only_before_aot_and_proves_off_symbols(self):
        source = SCRIPT.read_text()
        self.assertLess(source.index("G3C4_HELD_SOURCE_ONLY"),
                        source.index("verify_toolchain"))
        self.assertIn("g3c4-held-holder-eshkol-compile.sh", source)
        self.assertIn("--emit-depfile", COMPILE_HELPER.read_text())
        self.assertIn("g3c4_p2g2_held_holder_compiled_eshkol_deps.txt", source)
        self.assertIn("g3c4_p2g2_terminal_held_holder_source_closure.txt", source)
        self.assertIn("check-g3c4-p2g2-held-holder-symbols.py", source)
        self.assertIn('off "$evidence/$mode-holder-off-nm.txt"', source)
        self.assertIn("-Wl,--wrap=et_g3c4_private_p2g2_holder_begin_capture_v1", source)
        self.assertIn("-Wl,--wrap=et_g3c4_private_p2g2_holder_witness_v1", source)
        self.assertIn("-Wl,--wrap=et_g3c4_private_p2g2_holder_abort_attempt_v1", source)
        self.assertIn("ESHKOL_ARENA_POISON=1", source)
        self.assertIn('test "$status" -eq 134', source)

    def test_depfile_requires_object_mode_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            runner = base / "runner"
            runner.write_text(textwrap.dedent("""\
                #!/usr/bin/env python3
                import json
                import os
                from pathlib import Path
                import sys

                args = sys.argv[1:]
                with Path(os.environ['CALL_LOG']).open('a') as log:
                    log.write(json.dumps(args) + '\\n')
                output = Path(args[args.index('-o') + 1])
                output.write_bytes(b'object' if '--compile-only' in args else b'executable')
                if '--compile-only' in args and os.environ.get('DEPFILE_MODE') != 'missing':
                    depfile = Path(args[args.index('--emit-depfile') + 1])
                    depfile.write_text('' if os.environ.get('DEPFILE_MODE') == 'empty'
                                       else 'object: fixture.esk\\n')
                """))
            runner.chmod(0o700)

            for depfile_mode, fixture in (("normal", "on"), ("normal", "off"),
                                          ("missing", "on"), ("empty", "on")):
                with self.subTest(depfile_mode=depfile_mode, fixture=fixture):
                    directory = base / f"{depfile_mode}-{fixture}"
                    evidence = directory / "evidence"
                    evidence.mkdir(parents=True)
                    call_log = directory / "calls.jsonl"
                    env = {**os.environ, "CALL_LOG": str(call_log),
                           "DEPFILE_MODE": depfile_mode}
                    result = subprocess.run(
                        ["bash", "-ec", 'source "$1"; compile_eshkol_fixture '
                         '"$2" "$3" "$4" "$5" normal "$6" /bin/true "$7"',
                         "bash", str(COMPILE_HELPER), fixture,
                         str(base / "fixture.esk"), str(directory), str(evidence),
                         str(runner), str(ROOT)],
                        env=env, capture_output=True, text=True, check=False)
                    calls = [json.loads(line) for line in call_log.read_text().splitlines()]
                    if depfile_mode == "normal":
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(len(calls), 2)
                        object_call, executable_call = calls
                        self.assertIn("--compile-only", object_call)
                        self.assertIn("--emit-depfile", object_call)
                        self.assertNotIn("--compile-only", executable_call)
                        self.assertNotIn("--emit-depfile", executable_call)

                        def common(arguments):
                            result = list(arguments)
                            if "--compile-only" in result:
                                result.remove("--compile-only")
                            if "--emit-depfile" in result:
                                offset = result.index("--emit-depfile")
                                del result[offset:offset + 2]
                            offset = result.index("-o")
                            del result[offset:offset + 2]
                            return result

                        self.assertEqual(common(object_call), common(executable_call))
                        self.assertTrue((directory / f"{fixture}.d").stat().st_size)
                        self.assertTrue((directory / f"held-{fixture}").stat().st_size)
                    else:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertEqual(len(calls), 1)
                        self.assertFalse((directory / f"held-{fixture}").exists())

    def test_compiled_link_wraps_terminal_calls_used_by_native_shim(self):
        script = SCRIPT.read_text()
        wrapper = script.split('cat >"$directory/cxx-wrap" <<WRAPPER\n', 1)[1].split('\nWRAPPER', 1)[0]
        self.assertIn('#include "p2_g2_terminal_composer_native.c"', SHIM.read_text())
        shim = (ROOT / "tests/g3c4/p2_g2_terminal_composer_native.c").read_text()
        for symbol in (
            "et_g3c4_private_p2g2_terminal_snapshot_v1",
            "et_g3c4_private_p2g2_terminal_ids_clone_copy_v1",
        ):
            self.assertIn("__real_" + symbol, shim)
            self.assertEqual(wrapper.count("-Wl,--wrap=" + symbol), 1)

    def test_exact_eshkol_load_manifest_is_distinct_from_native_closure(self):
        include_dirs = ("internal/p1/lib", "internal/c1/lib", "internal/t1/lib",
                        "src", "lib", "native")
        seen = []

        def visit(name):
            if name in seen:
                return
            seen.append(name)
            text = (ROOT / name).read_text()
            for target in re.findall(r'^\s*\(load\s+"([^"]+)"', text, re.M):
                matches = [f"{directory}/{target}" for directory in include_dirs
                           if (ROOT / directory / target).is_file()]
                self.assertEqual(len(matches), 1, (name, target, matches))
                visit(matches[0])

        visit(DRIVER.relative_to(ROOT).as_posix())
        self.assertEqual(DEPS.read_text().splitlines(), seen)
        broad = (ROOT / "native/g3c4_p2g2_terminal_held_holder_source_closure.txt").read_text().splitlines()
        self.assertIn("native/g3c4_p2g2_terminal_held_holder_root.esk", broad)
        self.assertIn("native/g3c4_p2g2_terminal_held_holder_extension.esk", broad)
        self.assertEqual(sum(not path.endswith('.esk') for path in broad), 162)


if __name__ == "__main__":
    unittest.main()
