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
from tests.g3c4.p2_g2_held_holder_phase_overlay import MARKERS, instrument
from scripts.g3c4_held_holder_depfile import dependencies


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "tests/g3c4/p2_g2_held_holder_compiled_test.esk"
SCRIPT = ROOT / "scripts/test-g3c4-p2g2-held-holder-compiled.sh"
COMPILE_HELPER = ROOT / "scripts/g3c4-held-holder-eshkol-compile.sh"
SHIM = ROOT / "tests/g3c4/p2_g2_held_holder_compiled_native.c"
COMPOSER = ROOT / "tests/g3c4/p2_g2_terminal_composer_native.c"
OWNER = ROOT / "src/eshkol_transformer/g3c4_model_owner.c"
EXTENSION = ROOT / "native/g3c4_p2g2_terminal_held_holder_extension.esk"
DEPS = ROOT / "native/g3c4_p2g2_held_holder_compiled_eshkol_deps.txt"
INCLUDE_DIRS = ("internal/p1/lib", "internal/c1/lib", "internal/t1/lib",
                "src", "lib", "native")


def eshkol_import_closure(overrides=None):
    """Resolve top-level loads and requires with the compiled gate's include order."""
    overrides = overrides or {}
    seen = []

    def visit(name):
        if name in seen:
            return
        seen.append(name)
        source = overrides.get(name, (ROOT / name).read_text())
        loads = re.findall(r'^\s*\(load\s+"([^"]+)"', source, re.M)
        requires = re.findall(r'^\s*\(require\s+([\w.-]+)\s*\)', source, re.M)
        for target in loads:
            matches = [f"{directory}/{target}" for directory in INCLUDE_DIRS
                       if (ROOT / directory / target).is_file()]
            if len(matches) != 1:
                raise ValueError(f"unresolved or ambiguous load: {name}: {target}: {matches}")
            visit(matches[0])
        for target in requires:
            path = target.replace('.', '/') + '.esk'
            matches = [f"{directory}/{path}" for directory in INCLUDE_DIRS
                       if (ROOT / directory / path).is_file()]
            if not matches:
                raise ValueError(f"unresolved require: {name}: {target}")
            visit(matches[0])

    visit(DRIVER.relative_to(ROOT).as_posix())
    return seen


def phase_trace_contract(driver, shim, script):
    """Check the durable trace brackets the existing first-case assertions."""
    if not parse(driver):
        raise AssertionError("driver does not parse")
    for phase in (*range(1, 6), *range(10, 24), *range(30, 57)):
        if f"case {phase}:" not in shim:
            raise AssertionError(f"unmapped phase {phase}")
    phase_writer = shim[shim.index('static void held_trace(const char *phase) {'):
                        shim.index('static void held_trace_eos_args(', shim.index(
                            'static void held_trace(const char *phase) {'))]
    if 'write(4, line, (size_t)size)' not in phase_writer:
        raise AssertionError("trace must use the dedicated unbuffered fd")

    def ordered(source, fragments):
        cursor = 0
        for fragment in fragments:
            position = source.find(fragment, cursor)
            if position < 0:
                raise AssertionError(f"missing or out-of-order trace: {fragment}")
            cursor = position + len(fragment)

    ordered(driver[:driver.index("(define (fresh eos")], (
        "(held-phase 1)", "(define tokenizer", "(held-phase 2)",
        "(define model", "(held-phase 3)", "(define input",
        "(held-phase 4)", "(define args", "(held-phase 5)",
    ))
    fresh = driver[driver.index("(define (fresh eos"):
                   driver.index("(define (roots-clean?")]
    ordered(fresh, ("(held-phase 11)", "(g3c4-generator-create-internal",
                    "(held-phase 12)", "(g3c4-generator-entry-call-ready",
                    "(held-phase 13)", "(g3c4-p2g2-holder-new-internal",
                    "(held-phase 14)"))
    normal = driver[driver.index("(define (normal-case"):
                    driver.index("(define (wrong-handle)")]
    ordered(normal, (
        "(held-phase 10)", "(fresh eos kind phase)",
        '"arm normal"', "(held-phase 15)", "(held-phase 16)",
        "(g3c4-p2g2-terminal-generate-held-internal", "(held-phase 17)",
        "(g3c4-p2g2-terminal-snapshot-internal", "(held-phase 18)",
        "(held-phase 19)", '"authentic terminal count"',
        '"terminal release"', "(held-phase 20)",
        '"generator close"', "(held-phase 21)", "(held-phase 22)",
    ))
    for name, call in (
        ("begin", "__real_et_g3c4_private_p2g2_holder_begin_capture_v1("),
        ("witness", "__real_et_g3c4_private_p2g2_holder_witness_v1("),
        ("end", "__real_et_g3c4_private_p2g2_holder_end_after_busy_v1("),
        ("commit", "held_base_wrap_terminal_commit("),
    ):
        section = shim[shim.index(f"int64_t __wrap_et_g3c4_private_p2g2_holder_{name}"):] \
            if name != "commit" else shim[shim.index(
                "int64_t __wrap_et_g3c4_private_p2g2_terminal_commit_v1("):]
        ordered(section, (f'held_trace("native.{name}.enter");', call,
                          f'held_trace("native.{name}.return");'))
    for redirect in (
        '4>"$evidence/$mode-$fixture.trace"',
        '4>"$evidence/$mode-repeat.trace"',
    ):
        if redirect not in script:
            raise AssertionError(f"missing durable trace redirect: {redirect}")
    if script.count('4>"$evidence/$mode-$fixture.trace"') != 2:
        raise AssertionError("normal and fail-stop cases need separate traces")


def eos_args_fd4_contract(shim):
    """Pin the EOS argument writer itself to the durable trace descriptor."""
    start = shim.index('static void held_trace_eos_args(', shim.index(
        'static void held_trace(const char *phase) {'))
    end = shim.index('\n}', start) + 2
    writer = shim[start:end]
    if 'write(4, line, (size_t)size) != size' not in writer:
        raise AssertionError("EOS args trace must write to fd 4")
    return writer


def eos_observation_contract(driver, composer):
    """Every path reaching the EOS wrapper arms its native oracle first."""
    for start, end, marker in (
        ("(define (normal-case", "(define (wrong-handle)",
         '"mark native normal observation" (= (held-mark native 0 255) 0)'),
        ("(define (p3-first-eos-reject", "(define (retry-case",
         '"mark native first EOS observation" (= (held-mark native 0 255) 0)'),
        ("(define (retry-case", "(define first-id",
         '"capture native rollback reference" (= (held-mark native 0 255) 0)'),
    ):
        section = driver[driver.index(start):driver.index(end)]
        if not ("(held-mark native 0 255)" in section and
                section.index("(native (vector-ref triple 1))") <
                section.index(marker) <
                section.index("(g3c4-p2g2-terminal-generate-held-internal")):
            raise AssertionError("EOS native observation is not armed before generation")
    destructive = driver[driver.index('(if (string=? mode "normal")'):]
    destructive = destructive[destructive.index("(let* ((triple (fresh #f 1 2))"):]
    if not (destructive.index("(native (vector-ref triple 1))") <
            destructive.index('"mark native destructive observation"') <
            destructive.index("(held-mark native 0 255)") <
            destructive.index("(g3c4-p2g2-terminal-generate-held-internal")):
        raise AssertionError("destructive EOS observation is not armed")
    wrapper = composer[composer.index(
        "int64_t __wrap_et_g3c4_private_p2g2_eos_first_frame_carrier_v1("):]
    if not (wrapper.index("coordinator_owner == NULL") <
            wrapper.index("et_g3c4_p2g2_coordinator_test_observe_v1(")):
        raise AssertionError("EOS wrapper lacks an unarmed-oracle guard")


class HeldHolderCompiledSource(unittest.TestCase):
    def test_eos_observation_is_armed_and_missing_marks_fail_closed(self):
        driver, composer = DRIVER.read_text(), COMPOSER.read_text()
        eos_observation_contract(driver, composer)
        for marker in ('"mark native normal observation" (= (held-mark native 0 255) 0)',
                       '"mark native first EOS observation" (= (held-mark native 0 255) 0)',
                       '"capture native rollback reference" (= (held-mark native 0 255) 0)',
                       '"mark native destructive observation"\n               (= (held-mark native 0 255) 0)'):
            with self.subTest(marker=marker), self.assertRaises((ValueError, AssertionError)):
                eos_observation_contract(driver.replace(marker, "#t", 1),
                                         composer)
        with self.assertRaises((ValueError, AssertionError)):
            eos_observation_contract(driver, composer.replace(
                "coordinator_owner == NULL", "coordinator_owner != NULL", 1))

    def test_eos_boundary_trace_is_ordered_and_test_only(self):
        held, composer, owner = SHIM.read_text(), COMPOSER.read_text(), OWNER.read_text()
        feature = "ET_G3C4_P2_G2_EOS_BOUNDARY_TRACE"
        self.assertLess(held.index(f"#define {feature} 1"),
                        held.index('#include "p2_g2_terminal_composer_native.c"'))
        self.assertNotIn(f"#define {feature}", composer)
        args_writer = eos_args_fd4_contract(held)
        redirected = held.replace(args_writer, args_writer.replace(
            'write(4, line, (size_t)size)',
            'write(2, line, (size_t)size)', 1), 1)
        self.assertNotEqual(redirected, held)
        with self.assertRaisesRegex(AssertionError, "EOS args trace must write to fd 4"):
            eos_args_fd4_contract(redirected)
        for field in ("context=%p", "input=%p", "staging=%p", "bytes=%lld"):
            self.assertIn(field, held)

        wrapper = composer[composer.index(
            "int64_t __wrap_et_g3c4_private_p2g2_eos_first_frame_carrier_v1("):]
        carrier = owner[owner.index("static int64_t et_g3c4_p2g2_first_frame_carrier("):]
        sequences = (
            (wrapper, ('held_trace("eos.wrapper.enter")',
                       'held_trace_eos_args(context, input, staging, bytes)',
                       'held_trace("eos.real.enter")',
                       '__real_et_g3c4_private_p2g2_eos_first_frame_carrier_v1(\n',
                       'held_trace("eos.real.return")',
                       'held_trace("eos.wrapper.return")')),
            (carrier, ('held_trace("eos.carrier.enter")',
                       'held_trace_eos_args(\n',
                       'et_g3c4_error_reset_internal()',
                       'held_trace("eos.admit.active.enter")',
                       'et_g3c4_admit_active_call(context_candidate)',
                       'held_trace("eos.admit.active.return")',
                       'held_trace("eos.admit.input.enter")',
                       'et_g3c4_admit_input(input_candidate, 0)',
                       'held_trace("eos.admit.input.return")',
                       'held_trace("eos.pending.lookup.enter")',
                       'et_g3c4_pending_output_lookup(context, &pending)',
                       'held_trace("eos.pending.lookup.return")',
                       'held_trace("eos.prefill.enter")',
                       'et_g3c4_private_prompt_prefill_v1(\n',
                       'held_trace("eos.prefill.return")')),
        )
        for source, fragments in sequences:
            cursor = 0
            for fragment in fragments:
                with self.subTest(marker=fragment):
                    position = source.find(fragment, cursor)
                    self.assertGreaterEqual(position, 0)
                    cursor = position + len(fragment)

    def test_test_only_overlay_traces_each_unwrapped_interval(self):
        original = EXTENSION.read_text()
        overlay = instrument(original)
        self.assertTrue(parse(overlay))
        self.assertNotIn("(held-phase ", original)
        self.assertIn("(extern i64 held-phase", DRIVER.read_text().splitlines()[0])
        for before, after in MARKERS:
            with self.subTest(anchor=before[:45]):
                self.assertIn(before, original)
                self.assertIn(after, overlay)
                with self.assertRaises(ValueError):
                    instrument(original.replace(before, "", 1))
        self.assertEqual(
            [int(value) for value in re.findall(r"\(held-phase (\d+)\)", overlay)],
            list(range(30, 57)))
        post_bind = overlay[overlay.index("(held-phase 42)"):]
        ordered = ("(held-phase 42)", "(held-phase 43)",
                   "(held-phase 44)", "(g3c4-p2g2-native-eos-first",
                   "(held-phase 45)", "(held-phase 46)",
                   "(held-phase 47)", "(g3c4-p2g2-terminal-active-check",
                   "(held-phase 48)", "(held-phase 49)",
                   "(g3c4-p2g2-holder-begin!", "(held-phase 50)",
                   "(held-phase 51)", "(g3c4-p2g2-terminal-selected",
                   "(held-phase 52)", "(held-phase 53)",
                   "(t1-private-g3-decode-raw-into!", "(held-phase 54)",
                   "(held-phase 55)", "(g3c4-p2g2-terminal-active-check",
                   "(held-phase 56)")
        cursor = 0
        for fragment in ordered:
            position = post_bind.find(fragment, cursor)
            self.assertGreaterEqual(position, 0, fragment)
            cursor = position + len(fragment)
        self.assertIn('case 44: name = "eos.native.enter";', SHIM.read_text())
        self.assertIn('case 45: name = "eos.native.return";', SHIM.read_text())
        helper = COMPILE_HELPER.read_text()
        script = SCRIPT.read_text()
        self.assertLess(helper.index('"${overlay_include[@]}"'),
                        helper.index('-I "$project_root/native"'))
        self.assertIn('test "$fixture" = on', helper)
        self.assertIn('test -s "$overlay/g3c4_p2g2_terminal_held_holder_root.esk"', helper)
        self.assertIn('on = dependencies(on_depfile, root, overrides)', script)
        self.assertIn('assert on == expected', script)
        self.assertIn('assert not any(\'held_holder\' in name for name in off)', script)

    def test_depfile_rejects_canonical_extension_and_missing_overlay_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            native = root / "native"
            overlay = root / "overlay"
            native.mkdir()
            overlay.mkdir()
            names = ("g3c4_p2g2_terminal_held_holder_root.esk",
                     "g3c4_p2g2_terminal_held_holder_extension.esk")
            for name in names:
                (native / name).write_text("original")
                (overlay / name).write_text("overlay")
            overrides = {(overlay / name).resolve(): f"native/{name}"
                         for name in names}
            depfile = root / "on.d"

            def check(paths):
                depfile.write_text("object: " + " ".join(map(str, paths)) + "\n")
                return dependencies(depfile, root, overrides)

            self.assertEqual(check(overlay / name for name in names),
                             [f"native/{name}" for name in names])
            with self.assertRaises(AssertionError):
                check((overlay / names[0], native / names[1]))
            with self.assertRaises(AssertionError):
                check((native / names[0], overlay / names[1]))
            with self.assertRaises(AssertionError):
                check((overlay / names[1],))

            subprocess.run(["python3", str(ROOT /
                "tests/g3c4/p2_g2_held_holder_phase_overlay.py"), str(ROOT),
                str(overlay)], check=True)
            self.assertEqual((overlay / names[0]).read_bytes(),
                             (ROOT / "native" / names[0]).read_bytes())
            self.assertIn("(held-phase 56)", (overlay / names[1]).read_text())

    def test_phase_trace_brackets_first_generation_and_rejects_mutations(self):
        driver, shim, script = DRIVER.read_text(), SHIM.read_text(), SCRIPT.read_text()
        phase_trace_contract(driver, shim, script)
        variants = (
            (driver.replace("(held-phase 16)", "(held-phase 99)", 1), shim, script),
            (driver, shim.replace('held_trace("native.begin.enter");', "", 1), script),
            (driver, shim.replace('case 45: name = "eos.native.return";', "", 1), script),
            (driver, shim.replace('write(4, line, (size_t)size)',
                                   'write(2, line, (size_t)size)', 1), script),
            (driver, shim, script.replace(
                '4>"$evidence/$mode-$fixture.trace"', "", 1)),
        )
        for index, variant in enumerate(variants):
            with self.subTest(variant=index):
                with self.assertRaises(AssertionError):
                    phase_trace_contract(*variant)

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

            for depfile_mode, fixture, with_overlay in (
                    ("normal", "on", False), ("normal", "off", False),
                    ("normal", "on", True), ("missing", "on", False),
                    ("empty", "on", False)):
                with self.subTest(depfile_mode=depfile_mode, fixture=fixture,
                                  overlay=with_overlay):
                    directory = base / f"{depfile_mode}-{fixture}-{with_overlay}"
                    evidence = directory / "evidence"
                    evidence.mkdir(parents=True)
                    overlay = directory / "overlay"
                    if with_overlay:
                        overlay.mkdir()
                        (overlay / "g3c4_p2g2_terminal_held_holder_root.esk").write_text("fixture")
                        (overlay / EXTENSION.name).write_text("fixture")
                    call_log = directory / "calls.jsonl"
                    env = {**os.environ, "CALL_LOG": str(call_log),
                           "DEPFILE_MODE": depfile_mode}
                    result = subprocess.run(
                        ["bash", "-ec", 'source "$1"; compile_eshkol_fixture '
                         '"$2" "$3" "$4" "$5" normal "$6" /bin/true "$7" "$8"',
                         "bash", str(COMPILE_HELPER), fixture,
                         str(base / "fixture.esk"), str(directory), str(evidence),
                         str(runner), str(ROOT), str(overlay) if with_overlay else ""],
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
                        if with_overlay:
                            self.assertLess(object_call.index(str(overlay)),
                                            object_call.index(str(ROOT / "native")))
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

    def test_exact_eshkol_import_manifest_is_distinct_from_native_closure(self):
        seen = eshkol_import_closure()
        self.assertEqual(DEPS.read_text().splitlines(), seen)
        broad = (ROOT / "native/g3c4_p2g2_terminal_held_holder_source_closure.txt").read_text().splitlines()
        self.assertIn("native/g3c4_p2g2_terminal_held_holder_root.esk", broad)
        self.assertIn("native/g3c4_p2g2_terminal_held_holder_extension.esk", broad)
        required_transitive = set(seen[9:17])
        self.assertEqual(len(required_transitive), 8)
        self.assertTrue(required_transitive.issubset(broad))
        self.assertEqual(sum(not path.endswith('.esk') for path in broad), 162)

    def test_import_manifest_rejects_missing_extra_and_unknown_transitives(self):
        checkpoint = "internal/c1/lib/transformer/checkpoint_internal.esk"
        source = (ROOT / checkpoint).read_text()
        expected = DEPS.read_text().splitlines()
        self.assertIn("(require c1_sha256)", source)
        missing = eshkol_import_closure({checkpoint: source.replace(
            "(require c1_sha256)", "", 1)})
        self.assertNotIn("internal/c1/lib/c1_sha256.esk", missing)
        self.assertNotEqual(expected, missing)

        extra_path = "native/g3c4_p2g2_terminal_composer_extension.esk"
        self.assertTrue((ROOT / extra_path).is_file())
        extra = eshkol_import_closure({checkpoint: source +
            '\n(load "g3c4_p2g2_terminal_composer_extension.esk")\n'})
        self.assertIn(extra_path, extra)
        self.assertNotEqual(expected, extra)

        with self.assertRaisesRegex(ValueError, "unresolved require"):
            eshkol_import_closure({checkpoint: source +
                "\n(require missing_transitive_module)\n"})


if __name__ == "__main__":
    unittest.main()
