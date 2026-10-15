"""Host-only lexical and source isolation checks for the noninstalled E3 fault tuple."""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/e3_private"
RETENTION = TESTS / "retention_package"
FAULT = TESTS / "fault_package"


def rows(prefix: Path, suffix: str) -> list[str]:
    return Path(f"{prefix}_{suffix}.txt").read_text().splitlines()


class FaultTupleTest(unittest.TestCase):
    @staticmethod
    def function_expression(source: str, name: str):
        """Parse an actual test-only predicate, including its nested clauses."""
        start = source.index(f"(define ({name} ")
        tokens = re.findall(r"\(|\)|[^\s()]+", source[start:])
        def parse(index):
            if tokens[index] != "(":
                return tokens[index], index + 1
            values = []
            index += 1
            while tokens[index] != ")":
                value, index = parse(index)
                values.append(value)
            return values, index + 1
        definition, _ = parse(0)
        return definition[2]

    @staticmethod
    def bound_expression(source: str):
        return FaultTupleTest.function_expression(source, "e3-test-fault-bound?")

    @staticmethod
    def eval_bound(expression, environment):
        if isinstance(expression, str):
            return int(expression) if expression.isdigit() else environment[expression]
        op = expression[0]
        if op == "let":
            local = environment.copy()
            for name, value in expression[1]:
                local[name] = FaultTupleTest.eval_bound(value, environment)
            return FaultTupleTest.eval_bound(expression[2], local)
        if op == "and":
            return all(FaultTupleTest.eval_bound(value, environment)
                       for value in expression[1:])
        args = [FaultTupleTest.eval_bound(value, environment)
                for value in expression[1:]]
        if op == "vector-ref":
            return args[0][args[1]]
        if op == "vector?":
            return isinstance(args[0], list)
        if op == "=":
            return args[0] == args[1]
        if op == "eq?":
            return args[0] is args[1]
        if op == "e3-test-fault-path=?":
            return args[0] == tuple(args[1])
        raise AssertionError(f"unhandled guard operation: {op}")

    @staticmethod
    def eval_probe(expression, environment):
        if isinstance(expression, str):
            if expression == "#t":
                return True
            if expression == "#f":
                return False
            if expression.startswith("'"):
                return expression
            return int(expression) if expression.isdigit() else environment[expression]
        op = expression[0]
        if op == "and":
            return all(FaultTupleTest.eval_probe(value, environment)
                       for value in expression[1:])
        if op == "or":
            return any(FaultTupleTest.eval_probe(value, environment)
                       for value in expression[1:])
        if op == "not":
            return not FaultTupleTest.eval_probe(expression[1], environment)
        if op == "if":
            branch = expression[2] if FaultTupleTest.eval_probe(
                expression[1], environment) else expression[3]
            return FaultTupleTest.eval_probe(branch, environment)
        if op == "let":
            named = isinstance(expression[1], str)
            bindings = expression[2] if named else expression[1]
            names = [binding[0] for binding in bindings]
            values = [FaultTupleTest.eval_probe(binding[1], environment)
                      for binding in bindings]
            body = expression[3] if named else expression[2]
            def invoke(*arguments):
                local = environment | dict(zip(names, arguments, strict=True))
                if named:
                    local[expression[1]] = invoke
                return FaultTupleTest.eval_probe(body, local)
            return invoke(*values)
        args = [FaultTupleTest.eval_probe(value, environment)
                for value in expression[1:]]
        if op == "vector":
            return args
        if op == "vector-ref":
            return args[0][args[1]]
        if op == "vector-length":
            return len(args[0])
        if op == "=":
            return args[0] == args[1]
        if op == "<=":
            return args[0] <= args[1]
        if op == ">=":
            return args[0] >= args[1]
        if op == ">":
            return args[0] > args[1]
        if op == "+":
            return sum(args)
        if op == "-":
            return args[0] - args[1]
        if op == "*":
            return args[0] * args[1]
        if op in environment and callable(environment[op]):
            return environment[op](*args)
        raise AssertionError(f"unhandled probe operation: {op}")

    def test_exact_derived_inventories(self):
        for suffix in ("defined_symbols", "public_exports", "public_strings"):
            self.assertEqual(rows(FAULT, suffix),
                             sorted(rows(RETENTION, suffix) + ["et_e3_fault_run_v1"]))
        for suffix in ("undefined_symbols", "native_objects"):
            self.assertEqual(rows(FAULT, suffix), rows(RETENTION, suffix))
        self.assertEqual(rows(FAULT, "private_renames"),
                         rows(RETENTION, "private_renames") +
                         ["e3-test-fault-dispatch et_e3_private_fault_dispatch_cabi_v1"])
        self.assertEqual(rows(FAULT, "source_closure"),
                         ["tests/e3_private/fault_driver_root.esk"] +
                         rows(RETENTION, "source_closure") +
                         ["tests/e3_private/fault_driver.esk",
                          "tests/e3_private/n5_cap_driver.esk"])
        native = rows(RETENTION, "native_source_closure")
        native.insert(native.index("tests/e3_private/retention_bridge.c"),
                      "tests/e3_private/fault_bridge.c")
        self.assertEqual(rows(FAULT, "native_source_closure"), native)
        self.assertEqual(rows(FAULT, "archive_members"), ["e3_fault_package.o"])
        for suffix in ("sanitized_noninstrumentation_undefined_symbols",
                       "sanitized_instrumentation_undefined_symbols"):
            self.assertEqual(rows(FAULT, suffix), rows(RETENTION, suffix))
        root = (TESTS / "fault_driver_root.esk").read_text()
        self.assertEqual(root.count('(load "retention_driver_root.esk")'), 1)
        self.assertEqual(root.count('(load "fault_driver.esk")'), 1)
        self.assertEqual(root.count('(load "n5_cap_driver.esk")'), 1)
        self.assertNotIn("fault_driver", (ROOT / "native/e3_private_driver_root.esk").read_text())

    def test_policy_rejects_mixed_partial_symlink_and_ambient_roots(self):
        policy = ROOT / "scripts/e3-private-package-policy.sh"
        inputs = [TESTS / "fault_driver_root.esk", TESTS / "fault_bridge.c",
                  Path(f"{FAULT}_private_renames.txt"),
                  Path(f"{FAULT}_public_exports.txt")]
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
                 *(str(p) for p in paths), str(policy), extra],
                capture_output=True, text=True)
        accepted = select(inputs)
        self.assertEqual((accepted.returncode, accepted.stdout),
                         (0, "fault\n"), accepted.stderr)
        for index, replacement in (
            (0, TESTS / "retention_driver_root.esk"),
            (1, TESTS / "retention_bridge.c"),
            (2, Path(f"{RETENTION}_private_renames.txt")),
            (3, Path(f"{RETENTION}_public_exports.txt")),
        ):
            mixed = inputs.copy()
            mixed[index] = replacement
            self.assertEqual(select(mixed).returncode, 77)
        self.assertEqual(select(inputs, "ambient").returncode, 77)
        with tempfile.TemporaryDirectory() as directory:
            alias = Path(directory) / "fault_driver_root.esk"
            alias.symlink_to(inputs[0])
            self.assertEqual(select([alias, *inputs[1:]]).returncode, 77)

    def test_native_define_and_closed_binding(self):
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        self.assertEqual(builder.count("package_source_flags+=(-DET_E3_FAULT_TESTING"), 1)
        seam = builder[builder.index('if [[ "${package_policy}" == e3-private-aggregate &&\n'
                                     '          "${e3_tuple_kind}" == fault ]]; then'):]
        self.assertIn("package_source_flags+=(-DET_E3_FAULT_TESTING", seam[:350])
        self.assertEqual(seam[:650].count("-DET_E3_PARTIAL_PIN_TESTING"), 2)
        self.assertIn("et_g3t_model_pins_begin_fail_before_internal",
                      builder[builder.index('if [[ "${e3_tuple_kind}" == fault ]]; then\n'
                                            '        e3_required_private+=('):])
        source = (TESTS / "fault_driver.esk").read_text()
        signatures = (
            r'\(extern i64 e3-test-fault-arm ptr i64 i64 i64\s+:real et_e3_test_fault_arm_v1\)',
            r'\(extern i64 e3-test-fault-ref ptr i64\s+:real et_e3_test_fault_ref_v1\)',
            r'\(extern i64 e3-test-fault-reset ptr\s+:real et_e3_test_fault_reset_v1\)',
            r'\(extern i64 e3-test-fault-seed-publication ptr\s+:real et_e3_test_fault_seed_publication_v1\)',
            r'\(extern i64 e3-fault-value-borrow-begin ptr i64\s+:real et_e3_fault_value_borrow_begin_v1\)',
            r'\(extern i64 e3-fault-value-borrow-end ptr i64\s+:real et_e3_fault_value_borrow_end_v1\)',
        )
        def valid(text):
            return all(len(re.findall(pattern, text)) == 1 for pattern in signatures)
        self.assertTrue(valid(source))
        self.assertFalse(valid(source.replace("arm ptr i64 i64 i64", "arm ptr i64 i64")))
        self.assertFalse(valid(source.replace("et_e3_test_fault_ref_v1",
                                              "et_e3_private_forward_role_v1")))
        self.assertIn("(e3-frame-entry-live frame 'e3-test-fault-execute)", source)
        self.assertIn("(native (vector-ref entry 4))", source)
        execute = source.split("(define (e3-test-fault-execute ", 1)[1]
        execute = execute.split("(define (e3-test-fault-dispatch", 1)[0]
        self.assertLess(execute.index("(e3-test-fault-reset native)"),
                        execute.index("(e3-test-retention-call! model dataset frame"))
        self.assertNotIn("e3-test-fault-arm", (TESTS / "retention_driver.esk").read_text())

    def test_partial_pin_site_and_native_symbol_boundary(self):
        driver = (TESTS / "fault_driver.esk").read_text()
        frame = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        pins = (ROOT / "src/eshkol_transformer/m3_call_f32_integration.c").read_text()
        self.assertIn("(and (= (vector-ref scenario 3) 4)\n"
                      "                  (>= (vector-ref scenario 4) 0)\n"
                      "                  (<= (vector-ref scenario 4) 13)", driver)
        self.assertIn("site == 4 && ordinal >= 0 && ordinal < 14 && occurrence == 1", frame)
        self.assertLess(frame.index("frame->fault_state = 2; /* Consume before M3 admission"),
                        frame.index("pin_status = et_g3t_model_pins_begin_fail_before_internal"))
        self.assertIn("(if (= site 3) 42\n                    (if (= site 4) 47", driver)
        self.assertIn("(if (or (= site 3) (= site 4) (= site 5) (= site 6) (= site 7) (= site 8) (= site 9))\n        (e3-test-fault-check 'independent-twin", driver)
        self.assertIn("if (fail_before >= M3_CALL_PIN_COUNT)", pins)
        self.assertIn("m3_call_drain(pins, i)", pins)
        self.assertIn("for(size_t n=0;n<14;++n)", (ROOT / "tests/m3cg/test_pins.c").read_text())

    def test_stage8_ce_cut_and_dispatch_before_denial_mutant(self):
        frame = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        driver = (TESTS / "fault_driver.esk").read_text()
        native = (ROOT / "tests/e3_native/test_frame.c").read_text()
        ce = frame.split("int64_t et_e3_private_ce_v1(void *pointer) {", 1)[1]
        ce = ce.split("int64_t et_e3_private_reduce_v1", 1)[0]
        denial = "frame->fault_state = 2; /* Consume before L2 dispatch. */"
        dispatch = "int64_t rc = e3_dispatch(frame, E3_RUNTIME_L2,"
        def safe_order(candidate):
            return (candidate.index("frame->phase != E3_CE_READY") <
                    candidate.index("frame->fault_site == 5") <
                    candidate.index(denial) < candidate.index(dispatch))
        self.assertTrue(safe_order(ce))
        moved = ce[:ce.index("#ifdef ET_E3_FAULT_TESTING")] + \
            ce[ce.index(dispatch):ce.index("if (!rc) frame->next_role = 22;")] + \
            ce[ce.index("#ifdef ET_E3_FAULT_TESTING"):ce.index(dispatch)]
        self.assertFalse(safe_order(moved))
        self.assertIn("site == 5 && ordinal == 0 && occurrence >= 1 && occurrence <= 2", frame)
        self.assertIn("(if (= site 5) 107 95)", driver)
        self.assertIn("((= site 5) 'typed-stage8)", driver)
        self.assertIn("(if (= site 5) 8 7)", driver)
        self.assertIn("(<= (vector-ref scenario 5) 2)", driver)
        self.assertIn("(= (vector-ref scenario 4) 0)", driver)
        self.assertIn("(e3-test-fault-make-twin present? corpus-path)", driver)
        for clause in ("(e3-test-retention-modes=? frame modes)",
                       "(e3-test-retention-model=? model model-image)",
                       "(e3-test-fault-probes-restored?",
                       "(e3-test-fault-publication frame destinations)",
                       "(d2-core-bytes=? cursor (d2-token-dataset-cursor dataset))"):
            self.assertIn(clause, driver)
        self.assertIn("observed_dispatches == before_ce", native)
        self.assertIn("frame->fault_reached == occurrence", native)
        compiler = shutil.which("clang")
        if not compiler:
            self.skipTest("clang unavailable for CE dispatch mutant")
        cut_start = ce.index("#ifdef ET_E3_FAULT_TESTING")
        cut_end = ce.index("#endif", cut_start) + len("#endif\n")
        cut = ce[cut_start:cut_end]
        mutant_ce = ce[:cut_start] + ce[cut_end:]
        after_dispatch = mutant_ce.index("  if (!rc) frame->next_role = 22;")
        mutant_ce = mutant_ce[:after_dispatch] + cut + mutant_ce[after_dispatch:]
        mutant_source = frame.replace(ce, mutant_ce, 1)
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            (fixture / "src/eshkol_transformer").mkdir(parents=True)
            (fixture / "tests/e3_native").mkdir(parents=True)
            (fixture / "src/eshkol_transformer/e3_frame.c").write_text(mutant_source)
            (fixture / "tests/e3_native/test_frame.c").write_text(native)
            sources = [fixture / "tests/e3_native/test_frame.c"] + [
                ROOT / path for path in (
                    "src/eshkol_transformer/m3_call_f32_integration.c",
                    "src/eshkol_transformer/m3_i64_integration.c",
                    "native/kernel_abi.c", "native/t1_i64_shell.c",
                    "native/n3k_primitives_provider.c", "native/n2_primitives_provider.c",
                    "native/a2_attention_provider.c", "native/indexed_cross_entropy.c",
                    "native/l3s_masked_objective_provider.c",
                    "native/e3_evaluation_metrics_provider.c")]
            build = subprocess.run([
                compiler, "-std=c11", "-O2", "-DET_F32_TENSOR_TESTING",
                "-DET_I64_TENSOR_TESTING", "-DET_E3_PARTIAL_PIN_TESTING",
                "-I", str(ROOT / "include"), "-I", str(ROOT / "native"),
                "-I", str(ROOT / "src/eshkol_transformer"),
                *(str(path) for path in sources), "-lm", "-o", str(fixture / "frame")],
                capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            result = subprocess.run([str(fixture / "frame")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("observed_dispatches == before_ce", result.stderr)

        compiler, nm = shutil.which("clang"), shutil.which("nm")
        if not compiler or not nm:
            self.skipTest("clang/nm unavailable for native symbol check")
        with tempfile.TemporaryDirectory() as directory:
            def symbols(source, flags, name):
                obj = Path(directory) / name
                subprocess.run([compiler, "-std=c11", "-O0", "-c",
                                "-I", str(ROOT / "include"),
                                "-I", str(ROOT / "native"), *flags,
                                str(ROOT / source), "-o", str(obj)],
                               check=True, capture_output=True, text=True)
                return subprocess.run([nm, "-g", str(obj)], check=True,
                                      capture_output=True, text=True).stdout
            on = ["-DET_E3_PARTIAL_PIN_TESTING", "-DET_E3_FAULT_TESTING"]
            off = []
            m3_on = symbols("src/eshkol_transformer/m3_call_f32_integration.c", on, "m3-on.o")
            e3_on = symbols("src/eshkol_transformer/e3_frame.c", on, "e3-on.o")
            m3_off = symbols("src/eshkol_transformer/m3_call_f32_integration.c", off, "m3-off.o")
            e3_off = symbols("src/eshkol_transformer/e3_frame.c", off, "e3-off.o")
            symbol = "et_g3t_model_pins_begin_fail_before_internal"
            self.assertRegex(m3_on, rf"(?m)^\S+ T {symbol}$")
            self.assertRegex(e3_on, rf"(?m)^                 U {symbol}$")
            self.assertNotIn(symbol, m3_off + e3_off)

    def test_stage9_l3s_cut_and_dispatch_before_denial_mutant(self):
        frame = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        driver = (TESTS / "fault_driver.esk").read_text()
        native = (ROOT / "tests/e3_native/test_frame.c").read_text()
        reduce = frame.split("int64_t et_e3_private_reduce_v1(void *pointer) {", 1)[1]
        reduce = reduce.split("int64_t et_e3_private_correct_v1", 1)[0]
        denial = "frame->fault_state = 2; /* Consume before L3S dispatch. */"
        dispatch = "int64_t rc = e3_dispatch(frame, E3_RUNTIME_L3S,"
        self.assertLess(reduce.index("frame->phase != E3_CE_READY"),
                        reduce.index("frame->fault_site == 6"))
        self.assertLess(reduce.index("frame->fault_site == 6"),
                        reduce.index(denial))
        self.assertLess(reduce.index(denial), reduce.index(dispatch))
        self.assertIn("site == 6 && ordinal == 0 && occurrence >= 1 && occurrence <= 2",
                      frame)
        errors = (ROOT / "native/e3_errors_extension.esk").read_text()
        self.assertIn('((= stage 9) "evaluation reduction failed")', errors)
        self.assertIn("(+ (* stage 12) (e3-error-category-index category))", errors)
        self.assertIn("(else 11)", errors)
        self.assertEqual(9 * 12 + 11, 119)
        for clause in (
            "((= site 6) 'typed-stage9)",
            "(if (= site 6) 119",
            "(if (= site 6) 9",
            "(and (= (vector-ref scenario 3) 6)",
            "(= (vector-ref scenario 4) 0)",
            "(<= (vector-ref scenario 5) 2)",
            "(e3-test-fault-make-twin present? corpus-path)",
            "(e3-test-retention-modes=? frame modes)",
            "(e3-test-retention-model=? model model-image)",
            "(e3-test-fault-probes-restored?",
            "(d2-core-bytes=? cursor (d2-token-dataset-cursor dataset))",
        ):
            self.assertIn(clause, driver)
        self.assertIn("observed_dispatches == before_reduce", native)
        self.assertIn("frame->phase == E3_CE_READY && frame->next_role == 22",
                      native)
        compiler = shutil.which("clang")
        if not compiler:
            self.skipTest("clang unavailable for L3S dispatch mutant")
        cut_start = reduce.index("#ifdef ET_E3_FAULT_TESTING")
        cut_end = reduce.index("#endif", cut_start) + len("#endif\n")
        cut = reduce[cut_start:cut_end]
        mutant_reduce = reduce[:cut_start] + reduce[cut_end:]
        after_dispatch = mutant_reduce.index("  if (!rc) frame->phase = E3_REDUCED;")
        mutant_reduce = (mutant_reduce[:after_dispatch] + cut +
                         mutant_reduce[after_dispatch:])
        mutant_source = frame.replace(reduce, mutant_reduce, 1)
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            (fixture / "src/eshkol_transformer").mkdir(parents=True)
            (fixture / "tests/e3_native").mkdir(parents=True)
            (fixture / "src/eshkol_transformer/e3_frame.c").write_text(mutant_source)
            (fixture / "tests/e3_native/test_frame.c").write_text(native)
            sources = [fixture / "tests/e3_native/test_frame.c"] + [
                ROOT / path for path in (
                    "src/eshkol_transformer/m3_call_f32_integration.c",
                    "src/eshkol_transformer/m3_i64_integration.c",
                    "native/kernel_abi.c", "native/t1_i64_shell.c",
                    "native/n3k_primitives_provider.c", "native/n2_primitives_provider.c",
                    "native/a2_attention_provider.c", "native/indexed_cross_entropy.c",
                    "native/l3s_masked_objective_provider.c",
                    "native/e3_evaluation_metrics_provider.c")]
            build = subprocess.run([
                compiler, "-std=c11", "-O2", "-DET_F32_TENSOR_TESTING",
                "-DET_I64_TENSOR_TESTING", "-DET_E3_PARTIAL_PIN_TESTING",
                "-I", str(ROOT / "include"), "-I", str(ROOT / "native"),
                "-I", str(ROOT / "src/eshkol_transformer"),
                *(str(path) for path in sources), "-lm", "-o", str(fixture / "frame")],
                capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            result = subprocess.run([str(fixture / "frame")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("observed_dispatches == before_reduce", result.stderr)

    def test_stage10_correct_cut_and_dispatch_before_denial_mutant(self):
        frame = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        driver = (TESTS / "fault_driver.esk").read_text()
        native = (ROOT / "tests/e3_native/test_frame.c").read_text()
        correct = frame.split("int64_t et_e3_private_correct_v1(void *pointer) {", 1)[1]
        correct = correct.split("static int64_t e3_accumulate_dispatch", 1)[0]
        denial = "frame->fault_state = 2; /* Consume before E3-N correctness dispatch. */"
        dispatch = "if (et_kernel_runtime_dispatch(frame->runtime[E3_RUNTIME_METRICS],"
        self.assertLess(correct.index("frame->phase != E3_REDUCED"),
                        correct.index("frame->fault_site == 7"))
        self.assertLess(correct.index("frame->fault_site == 7"),
                        correct.index(denial))
        self.assertLess(correct.index(denial), correct.index(dispatch))
        self.assertIn("site == 7 && ordinal == 0 && occurrence >= 1 && occurrence <= 2",
                      frame)
        errors = (ROOT / "native/e3_errors_extension.esk").read_text()
        self.assertIn('((= stage 10) "evaluation correctness metric failed")', errors)
        self.assertIn("(+ (* stage 12) (e3-error-category-index category))", errors)
        self.assertIn("(else 11)", errors)
        self.assertEqual(10 * 12 + 11, 131)
        for clause in (
            "((= site 7) 'typed-stage10)",
            "(if (= site 7) 131",
            "(if (= site 7) 10",
            "(and (= (vector-ref scenario 3) 7)",
            "(= (vector-ref scenario 4) 0)",
            "(<= (vector-ref scenario 5) 2)",
            "(e3-test-fault-make-twin present? corpus-path)",
            "(e3-test-retention-modes=? frame modes)",
            "(e3-test-retention-model=? model model-image)",
            "(e3-test-fault-probes-restored?",
            "(d2-core-bytes=? cursor (d2-token-dataset-cursor dataset))",
        ):
            self.assertIn(clause, driver)
        self.assertIn("observed_dispatches == before_correct", native)
        self.assertIn("observed_dispatches == before_ce + 2", native)
        self.assertIn("frame->phase == E3_REDUCED && frame->next_role == 22",
                      native)
        compiler = shutil.which("clang")
        if not compiler:
            self.skipTest("clang unavailable for correct dispatch mutant")
        cut_start = correct.index("#ifdef ET_E3_FAULT_TESTING")
        cut_end = correct.index("#endif", cut_start) + len("#endif\n")
        cut = correct[cut_start:cut_end]
        mutant_correct = correct[:cut_start] + correct[cut_end:]
        after_dispatch = mutant_correct.index("  if (!rc) frame->phase = E3_CORRECTED;")
        mutant_correct = (mutant_correct[:after_dispatch] + cut +
                          mutant_correct[after_dispatch:])
        mutant_source = frame.replace(correct, mutant_correct, 1)
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            (fixture / "src/eshkol_transformer").mkdir(parents=True)
            (fixture / "tests/e3_native").mkdir(parents=True)
            (fixture / "src/eshkol_transformer/e3_frame.c").write_text(mutant_source)
            (fixture / "tests/e3_native/test_frame.c").write_text(native)
            sources = [fixture / "tests/e3_native/test_frame.c"] + [
                ROOT / path for path in (
                    "src/eshkol_transformer/m3_call_f32_integration.c",
                    "src/eshkol_transformer/m3_i64_integration.c",
                    "native/kernel_abi.c", "native/t1_i64_shell.c",
                    "native/n3k_primitives_provider.c", "native/n2_primitives_provider.c",
                    "native/a2_attention_provider.c", "native/indexed_cross_entropy.c",
                    "native/l3s_masked_objective_provider.c",
                    "native/e3_evaluation_metrics_provider.c")]
            build = subprocess.run([
                compiler, "-std=c11", "-O2", "-DET_F32_TENSOR_TESTING",
                "-DET_I64_TENSOR_TESTING", "-DET_E3_PARTIAL_PIN_TESTING",
                "-I", str(ROOT / "include"), "-I", str(ROOT / "native"),
                "-I", str(ROOT / "src/eshkol_transformer"),
                *(str(path) for path in sources), "-lm", "-o", str(fixture / "frame")],
                capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            result = subprocess.run([str(fixture / "frame")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("observed_dispatches == before_correct", result.stderr)

    def test_stage11_accumulate_cut_and_dispatch_before_denial_mutant(self):
        frame = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        driver = (TESTS / "fault_driver.esk").read_text()
        native = (ROOT / "tests/e3_native/test_frame.c").read_text()
        accumulate = frame.split("int64_t et_e3_private_accumulate_v1(void *pointer) {", 1)[1]
        accumulate = accumulate.split("int64_t et_e3_private_finalize_v1", 1)[0]
        denial = "frame->fault_state = 2; /* Consume before accumulation view borrow/K1 dispatch. */"
        dispatch = "int64_t rc = e3_accumulate_dispatch(frame);"
        self.assertLess(accumulate.index("frame->phase != E3_CORRECTED"),
                        accumulate.index("frame->fault_site == 8"))
        self.assertLess(accumulate.index("frame->fault_site == 8"),
                        accumulate.index(denial))
        self.assertLess(accumulate.index(denial), accumulate.index(dispatch))
        self.assertIn("site == 8 && ordinal == 0 && occurrence >= 1 && occurrence <= 2",
                      frame)
        errors = (ROOT / "native/e3_errors_extension.esk").read_text()
        self.assertIn('((= stage 11) "evaluation accumulation failed")', errors)
        self.assertIn("(+ (* stage 12) (e3-error-category-index category))", errors)
        self.assertIn("(else 11)", errors)
        self.assertEqual(11 * 12 + 11, 143)
        for clause in (
            "((= site 8) 'typed-stage11)",
            "(if (= site 8) 143",
            "(if (= site 8) 11",
            "(and (= (vector-ref scenario 3) 8)",
            "(= (vector-ref scenario 4) 0)",
            "(<= (vector-ref scenario 5) 2)",
            "(e3-test-fault-make-twin present? corpus-path)",
            "(e3-test-retention-modes=? frame modes)",
            "(e3-test-retention-model=? model model-image)",
            "(e3-test-fault-probes-restored?",
            "(d2-core-bytes=? cursor (d2-token-dataset-cursor dataset))",
        ):
            self.assertIn(clause, driver)
        self.assertIn("observed_dispatches == before_accumulate", native)
        self.assertIn("frame->counts[frame->sum_bank][1] == 1", native)
        self.assertIn("frame->phase == E3_CORRECTED && frame->sum_bank == bank",
                      native)
        compiler = shutil.which("clang")
        if not compiler:
            self.skipTest("clang unavailable for accumulate dispatch mutant")
        feature_off = subprocess.run([
            compiler, "-E", "-P", "-std=c11", "-I", str(ROOT / "include"),
            "-I", str(ROOT / "native"),
            str(ROOT / "src/eshkol_transformer/e3_frame.c")],
            capture_output=True, text=True)
        self.assertEqual(feature_off.returncode, 0, feature_off.stderr)
        self.assertNotIn("frame->fault_site == 8", feature_off.stdout)
        cut_start = accumulate.index("#ifdef ET_E3_FAULT_TESTING")
        cut_end = accumulate.index("#endif", cut_start) + len("#endif\n")
        cut = accumulate[cut_start:cut_end]
        mutant_accumulate = accumulate[:cut_start] + accumulate[cut_end:]
        after_dispatch = mutant_accumulate.index("  if (!rc) {")
        mutant_accumulate = (mutant_accumulate[:after_dispatch] + cut +
                          mutant_accumulate[after_dispatch:])
        mutant_source = frame.replace(accumulate, mutant_accumulate, 1)
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            (fixture / "src/eshkol_transformer").mkdir(parents=True)
            (fixture / "tests/e3_native").mkdir(parents=True)
            (fixture / "src/eshkol_transformer/e3_frame.c").write_text(mutant_source)
            (fixture / "tests/e3_native/test_frame.c").write_text(native)
            sources = [fixture / "tests/e3_native/test_frame.c"] + [
                ROOT / path for path in (
                    "src/eshkol_transformer/m3_call_f32_integration.c",
                    "src/eshkol_transformer/m3_i64_integration.c",
                    "native/kernel_abi.c", "native/t1_i64_shell.c",
                    "native/n3k_primitives_provider.c", "native/n2_primitives_provider.c",
                    "native/a2_attention_provider.c", "native/indexed_cross_entropy.c",
                    "native/l3s_masked_objective_provider.c",
                    "native/e3_evaluation_metrics_provider.c")]
            build = subprocess.run([
                compiler, "-std=c11", "-O2", "-DET_F32_TENSOR_TESTING",
                "-DET_I64_TENSOR_TESTING", "-DET_E3_PARTIAL_PIN_TESTING",
                "-I", str(ROOT / "include"), "-I", str(ROOT / "native"),
                "-I", str(ROOT / "src/eshkol_transformer"),
                *(str(path) for path in sources), "-lm", "-o", str(fixture / "frame")],
                capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            result = subprocess.run([str(fixture / "frame")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("observed_dispatches == before_accumulate", result.stderr)

    def test_stage12_finalize_cut_and_dispatch_before_denial_mutant(self):
        frame = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        driver = (TESTS / "fault_driver.esk").read_text()
        native = (ROOT / "tests/e3_native/test_frame.c").read_text()
        finalize = frame.split("int64_t et_e3_private_finalize_v1(void *pointer) {", 1)[1]
        finalize = finalize.split("int64_t et_e3_private_cleanup_preflight_v1", 1)[0]
        cut_start = finalize.index("#ifdef ET_E3_FAULT_TESTING")
        cut_end = finalize.index("#endif", cut_start) + len("#endif\n")
        cut = finalize[cut_start:cut_end]
        dispatch = "int64_t rc = e3_dispatch(frame, E3_RUNTIME_METRICS,"
        self.assertLess(finalize.index("frame->counts[bank][1] <= 0"), cut_start)
        self.assertLess(cut_end, finalize.index("e3_operand inputs[]"))
        self.assertLess(cut_end, finalize.index(dispatch))
        self.assertIn("frame->fault_state = 2; /* Consume before result views/K1 dispatch. */", cut)
        self.assertIn("site == 9 && ordinal == 0 && occurrence >= 1 && occurrence <= 2", frame)
        self.assertIn("((= site 9) 'typed-stage12-finalize)", driver)
        self.assertIn("(if (= site 9) 155", driver)
        self.assertIn("(if (= site 9) 12", driver)
        self.assertIn("(and (= (vector-ref scenario 3) 9)", driver)
        self.assertIn("(and (= (vector-ref scenario 3) 9)\n                  (= (vector-ref scenario 4) 0)\n                  (= (vector-ref scenario 5) 1))", driver)
        self.assertIn("(e3-test-fault-make-twin present? corpus-path)", driver)
        for witness in ("'prepared-gradient", "'cursor", "'output", "'modes",
                        "'model", "'rng", "'owners", "'independent-twin",
                        "'retry-output", "'retry-rng"):
            self.assertIn(witness, driver)
        self.assertIn("observed_dispatches == before_finalize", native)
        self.assertIn("frame->committed == 0", native)
        self.assertIn("frame->fault_state == 1 && frame->fault_reached == 1 &&", native)
        self.assertIn("(occurrence == 1 ? 2 : 3)", native)
        self.assertEqual(12 * 12 + 11, 155)
        compiler = shutil.which("clang")
        if not compiler:
            self.skipTest("clang unavailable for finalizer dispatch mutant")
        feature_off = subprocess.run([
            compiler, "-E", "-P", "-std=c11", "-I", str(ROOT / "include"),
            "-I", str(ROOT / "native"),
            str(ROOT / "src/eshkol_transformer/e3_frame.c")],
            capture_output=True, text=True)
        self.assertEqual(feature_off.returncode, 0, feature_off.stderr)
        self.assertNotIn("frame->fault_site == 9", feature_off.stdout)
        mutant_finalize = finalize[:cut_start] + finalize[cut_end:]
        after_dispatch = mutant_finalize.index("  if (!rc) {")
        mutant_finalize = mutant_finalize[:after_dispatch] + cut + mutant_finalize[after_dispatch:]
        mutant_source = frame.replace(finalize, mutant_finalize, 1)
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            (fixture / "src/eshkol_transformer").mkdir(parents=True)
            (fixture / "tests/e3_native").mkdir(parents=True)
            (fixture / "src/eshkol_transformer/e3_frame.c").write_text(mutant_source)
            (fixture / "tests/e3_native/test_frame.c").write_text(native)
            sources = [fixture / "tests/e3_native/test_frame.c"] + [
                ROOT / path for path in (
                    "src/eshkol_transformer/m3_call_f32_integration.c",
                    "src/eshkol_transformer/m3_i64_integration.c",
                    "native/kernel_abi.c", "native/t1_i64_shell.c",
                    "native/n3k_primitives_provider.c", "native/n2_primitives_provider.c",
                    "native/a2_attention_provider.c", "native/indexed_cross_entropy.c",
                    "native/l3s_masked_objective_provider.c",
                    "native/e3_evaluation_metrics_provider.c")]
            build = subprocess.run([
                compiler, "-std=c11", "-O2", "-DET_F32_TENSOR_TESTING",
                "-DET_I64_TENSOR_TESTING", "-DET_E3_PARTIAL_PIN_TESTING",
                "-I", str(ROOT / "include"), "-I", str(ROOT / "native"),
                "-I", str(ROOT / "src/eshkol_transformer"),
                *(str(path) for path in sources), "-lm", "-o", str(fixture / "frame")],
                capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            result = subprocess.run([str(fixture / "frame")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("observed_dispatches == before_finalize", result.stderr)

    def test_generated_stage_is_owned_and_fresh(self):
        helper = ROOT / "scripts/e3-mode-graph-stage-cleanup.sh"
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory)
            stage = gate / ".e3-fault.A1b2C3"
            stage.mkdir()
            obj = stage / "e3_fault_package.o"
            def select(path):
                return subprocess.run(
                    ["bash", "-c", 'source "$1"; e3_fault_generated_stage "$2"',
                     "bash", str(helper), str(path)], capture_output=True, text=True)
            self.assertEqual((select(obj).returncode, select(obj).stdout),
                             (0, f"{gate}/.e3-fault-generated\n"))
            self.assertNotEqual(select(stage / "foreign.o").returncode, 0)
            alias = gate / ".e3-fault.Z9y8X7"
            alias.symlink_to(stage, target_is_directory=True)
            self.assertNotEqual(select(alias / obj.name).returncode, 0)
            generated = gate / ".e3-fault-generated"
            generated.mkdir()
            self.assertNotEqual(select(obj).returncode, 0)

    def test_sanitizer_and_unresolved_policies_are_fault_scoped(self):
        sanitize = ROOT / "scripts/e3-retention-sanitizer-import-policy.sh"
        unresolved = ROOT / "scripts/e3-private-unresolved-policy.sh"
        script = r'''
set -euo pipefail
PROJECT_ROOT=$1
e3_tuple_kind=fault
die() { printf '%s\n' "$*" >&2; exit 77; }
source "$2"
e3_retention_verify_sanitizer_imports "$3" "$4" "$5"
source "$6"
e3_reject_unresolved_private fault "$3" "$3"
'''
        accepted = subprocess.run(
            ["bash", "-c", script, "bash", str(ROOT), str(sanitize),
             str(TESTS / "fault_package_undefined_symbols.txt"),
             str(TESTS / "fault_package_sanitized_noninstrumentation_undefined_symbols.txt"),
             str(TESTS / "fault_package_sanitized_instrumentation_undefined_symbols.txt"),
             str(unresolved)], capture_output=True, text=True)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        with tempfile.TemporaryDirectory() as directory:
            forged = Path(directory) / "forged.txt"
            forged.write_text("et_e3_private_publish_v1\n")
            denied = subprocess.run(
                ["bash", "-c", 'die() { exit 77; }; source "$1"; '
                 'e3_reject_unresolved_private fault "$2" "$2"',
                 "bash", str(unresolved), str(forged)], capture_output=True)
            self.assertEqual(denied.returncode, 77)

    def test_bound_selector_rejects_mismatch_and_phase_reordering(self):
        source = (TESTS / "fault_driver.esk").read_text()
        guard = self.bound_expression(source)
        scenario = ["fault", "execute", True, 1, 1, 1]
        bound = [1, 1, 1, True, tuple("corpus-a"), 1]
        def admitted(candidate=scenario, state=bound, path="corpus-a", phase=1):
            return self.eval_bound(guard, {
                "e3-test-fault-root": [state], "scenario": candidate,
                "corpus-path": path, "phase": phase,
            })
        self.assertTrue(admitted())
        for field, replacement in ((2, False), (3, 0), (4, 2), (5, 2)):
            changed = scenario.copy()
            changed[field] = replacement
            self.assertFalse(admitted(changed), f"field {field}")
        self.assertFalse(admitted(path="corpus-b"))
        self.assertFalse(admitted(state=None))  # execute before prepare
        self.assertFalse(admitted(phase=2))  # report before execute
        executed = bound.copy()
        executed[5] = 2
        self.assertFalse(admitted(state=executed))  # duplicate execute
        self.assertTrue(admitted(state=executed, phase=2))
        self.assertFalse(admitted(scenario[:5] + [2], state=executed, phase=2))
        self.assertFalse(admitted(state=executed, phase=2, path="corpus-b"))
        self.assertIn("(make-vector length #f)", source)
        self.assertIn("(vector-set! image index (string-ref path index))", source)
        dispatch = source[source.index("(define (e3-test-fault-dispatch"):]
        self.assertLess(dispatch.index("'bound-execute"),
                        dispatch.index("(e3-test-fault-execute"))
        self.assertLess(dispatch.index("'bound-report"),
                        dispatch.index("(e3-test-retention-report"))

    def test_complete_call_images_and_first_error_are_guarded(self):
        source = (TESTS / "fault_driver.esk").read_text()
        publication = source.split("(define (e3-test-fault-publication ", 1)[1]
        publication = publication.split("(define (e3-test-fault-probes-restored?", 1)[0]
        self.assertEqual(publication.count("(e3-test-u32-ref"), 4)
        self.assertEqual(publication.count("(e3-counter-ref"), 2)
        execute = source.split("(define (e3-test-fault-execute ", 1)[1]
        execute = execute.split("(define (e3-test-fault-dispatch", 1)[0]
        snapshots = (
            "(cursor (d2-token-dataset-cursor dataset))",
            "(modes (e3-test-retention-modes frame))",
            "(model-image (e3-test-retention-model-snapshot model))",
            "(rng-algorithm (m3t-public-initializer-algorithm initializer))",
            "(rng (m3t-public-initializer-words initializer))",
            "(publication (e3-test-fault-publication frame destinations))",
            "(before (e3-test-retention-probes))",
        )
        for snapshot in snapshots:
            self.assertIn(snapshot, execute)
            self.assertLess(execute.index(snapshot), execute.index("(e3-test-fault-arm native"))
        required = (
            "(if (= site 9) 155 (if (= site 2) 155 (if (= site 5) 107 95)))",
            "(e3-test-fault-check 'cursor\n        (d2-core-bytes=? cursor",
            "(e3-test-fault-check 'output\n        (equal? publication",
            "(e3-test-fault-check 'modes\n        (e3-test-retention-modes=?",
            "(e3-test-fault-check 'model\n        (e3-test-retention-model=?",
            "(e3-test-fault-check 'rng\n        (and (eq? rng-algorithm",
            "(e3-test-fault-check 'owners\n        (e3-test-fault-probes-restored?",
            "(e3-test-fault-check 'retry-output\n      (equal? (if (or (= site 2) (= site 3) (= site 4) (= site 5) (= site 6) (= site 7) (= site 8) (= site 9))",
            "(e3-test-fault-check 'retry-rng\n      (and (eq? rng-algorithm",
        )
        def guarded(candidate):
            return all(clause in candidate for clause in required)
        self.assertTrue(guarded(execute))
        self.assertLess(execute.index("(e3-test-fault-reset native)"),
                        execute.index("(e3-test-retention-call! model dataset frame"))
        self.assertLess(execute.index("(e3-test-retention-call! model dataset frame"),
                        execute.index("'retry-output"))
        for removal in required:
            self.assertFalse(guarded(execute.replace(removal, "#f", 1)),
                             f"omitted call assertion: {removal}")

    def test_real_stage3_value_borrow_denial_is_causal(self):
        source = (TESTS / "fault_driver.esk").read_text()
        execute = source.split("(define (e3-test-fault-execute ", 1)[1]
        execute = execute.split("(define (e3-test-fault-dispatch", 1)[0]
        bridge = (TESTS / "fault_bridge.c").read_text()
        contract = (
            "(held (if (= site 3)",
            "(e3-fault-value-borrow-begin owner role)",
            "(vector-ref before 3) (+ (vector-ref unheld 3) 1)",
            "(if (or (= site 2) (= site 3) (= site 4) (= site 5) (= site 6) (= site 7) (= site 8) (= site 9))\n        (e3-test-fault-check 'distinct-precall",
            "(equal? publication e3-test-fault-sentinels)",
            "(if (= site 3) 'invalid-state 'internal)",
            "(if (= site 9) 155 (if (= site 2) 155 (if (= site 5) 107 95)))",
            "(if (= site 2) 12 (if (= site 5) 8 7))",
            "(if (or (= site 3) (= site 4)) 3 0)",
            "(if (= site 3) 4 (if (= site 4) 6 5))",
            "(if (= site 3) 8 (if (= site 4) 6 9))",
            "(e3-test-fault-probes-restored?\n          before",
            "(e3-fault-value-borrow-end owner role)",
            "(vector-ref unheld 3)",
            "(not (eq? model (vector-ref twin 0)))",
            "(not (eq? frame (vector-ref twin 3)))",
            "(not (eq? dataset (vector-ref twin 4)))",
            "(vector-ref twin 6)",
            "(vector-ref twin 5)",
        )
        self.assertTrue(all(clause in execute for clause in contract))
        self.assertLess(execute.index("(e3-fault-value-borrow-begin owner role)"),
                        execute.index("(model-image (e3-test-retention-model-snapshot model))"))
        self.assertLess(execute.index("(e3-fault-value-borrow-end owner role)"),
                        execute.index("(e3-test-retention-call! model dataset frame"))
        self.assertLess(execute.index("(e3-test-retention-call! model dataset frame"),
                        execute.index("'independent-twin"))
        self.assertIn("(twin (if (or (= (vector-ref scenario 3) 3)", source)
        self.assertIn("(e3-test-fault-make-twin present? corpus-path)", source)
        self.assertLess(source.index("(e3-test-fault-make-twin present? corpus-path)",
                                     source.index("(define (e3-test-fault-dispatch")),
                        source.index("(e3-test-retention-prepare 1024 present? corpus-path)"))
        for clause in contract:
            mutant = execute.replace(clause, "#f", 1)
            self.assertFalse(all(item in mutant for item in contract), clause)
        native_contract = (
            "et_m3t_private_owner_parameter_v1(owner, index)",
            "et_f32_parameter_value_borrow_begin_v1(",
            "et_f32_tensor_borrow_end_v1(&e3_fault_borrow, &error)",
        )
        self.assertTrue(all(clause in bridge for clause in native_contract))
        self.assertNotIn("et_e3_fault_value_borrow_begin_v1",
                         (ROOT / "src/eshkol_transformer/e3_frame.c").read_text())

    def test_six_publication_words_use_actual_owners(self):
        source = (TESTS / "fault_driver.esk").read_text()
        expression = self.function_expression(source, "e3-test-fault-publication")
        environment = {
            "frame": [5, 3], "destinations": [11, 22, 33, 44],
            "e3-test-u32-ref": lambda value: value,
            "e3-counter-ref": lambda frame, index: frame[index],
        }
        expected = [11, 22, 33, 44, 5, 3]
        self.assertEqual(self.eval_probe(expression, environment), expected)
        for field in range(6):
            changed = environment.copy()
            changed["frame"] = environment["frame"].copy()
            changed["destinations"] = environment["destinations"].copy()
            if field < 4:
                changed["destinations"][field] += 1
            else:
                changed["frame"][field - 4] += 1
            self.assertNotEqual(self.eval_probe(expression, changed), expected,
                                f"publication word {field}")
        mutant = source.replace("(e3-counter-ref frame 1)",
                                "(e3-counter-ref frame 0)", 1)
        changed_expression = self.function_expression(
            mutant, "e3-test-fault-publication")
        self.assertNotEqual(self.eval_probe(changed_expression, environment),
                            expected)

    def test_stage12_precommit_cut_and_distinct_sentinels(self):
        native = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        driver = (TESTS / "fault_driver.esk").read_text()
        publish = native[native.index("int64_t et_e3_private_publish_v1("):]
        publish = publish[:publish.index("void et_e3_private_finish_v1(")]
        cut = publish.index("frame->fault_site == 2")
        self.assertLess(publish.index("e3_publication_counts(frame)"), cut)
        self.assertLess(publish.index("results[i].view.byte_length"), cut)
        self.assertLess(publish.index("destinations[i].view.byte_length"), cut)
        self.assertLess(cut, publish.index("memcpy(destinations[i].view.data"))
        self.assertIn("e3_scope_end(&destinations[--destination_count])",
                      publish[cut:publish.index("memcpy(destinations[i].view.data")])
        self.assertIn("et_e3_test_fault_seed_publication_v1", native)
        self.assertIn("(equal? publication e3-test-fault-sentinels)", driver)
        self.assertIn("(e3-test-fault-all-distinct?", driver)
        self.assertIn("(if (= site 9) 155 (if (= site 2) 155 (if (= site 5) 107 95)))", driver)
        self.assertIn("(equal? publication\n                (e3-test-fault-publication", driver)
        self.assertIn("(if (or (= site 2) (= site 3) (= site 4) (= site 5) (= site 6) (= site 7) (= site 8) (= site 9))\n                  e3-test-fault-final-publication publication)", driver)

        for mutated in (
            publish[:cut] + publish[cut:].replace("frame->fault_state = 2;", "", 1),
            publish[:cut] + publish[cut:].replace(
                "while (destination_count) e3_scope_end(&destinations[--destination_count]);",
                "", 1),
            publish[:cut] + publish[cut:].replace(
                "return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);", "return 0;", 1),
        ):
            self.assertNotEqual(mutated, publish)
            self.assertFalse(all(fragment in mutated[mutated.index("frame->fault_site == 2"):
                                                mutated.index("memcpy(destinations[i].view.data")]
                                 for fragment in (
                                     "frame->fault_state = 2;",
                                     "e3_scope_end(&destinations[--destination_count])",
                                     "return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);")))

    def test_stage12_each_sentinel_is_causally_distinct(self):
        source = (TESTS / "fault_driver.esk").read_text()
        expression = self.function_expression(source, "e3-test-fault-all-distinct?")
        old = [1065353216, 1073741824, 1077936128, 1082130432, 7, 4]
        final = [1085366897, 1084227584, 1132430055, 0, 5, 3]
        self.assertTrue(self.eval_probe(expression, {"left": old, "right": final}))
        for index in range(6):
            mutant = old.copy()
            mutant[index] = final[index]
            self.assertFalse(self.eval_probe(expression,
                                             {"left": mutant, "right": final}),
                             f"undistinguished publication word {index}")

    def test_probe_balance_negative_mutants(self):
        source = (TESTS / "fault_driver.esk").read_text()
        expression = self.function_expression(source, "e3-test-fault-probes-restored?")
        before = [0] * 29
        before[9], before[13], before[14], before[15], before[16] = 4, 64, 10, 10, 10
        before[27], before[28] = 2, 16
        after = before.copy()
        after[9], after[13], after[14], after[15], after[16] = 6, 96, 12, 14, 12
        after[27] = 3
        def admits(candidate=after, predicate=expression):
            return self.eval_probe(predicate, {"before": before, "after": candidate})
        self.assertTrue(admits())
        for field in (0, 20, 21, 23, 24, 26):
            leaked = after.copy()
            leaked[field] += 1
            self.assertFalse(admits(leaked), f"live owner {field}")
        for field in (9, 13, 14, 16):
            unbalanced = after.copy()
            unbalanced[field] += 1
            self.assertFalse(admits(unbalanced), f"event {field}")
        mutations = (
            ("(= (vector-ref before index) (vector-ref after index))", 0),
            ("(= retired ends)", 9),
            ("(= bytes (* ends (vector-ref before 28)))", 13),
        )
        for clause, field in mutations:
            self.assertIn(clause, source)
            mutant = self.function_expression(source.replace(clause, "#t", 1),
                                              "e3-test-fault-probes-restored?")
            changed = after.copy()
            changed[field] += 1
            self.assertTrue(admits(changed, mutant), f"mutant failed to escape {field}")

    def test_fourteen_model_images_reject_gradient_and_metadata_mutants(self):
        source = (TESTS / "retention_driver.esk").read_text()
        source = source[source.index("(define (e3-test-retention-model=? "):]
        expression = self.function_expression(source, "e3-test-retention-model=?")
        owner = [
            [[1000 + parameter, 2000 + parameter],
             [3000 + parameter, 4000 + parameter],
             [1, parameter + 1, 1065353216]]
            for parameter in range(14)
        ]
        expected = [[fields[0].copy(), fields[1].copy(), *fields[2]]
                    for fields in owner]
        def copy_owner():
            return [[part.copy() for part in parameter] for parameter in owner]
        def admits(candidate=owner, predicate=expression):
            return self.eval_probe(predicate, {
                "model": candidate, "expected": expected,
                "m3t-entry": lambda model, _kind, _operation, _live: [None, None, None, model],
                "e3-test-model-word-count":
                    lambda model, parameter, gradient: len(model[parameter][gradient]),
                "e3-test-model-word":
                    lambda model, parameter, gradient, index: model[parameter][gradient][index],
                "e3-test-gradient-metadata":
                    lambda model, parameter, field: model[parameter][2][field],
            })
        self.assertTrue(admits())
        for parameter in range(14):
            for gradient in (0, 1):
                changed = copy_owner()
                changed[parameter][gradient][1] += 1
                self.assertFalse(admits(changed),
                                 f"parameter {parameter} gradient {gradient} word")
            for field in range(3):
                changed = copy_owner()
                changed[parameter][2][field] += 1
                self.assertFalse(admits(changed),
                                 f"parameter {parameter} metadata {field}")

        changed_gradient = copy_owner()
        changed_gradient[7][1][1] += 1
        self.assertFalse(admits(changed_gradient))
        skipped_gradient = source.replace("(= gradient 2)", "(= gradient 1)", 1)
        self.assertNotEqual(skipped_gradient, source)
        gradient_mutant = self.function_expression(
            skipped_gradient, "e3-test-retention-model=?")
        self.assertTrue(admits(changed_gradient, gradient_mutant))

        changed_metadata = copy_owner()
        changed_metadata[11][2][2] += 1
        self.assertFalse(admits(changed_metadata))
        skipped_metadata = source.replace("(= field 3)", "(= field 0)", 1)
        self.assertNotEqual(skipped_metadata, source)
        metadata_mutant = self.function_expression(
            skipped_metadata, "e3-test-retention-model=?")
        self.assertTrue(admits(changed_metadata, metadata_mutant))


if __name__ == "__main__":
    unittest.main()
