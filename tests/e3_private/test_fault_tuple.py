"""Host-only lexical and source isolation checks for the noninstalled E3 fault tuple."""

from pathlib import Path
import hashlib
import re
import subprocess
import sys
import tempfile
import unittest

from tests.e3_private.generate_generation_rewind import (
    ANCHOR, MUTANT, MUTANT_SHA256, SOURCE, generate,
)

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/e3_private"
RETENTION = TESTS / "retention_package"
FAULT = TESTS / "fault_package"


def rows(prefix: Path, suffix: str) -> list[str]:
    return Path(f"{prefix}_{suffix}.txt").read_text().splitlines()


class FaultTupleTest(unittest.TestCase):
    @staticmethod
    def expression_after(source: str, marker: str):
        """Parse a source expression, so changed assertions change host controls."""
        start = source.index(marker)
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
        return FaultTupleTest.expression_after(
            source, "(define (e3-test-fault-bound?")

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
        if op == "+":
            return sum(args)
        if op == "eq?":
            return args[0] is args[1]
        if op == "e3-test-fault-path=?":
            return args[0] == tuple(args[1])
        raise AssertionError(f"unhandled guard operation: {op}")

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
                         ["tests/e3_private/fault_driver.esk"])
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
        self.assertEqual(builder.count("package_source_flags+=(-DET_E3_FAULT_TESTING)"), 1)
        seam = builder[builder.index('if [[ "${package_policy}" == e3-private-aggregate &&\n'
                                     '          "${e3_tuple_kind}" == fault &&\n'
                                     '          "${package_native_source}" == '
                                     '"${PROJECT_ROOT}/src/eshkol_transformer/e3_frame.c" ]]; then'):]
        self.assertIn("package_source_flags+=(-DET_E3_FAULT_TESTING)", seam[:350])
        source = (TESTS / "fault_driver.esk").read_text()
        signatures = (
            r'\(extern i64 e3-test-fault-arm ptr i64 i64 i64\s+:real et_e3_test_fault_arm_v1\)',
            r'\(extern i64 e3-test-fault-ref ptr i64\s+:real et_e3_test_fault_ref_v1\)',
            r'\(extern i64 e3-test-fault-reset ptr\s+:real et_e3_test_fault_reset_v1\)',
        )
        def valid(text):
            return all(len(re.findall(pattern, text)) == 1 for pattern in signatures)
        self.assertTrue(valid(source))
        self.assertFalse(valid(source.replace("arm ptr i64 i64 i64", "arm ptr i64 i64")))
        self.assertFalse(valid(source.replace("et_e3_test_fault_ref_v1",
                                              "et_e3_private_forward_role_v1")))
        self.assertIn("(e3-frame-entry-live frame 'e3-test-fault-execute)", source)
        self.assertIn("(native (vector-ref entry 4))", source)
        self.assertLess(source.index("(e3-test-fault-reset native)"),
                        source.index("(e3-test-retention-call! model dataset frame"))
        self.assertNotIn("e3-test-fault-arm", (TESTS / "retention_driver.esk").read_text())

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

    def test_stage7_rollback_preserves_monotonic_generation(self):
        source = (TESTS / "fault_driver.esk").read_text()
        marker = "(e3-test-fault-check 'generation-advanced"
        expression = self.expression_after(source, marker)
        self.assertEqual(expression,
                         ["=", ["vector-ref", "d2-state", "8"],
                          ["+", "generation-before", "occurrence"]])
        state = [0] * 19
        for prior, occurrence in ((3, 1), (1027, 3), (8192, 17)):
            for observed in (prior, prior + occurrence - 1,
                             prior + occurrence, prior + occurrence + 1):
                state[8] = observed
                self.assertEqual(self.eval_bound(expression, {
                    "d2-state": state, "generation-before": prior,
                    "occurrence": occurrence,
                }), observed == prior + occurrence)
        self.assertLess(source.index("(d2-state (dataset d2-dataset-query))"),
                        source.index("(e3-test-fault-arm native site role occurrence)"))
        self.assertLess(source.index("(e3-test-fault-check 'cursor"),
                        source.index(marker))
        self.assertLess(source.index("(e3-test-fault-check 'typed-stage7"),
                        source.index(marker))
        self.assertLess(source.index(marker),
                        source.index("(e3-test-fault-reset native)"))
        for bad in ("#t", "(= (vector-ref d2-state 8) generation-before)",
                    "(= (vector-ref d2-state 8) (+ generation-before 1))"):
            changed = source.replace(
                "(= (vector-ref d2-state 8) (+ generation-before occurrence))",
                bad, 1)
            self.assertNotEqual(self.expression_after(changed, marker), expression)

    def test_generation_witness_is_downstream_of_real_d2_publication(self):
        native_d2 = (ROOT / "native/d2_native.c").read_text()
        e3 = (ROOT / "native/e3_private_extension.esk").read_text()
        frame = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        restore = SOURCE.read_text()
        self.assertLess(native_d2.index("dataset->current_batch = batch;"),
                        native_d2.index("dataset->next_batch_generation++;"))
        self.assertLess(e3.index("(e3-stage-batch! frame batch)"),
                        e3.index("(e3-forward-schedule-internal frame)"))
        hook = frame[frame.index("int64_t et_e3_private_forward_role_v1"):
                     frame.index("et_kernel_tensor_view_v1 ids = e3_native_view(",
                                 frame.index("int64_t et_e3_private_forward_role_v1"))]
        self.assertIn("++frame->fault_reached;", hook)
        self.assertIn("frame->fault_reached == frame->fault_occurrence", hook)
        self.assertIn("/* Consume before any provider dispatch. */", hook)
        commit = restore[restore.index("(define (e3-d2-restore-commit!"):
                         restore.index("(define (e3-d2-cancel-prepare!")]
        self.assertNotIn("(vector-set! state 8", commit)

    def test_rewind_mutant_is_exact_private_source_delta(self):
        original = SOURCE.read_bytes()
        mutated = generate(original)
        self.assertEqual(mutated, original.replace(ANCHOR.encode(),
                                                   MUTANT.encode(), 1))
        self.assertEqual(mutated.count(b"(vector-set! state 8"), 1)
        self.assertEqual(hashlib.sha256(mutated).hexdigest(), MUTANT_SHA256)
        self.assertEqual(SOURCE.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "differs"):
            generate(original + b"; drift\n")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "e3_d2_restore.esk"
            command = [sys.executable, "-B",
                       str(TESTS / "generate_generation_rewind.py"),
                       "--output", str(output)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(output.read_bytes(), mutated)
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(output.read_bytes(), mutated)


if __name__ == "__main__":
    unittest.main()
