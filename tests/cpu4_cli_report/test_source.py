"""Source admission checks only. No compiler/native/AOT execution."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C = (ROOT / "native/cpu4_cli_package_bridge.c").read_text()
S = (ROOT / "native/cpu4_cli_report_extension.esk").read_text()
H = (ROOT / "src/eshkol_transformer/cpu4_cli_scalar_private.h").read_text()


class SourceContract(unittest.TestCase):
    def test_exact_native_abi(self):
        names = re.findall(r"\b(et_cpu4_private_cli_scalar_\w+_v1)\s*\(", H)
        self.assertEqual(names, ["et_cpu4_private_cli_scalar_" + s + "_v1"
                                for s in ("create", "read_bits", "claim", "release", "last_error_ref")])
        for name in names:
            self.assertEqual(len(re.findall(r"\b" + name + r"\s*\(", C)), 1)
        self.assertNotRegex(H, r"(?:bits|error)\s*\*\s*(?:output|destination)")
        self.assertNotIn("ET_F32_TENSOR_API", H)

    def test_actual_rank0_scoped_provenance(self):
        for fragment in ("sizeof(cli_scalar_custody) == 32", "calloc(1, sizeof(*row))",
                         "et_f32_tensor_create_v1(0, NULL, &owner, &error)",
                         "et_f32_tensor_private_destroy_ready_v1(row->original, &error)",
                         "et_f32_tensor_scoped_begin_internal", "et_f32_tensor_scoped_end_internal",
                         "et_f32_tensor_copy_bits_to_v1(row->original, &bits, 1, &error)",
                         "et_f32_tensor *local = row->original", "row->state = 2; row->claimed = 0"):
            self.assertIn(fragment, C)
        self.assertNotRegex(C, r"struct\s+et_f32_tensor\s*\{")
        self.assertNotRegex(C, r"\b(?:float|double)\b")
        self.assertNotIn("TESTING", C)

    def test_tls_fallback_and_reset(self):
        self.assertIn("static _Thread_local int64_t cli_error[3]", C)
        self.assertIn("error->category == (uint32_t)status", C)
        self.assertIn("error->code >= 1 && error->code <= 11", C)
        self.assertIn("return cli_fail(6)", C)
        self.assertEqual(C.count("  cli_reset();"), 4)
        self.assertIn("memset(&error, 0, sizeof(error))", C)
        self.assertIn("et_f32_tensor_error error = {0}", C)

    def test_closed_interfaces_and_single_actual_producer(self):
        names = re.findall(r"\(define \((cpu4-cli-(?:scalar-owner-create-internal|scalar-readbits-internal|scalar-owner-release-internal!|report-bind-internal|controls-report-copy-internal|report-step-internal!|final-step-report-copy-internal|evaluation-report-internal|report-release-internal!))\b", S)
        # '!' is followed by whitespace, so enumerate exact definitions directly.
        expected = ("scalar-owner-create-internal", "scalar-readbits-internal",
                    "scalar-owner-release-internal!", "report-bind-internal",
                    "controls-report-copy-internal", "report-step-internal!",
                    "final-step-report-copy-internal", "evaluation-report-internal", "report-release-internal!")
        for name in expected:
            self.assertEqual(S.count("(define (cpu4-cli-" + name), 1)
        self.assertEqual(S.count("(cpu4-cli-producer-step (vector-ref row 4))"), 1)
        self.assertIn("(define cpu4-cli-producer-step cpu4-training-step-internal!)", S)
        self.assertNotIn("(provide", S)
        self.assertNotIn("TESTING", S)
        self.assertNotIn("StageA", S)

    def test_guard_precedes_busy_and_commit_retention(self):
        macro = S[S.index("(define-syntax cpu4-cli-report-operation"):S.index("(define (cpu4-cli-report-pending?")]
        self.assertLess(macro.index("(guard"), macro.index("(cpu4-cli-report-enter! row)"))
        self.assertIn("(if entered", macro)
        self.assertEqual(S.count("(cpu4-cli-report-operation row"), 5)
        step = S[S.index("(define (cpu4-cli-report-step-internal!"):S.index("(define (cpu4-cli-u32-le")]
        self.assertLess(step.index("(cpu4-cli-controls-snapshot! row)"), step.index("(cpu4-cli-producer-step"))
        self.assertIn("(>= (+ (vector-ref row 15) (vector-ref row 16)) 250)", step)
        tail = step[step.index("(let ((result (cpu4-cli-producer-step"):]
        self.assertNotIn("abort", tail)
        self.assertNotIn("make-vector", tail)
        self.assertIn("(vector-set! row 17 result)", tail)
        self.assertTrue(tail.rstrip().endswith("#t))))"))

    def test_observation_offsets_and_retry_identity(self):
        for label, offset in (("numerator", 0), ("weight", 4), ("mean", 8)):
            self.assertIn(f"({label} (cpu4-cli-u32-le observation {offset}))", S)
        self.assertIn("cpu4-cli-final-step-report-v1", S)
        self.assertIn("cpu4-cli-evaluation-report-v1", S)
        self.assertIn("evaluation retry changed heldout identity", S)
        self.assertIn("(vector-ref (vector-ref row 24) 2)", S)
        self.assertIn("(e3-evaluate-into! (vector-ref row 8) heldout-dataset (vector-ref row 20))", S)
        self.assertLess(S.index("(e3-frame-unbind! frame)"), S.index("(cpu4-cli-scalar-claim! scalar 2)"))

    def test_preservation_reuses_actual_snapshot_contract(self):
        original = (ROOT / "tests/cpu4_resume/body.esk").read_text()
        start = original.index(";;; Fixed-array capture")
        end = original.index("(define (cpu4-resume-counts!", start)
        reuse = (ROOT / "tests/cpu4_cli_report/preservation.esk").read_text()
        self.assertTrue(reuse.endswith(original[start:end]))
        runtime = (ROOT / "tests/cpu4_cli_report/runtime.esk").read_text()
        self.assertEqual(runtime.count('(load "cpu4_v1_private_root.esk")'), 1)
        self.assertEqual(runtime.count("(cpu4-cli-report-step-internal! context)"), 2)
        self.assertIn("(cpu4-resume-copy=? before after)", runtime)
        self.assertIn("cpu4-cli-test-gradients-absent?", runtime)


if __name__ == "__main__":
    unittest.main()
