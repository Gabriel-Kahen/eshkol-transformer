"""Source-only contract for the post-correct, pre-accumulate test cut."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class Stage11AccumulateSourceTest(unittest.TestCase):
    def test_cut_follows_correct_and_precedes_accumulate_dispatch(self):
        frame = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        ce = frame.split("int64_t et_e3_private_ce_v1(void *pointer) {", 1)[1]
        reduce = frame.split("int64_t et_e3_private_reduce_v1(void *pointer) {", 1)[1]
        correct = frame.split("int64_t et_e3_private_correct_v1(void *pointer) {", 1)[1]
        accumulate = frame.split("int64_t et_e3_private_accumulate_v1(void *pointer) {", 1)[1]
        accumulate = accumulate.split("int64_t et_e3_private_finalize_v1", 1)[0]
        self.assertIn("if (!rc) frame->next_role = 22;", ce)
        self.assertIn("if (!rc) frame->phase = E3_REDUCED;", reduce)
        self.assertIn("if (!rc) frame->phase = E3_CORRECTED;", correct)
        phase = accumulate.index("frame->phase != E3_CORRECTED")
        guard = accumulate.index("#ifdef ET_E3_FAULT_TESTING")
        denial = accumulate.index("frame->fault_state = 2; /* Consume after correct, before accumulate dispatch. */")
        dispatch = accumulate.index("e3_accumulate_dispatch(frame)")
        self.assertLess(phase, guard)
        self.assertLess(guard, denial)
        self.assertLess(denial, dispatch)
        self.assertIn("site == 8 && ordinal == 0 && occurrence == 1", frame)
        self.assertIn("++frame->fault_reached;", accumulate[guard:denial])
        self.assertIn("return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);", accumulate[denial:dispatch])

    def test_native_witness_and_fault_driver(self):
        native = (ROOT / "tests/e3_native/test_frame.c").read_text()
        driver = (ROOT / "tests/e3_private/fault_driver.esk").read_text()
        runtime = (ROOT / "tests/e3_private/r1_stage11_runtime.esk").read_text()
        for witness in ("test_accumulate_preinvoke_selector(model, destination);",
                        "OK(et_e3_private_ce_v1(frame));",
                        "OK(et_e3_private_reduce_v1(frame));",
                        "OK(et_e3_private_correct_v1(frame));",
                        "frame->phase == E3_CORRECTED && observed_dispatches == before_ce + 3",
                        "CHECK(et_e3_private_accumulate_v1(frame) == E3_INTERNAL);",
                        "frame->fault_reached == 1 && observed_dispatches == before_ce + 3"):
            self.assertIn(witness, native)
        self.assertIn("(vector 'fault 'prepare present? 8 0 1)", runtime)
        for witness in ("((= site 8) 'typed-stage11)", "(if (= site 8) 143 95)",
                        "(if (= site 8) 11 7)", "(= (vector-ref scenario 3) 8)",
                        "(equal? publication e3-test-fault-sentinels)",
                        "(e3-test-fault-check 'cursor", "(e3-test-fault-check 'modes",
                        "(e3-test-fault-check 'model", "(e3-test-fault-check 'rng",
                        "(e3-test-fault-check 'owners",
                        "(e3-test-fault-check 'independent-twin",
                        "(e3-test-fault-check 'retry-output"):
            self.assertIn(witness, driver)

    def test_noninstalled_recipe_and_abi_boundary(self):
        build = (ROOT / "scripts/build-e3-r1-stage11-test.sh").read_text()
        gate = (ROOT / "scripts/test-e3-r1-stage11-compiled.sh").read_text()
        for witness in ("fault_driver_root.esk", "fault_package"):
            self.assertIn(witness, build)
        for witness in ("EXPECTED_HEAD EXPECTED_TREE", "verify_toolchain",
                        "r1_stage11_runtime.esk", "for mode in normal sanitized; do",
                        "for mode in normal normal-repeat poisoned sanitized; do",
                        "for gradient in absent present; do"):
            self.assertIn(witness, gate)
        self.assertNotIn("docker ", gate)
        self.assertNotIn("flock ", gate)


if __name__ == "__main__":
    unittest.main()
