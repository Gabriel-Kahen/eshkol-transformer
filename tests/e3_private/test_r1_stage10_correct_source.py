"""Source-only contract for the post-L3S, pre-correct test cut."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class Stage10CorrectSourceTest(unittest.TestCase):
    def test_cut_follows_prior_providers_and_precedes_correct_dispatch(self):
        frame = (ROOT / 'src/eshkol_transformer/e3_frame.c').read_text()
        correct = frame.split('int64_t et_e3_private_correct_v1(void *pointer) {', 1)[1]
        correct = correct.split('static int64_t e3_accumulate_dispatch', 1)[0]
        phase = correct.index('frame->phase != E3_REDUCED')
        guard = correct.index('#ifdef ET_E3_FAULT_TESTING')
        denial = correct.index('frame->fault_state = 2; /* Consume after L3S, before correct dispatch. */')
        dispatch = correct.index('et_kernel_runtime_dispatch(frame->runtime[E3_RUNTIME_METRICS]')
        self.assertLess(phase, guard)
        self.assertLess(guard, denial)
        self.assertLess(denial, correct.index('e3_scope_begin(float_output[0]'))
        self.assertLess(denial, dispatch)
        self.assertIn('site == 7 && ordinal == 0 && occurrence == 1', frame)
        self.assertIn('++frame->fault_reached;', correct[guard:denial])
        self.assertIn('return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);', correct[denial:dispatch])

    def test_native_witness_and_fault_driver(self):
        native = (ROOT / 'tests/e3_native/test_frame.c').read_text()
        driver = (ROOT / 'tests/e3_private/fault_driver.esk').read_text()
        runtime = (ROOT / 'tests/e3_private/r1_stage10_runtime.esk').read_text()
        self.assertIn('test_correct_preinvoke_selector(model, destination);', native)
        for witness in ('OK(et_e3_private_ce_v1(frame));',
                        'OK(et_e3_private_reduce_v1(frame));',
                        'frame->phase == E3_REDUCED && observed_dispatches == before_ce + 2',
                        'CHECK(et_e3_private_correct_v1(frame) == E3_INTERNAL);'):
            self.assertIn(witness, native)
        self.assertIn("(vector 'fault 'prepare present? 7 0 1)", runtime)
        self.assertIn("((= site 7) 'typed-stage10)", driver)
        self.assertIn('(if (= site 7) 131 95)', driver)
        self.assertIn('(if (= site 7) 10 7)', driver)
        self.assertIn('(= (vector-ref scenario 3) 7)', driver)
        self.assertIn('(equal? publication e3-test-fault-sentinels)', driver)
        self.assertIn("(e3-test-fault-check 'independent-twin", driver)
        self.assertIn("(e3-test-fault-check 'retry-output", driver)

    def test_noninstalled_recipe_and_abi_boundary(self):
        build = (ROOT / 'scripts/build-e3-r1-stage10-test.sh').read_text()
        gate = (ROOT / 'scripts/test-e3-r1-stage10-compiled.sh').read_text()
        self.assertIn('fault_driver_root.esk', build)
        self.assertIn('fault_package', build)
        self.assertIn('EXPECTED_HEAD EXPECTED_TREE', gate)
        self.assertIn('verify_toolchain', gate)
        self.assertIn('r1_stage10_runtime.esk', gate)
        self.assertIn('for mode in normal sanitized; do', gate)
        self.assertIn('for mode in normal normal-repeat poisoned sanitized; do', gate)
        self.assertIn('for gradient in absent present; do', gate)
        self.assertNotIn('docker ', gate)
        self.assertNotIn('flock ', gate)


if __name__ == '__main__':
    unittest.main()
