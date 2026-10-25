"""Source-only contract for the post-CE, pre-L3S test cut."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class Stage9L3SSourceTest(unittest.TestCase):
    def test_cut_is_one_shot_after_ce_and_before_l3s(self):
        frame = (ROOT / 'src/eshkol_transformer/e3_frame.c').read_text()
        reduce = frame.split('int64_t et_e3_private_reduce_v1(void *pointer) {', 1)[1]
        reduce = reduce.split('int64_t et_e3_private_correct_v1', 1)[0]
        phase = reduce.index('frame->next_role != 22')
        guard = reduce.index('#ifdef ET_E3_FAULT_TESTING')
        denial = reduce.index('frame->fault_state = 2; /* Consume after CE, before L3S dispatch. */')
        dispatch = reduce.index('int64_t rc = e3_dispatch(frame, E3_RUNTIME_L3S,')
        self.assertLess(phase, guard)
        self.assertLess(guard, denial)
        self.assertLess(denial, dispatch)
        self.assertIn('site == 6 && ordinal == 0 && occurrence == 1', frame)
        self.assertIn('++frame->fault_reached;', reduce[guard:denial])
        self.assertIn('return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);', reduce[denial:dispatch])

    def test_native_witness_and_fault_driver(self):
        native = (ROOT / 'tests/e3_native/test_frame.c').read_text()
        driver = (ROOT / 'tests/e3_private/fault_driver.esk').read_text()
        runtime = (ROOT / 'tests/e3_private/r1_stage9_runtime.esk').read_text()
        self.assertIn('test_l3s_preinvoke_selector(model, destination);', native)
        for witness in ('OK(et_e3_private_ce_v1(frame));',
                        'frame->next_role == 22 && observed_dispatches == before_ce + 1',
                        'CHECK(et_e3_private_reduce_v1(frame) == E3_INTERNAL);',
                        'observed_dispatches == before_ce + 1'):
            self.assertIn(witness, native)
        self.assertIn("(vector 'fault 'prepare present? 6 0 1)", runtime)
        self.assertIn("((= site 6) 'typed-stage9)", driver)
        self.assertIn('(if (= site 6) 119 95)', driver)
        self.assertIn('(if (= site 6) 9 7)', driver)
        self.assertIn('(= (vector-ref scenario 3) 6)', driver)
        self.assertIn('(equal? publication e3-test-fault-sentinels)', driver)
        self.assertIn("(e3-test-fault-check 'independent-twin", driver)
        self.assertIn("(e3-test-fault-check 'retry-output", driver)

    def test_noninstalled_recipe_and_abi_boundary(self):
        build = (ROOT / 'scripts/build-e3-r1-stage9-test.sh').read_text()
        gate = (ROOT / 'scripts/test-e3-r1-stage9-compiled.sh').read_text()
        self.assertIn('fault_driver_root.esk', build)
        self.assertIn('fault_package', build)
        self.assertIn('EXPECTED_HEAD EXPECTED_TREE', gate)
        self.assertIn('verify_toolchain', gate)
        self.assertIn('r1_stage9_runtime.esk', gate)
        self.assertIn('for mode in normal sanitized; do', gate)
        self.assertIn('for mode in normal normal-repeat poisoned sanitized; do', gate)
        self.assertIn('for gradient in absent present; do', gate)
        self.assertNotIn('docker ', gate)
        self.assertNotIn('flock ', gate)


if __name__ == '__main__':
    unittest.main()
