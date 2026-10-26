"""Source-only contract for the authentic role-1 preinvoke cut."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class Stage7Role1SourceTest(unittest.TestCase):
    def test_sanitized_fault_package_uses_fault_selector(self):
        wrapper = (ROOT / 'scripts/build-e3-r1-stage7-test.sh').read_text()
        policy = (ROOT / 'scripts/build-e1b-consumer.sh').read_text()
        self.assertIn('E3_FAULT_SANITIZE="$sanitize"', wrapper)
        self.assertNotIn('E3_RETENTION_SANITIZE=', wrapper)
        self.assertIn('"${e3_tuple_kind}" == fault', policy)
        self.assertIn('"${E3_FAULT_SANITIZE}" == 1', policy)

    def test_native_cut_follows_completed_role_zero(self):
        frame = (ROOT / 'src/eshkol_transformer/e3_frame.c').read_text()
        native = (ROOT / 'tests/e3_native/test_frame.c').read_text()
        role = frame.split('int64_t et_e3_private_forward_role_v1(', 1)[1]
        role = role.split('int64_t et_e3_private_ce_v1(', 1)[0]
        self.assertLess(role.index('frame->next_role != (uint32_t)role'),
                        role.index('frame->fault_site == 1'))
        self.assertLess(role.index('frame->fault_state = 2; /* Consume before any provider dispatch. */'),
                        role.index('int64_t rc = e3_dispatch(frame, runtime,'))
        for witness in ('OK(et_e3_private_forward_role_v1(frame, 0));',
                        'frame->next_role == 1 && observed_dispatches == before_role_zero + 1',
                        'CHECK(et_e3_private_forward_role_v1(frame, 1) == E3_INTERNAL);',
                        'observed_dispatches == before_role_zero + 1'):
            self.assertIn(witness, native)

    def test_driver_seeds_six_words_and_checks_first_error_retry(self):
        driver = (ROOT / 'tests/e3_private/fault_driver.esk').read_text()
        execute = driver.split('(define (e3-test-fault-execute ', 1)[1]
        execute = execute.split('(define (e3-test-fault-dispatch', 1)[0]
        runtime = (ROOT / 'tests/e3_private/r1_stage7_runtime.esk').read_text()
        self.assertIn("(vector 'fault 'prepare present? 1 1 1)", runtime)
        self.assertEqual(driver.count('(if (or (= (vector-ref scenario 3) 1)'), 2)
        self.assertEqual(driver.count('(if (or (= site 1)'), 3)
        self.assertIn('(e3-test-fault-seed-publication native)', driver)
        self.assertIn('(equal? publication e3-test-fault-sentinels)', execute)
        self.assertIn('(e3-test-fault-all-distinct?', execute)
        self.assertIn('(if (or (= site 1) (= site 3)', execute)
        self.assertIn('(vector-ref twin 6)', execute)
        self.assertIn('(if (= site 2) 155 (if (= site 5) 107 95))', execute)
        self.assertIn('(if (= site 2) 12 (if (= site 5) 8 7))', execute)
        self.assertIn('(if (= site 3) 4 (if (= site 4) 6 5))', execute)
        self.assertLess(execute.index('(e3-test-fault-reset native)'),
                        execute.index('(e3-test-retention-call! model dataset frame'))
        self.assertIn('E3-R1-STAGE7-PASS', runtime)

    def test_compiled_recipe_remains_exact_fault_tuple_and_unrun(self):
        build = (ROOT / 'scripts/build-e3-r1-stage7-test.sh').read_text()
        gate = (ROOT / 'scripts/test-e3-r1-stage7-compiled.sh').read_text()
        self.assertIn('tests/e3_private/fault_driver_root.esk', build)
        self.assertIn('tests/e3_private/fault_bridge.c', build)
        self.assertIn('fault_package', build)
        self.assertIn('EXPECTED_HEAD EXPECTED_TREE', gate)
        self.assertIn('verify_toolchain', gate)
        self.assertIn('for mode in normal sanitized; do', gate)
        self.assertIn('for mode in normal normal-repeat poisoned sanitized; do', gate)
        self.assertIn('for gradient in absent present; do', gate)
        self.assertIn('cmp "$out/runtime-normal-$gradient.stdout"', gate)
        self.assertIn('r1_stage7_runtime.esk', gate)
        self.assertNotIn('docker ', gate)
        self.assertNotIn('flock ', gate)


if __name__ == '__main__':
    unittest.main()
