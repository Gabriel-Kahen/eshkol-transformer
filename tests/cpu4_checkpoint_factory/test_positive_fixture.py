"""Source checks for the prepared actual FILE data-movement fragment; no runtime."""
from pathlib import Path
import unittest
from tests.cpu4_trainer_control.test_source_contract import parse, walk

ROOT = Path(__file__).resolve().parents[2]


class PositiveFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (ROOT / 'tests/cpu4_checkpoint_factory/positive_runtime.esk').read_text()
        cls.forms = parse(cls.text)
        cls.fn = {f[1][0]: f for f in cls.forms
                  if f[0] == 'define' and isinstance(f[1], list)}

    def test_one_owning_root_and_actual_reader_writer(self):
        self.assertFalse(any(f[0] in ('load', 'require', 'import') for f in self.forms))
        body = self.fn['cpu4-factory-data-runtime!']
        calls = [f[0] for f in walk(body) if f and isinstance(f[0], str)]
        for name in ['k2-public-capability-discover', 'cpu4-trainer-restore-internal!',
                     'cpu4-fdt-snapshot', 'c2-public-checkpoint-save!', 'cpu4-fdt-create']:
            self.assertIn(name, calls)
        self.assertIn('cpu4-checkpoint-factory-create-internal!', self.text)
        for forbidden in ['optimizer-step!', 'module-forward', 'eval-frame', 'generator-create',
                          'tr3-trainer', 'm3t-model', '(length ', '(let match ']:
            self.assertNotIn(forbidden, self.text)

    def test_distinct_finite_payloads_and_real_owned_seed_adoption(self):
        self.assertIn('(* plane 4096)', self.text)
        self.assertIn('(modulo index 4096)', self.text)
        self.assertIn('(if (= index 14) 10 index)', self.text)
        self.assertIn('(state-dict-adopt-owned-internal!', self.text)
        self.assertIn('(i2-checkpoint-decode request)', self.text)
        self.assertIn('(o2-state-reconstruct-internal request)', self.text)
        self.assertIn('(make-vector 14 #f)', self.text)
        self.assertIn('(make-vector 28 #f)', self.text)
        self.assertIn("'(linear 2 10 \"3e800000\")", self.text)

    def test_initial_cursor_restored_and_all_images_closed_before_source_retire(self):
        body = self.text[self.text.index('(define (cpu4-factory-data-runtime!'):]
        self.assertLess(body.index('(initial-controls (cpu4-fdt-controls trainer))'),
                        body.index('(batch (d2-token-dataset-next-batch dataset))'))
        self.assertLess(body.index('(cpu4-d2-cursor-restore-internal! dataset initial-cursor)'),
                        body.index('(cpu4-trainer-restore-internal! trainer seed)'))
        self.assertLess(body.index('(c2-training-state-release-internal! seed'),
                        body.index('(cpu4-owned-candidate-retire-internal! source)'))
        self.assertLess(body.index('(c2-training-state-release-internal! snapshot'),
                        body.index('(cpu4-owned-candidate-retire-internal! source)'))
        helper = self.fn['cpu4-fdt-capture-trainer']
        self.assertTrue(any(f and f[0] == 'c2-training-state-release-internal!' for f in walk(helper)))

    def test_all_planes_controls_cursor_native_schedule_and_regional_held_ticket(self):
        for literal in ['(vector-ref expected 0) (vector-ref actual 0) 14',
                        '(vector-ref expected 1) (vector-ref actual 1) 28',
                        '(cpu4-fdt-controls=?', 'o2-optimizer-completed-updates-internal',
                        'o2-optimizer-schedule-factor-bits-internal',
                        '(cpu4-factory-region-check-held-ticket! ticket)']:
            self.assertIn(literal, self.text)
        capture = self.text[self.text.index('(define (cpu4-fdt-capture image)'):
                            self.text.index('(define (cpu4-fdt-snapshot trainer)')]
        self.assertLess(capture.index('(controls (cpu4-checkpoint-controls-copy'),
                        capture.index('(c2-training-state-borrow-end-internal borrow)'))
        self.assertIn('(o2-copy-paths (vector-ref source-o2 1))', capture)
        self.assertIn('(o2-copy-config-data operation (vector-ref source-o2 0))', capture)

    def test_partial_cuts_call_real_creators_and_restore_only_test_entry_names(self):
        for name, original in [('cpu4-fdt-candidate', 'cpu4-fdt-original-candidate'),
                               ('cpu4-fdt-binding', 'cpu4-fdt-original-binding'),
                               ('cpu4-fdt-trainer', 'cpu4-fdt-original-trainer')]:
            self.assertTrue(any(f and f[0] == original for f in walk(self.fn[name])))
        sets = {f[1] for f in walk(self.forms) if len(f) > 2 and f[0] == 'set!'}
        self.assertEqual(sets - {'cpu4-fdt-checks', 'cpu4-fdt-before-child', 'cpu4-fdt-cut', 'cpu4-fdt-observed-cut'},
                         {'cpu4-owned-candidate-create-internal', 'cpu4-owned-binding-create-internal!',
                          'cpu4-owned-trainer-create-internal!'})
        self.assertIn("'(optimizer binding trainer)", self.text)
        self.assertIn('(cpu4-fdt-o2-fail 0)', self.text)
        self.assertIn('(cpu4-fdt-restore-entries!) (raise caught)', self.text)
        self.assertIn('(eq? cpu4-fdt-observed-cut (car remaining))', self.text)
        for phase in ['optimizer', 'binding', 'trainer']:
            self.assertIn("(set! cpu4-fdt-observed-cut '" + phase + ')', self.text)

    def test_external_c2_checkpoint_is_never_factory_owned_and_retry_is_explicit(self):
        body = self.text[self.text.index('(let* ((external (cpu4-fdt-snapshot restored))'):]
        self.assertLess(body.index('(cpu4-checkpoint-factory-retire-internal! ticket)'),
                        body.index('(c2-training-state-release-internal! external'))
        self.assertIn('(eq? child (vector-ref row 14))', body)
        self.assertIn('(eq? restored (vector-ref row 15))', body)
        self.assertIn('(cpu4-fdt-capture external)', body)
        self.assertIn('(cpu4-d2-cursor-capture-internal dataset)', body)


if __name__ == '__main__':
    unittest.main()
