"""Source/host checks only; genuine Eshkol/native admission is separate."""
from pathlib import Path
import unittest
from tests.cpu4_trainer_control.test_source_contract import parse, walk

ROOT = Path(__file__).resolve().parents[2]


class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (ROOT / 'native/cpu4_checkpoint_extension.esk').read_text()
        cls.forms = parse(cls.text)
        cls.fn = {x[1][0]: x for x in cls.forms
                  if len(x) > 2 and x[0] == 'define' and isinstance(x[1], list)}

    def test_single_identity_root_and_no_old_authority(self):
        forms = parse((ROOT / 'native/cpu4_checkpoint_root.esk').read_text())
        loads = [x[1] for x in forms if x[0] == 'load']
        self.assertEqual(loads.count('d2_wave2_root.esk'), 1)
        self.assertFalse(any('lease_core' in x or 'joint_restore' in x or
                             'snapshot_root' in x or 'm3t_package_root' in x for x in loads))
        self.assertLess(loads.index('cpu4_trainer_control_extension.esk'),
                        loads.index('cpu4_checkpoint_extension.esk'))
        self.assertNotIn('tr3-lease-', self.text)
        self.assertNotIn('m3t-model', self.text)

    def test_actual_c4_schedule_and_serialized_controls(self):
        self.assertIn('(cpu4-checkpoint-vector->list g3c4-model-paths 14)', self.text)
        self.assertIn('(cpu4-checkpoint-vector->list g3c4-model-shapes 14)', self.text)
        self.assertNotIn('(2 4)', self.text)
        body = self.fn['cpu4-checkpoint-controls-copy']
        self.assertTrue(any(x == ['vector', ['vector-ref', ['vector-ref', 'copied', '3'], '0'],
                                 ['vector-ref', ['vector-ref', 'copied', '3'], '1']]
                            for x in walk(body)))

    def test_no_stdlib_length_or_reserved_named_loop(self):
        for form in self.forms:
            for node in walk(form):
                self.assertFalse(node and node[0] == 'length')
                self.assertFalse(len(node) > 1 and node[:2] == ['let', 'match'])

    def test_prepare_and_commit_order(self):
        prep = self.text[self.text.index('(define (cpu4-checkpoint-prepare!'):
                         self.text.index('(define (cpu4-checkpoint-abort!')]
        order = ['cpu4-trainer-controls-restore-stage-internal', 'tr3-d2-restore-prepare-internal',
                 'tr3-c-restore-o2-prepare-internal', 'tr3-c-restore-i2-create-internal',
                 'cpu4-p1-fixed-recheck-internal', 'cpu4-trainer-controls-restore-precommit-internal',
                 'tr3-d2-restore-check-internal', 'tr3-c-restore-o2-check-internal',
                 'tr3-c-restore-i2-prepare-internal!']
        positions = [prep.index(x) for x in order]
        self.assertEqual(positions, sorted(positions))
        commit = self.fn['cpu4-checkpoint-commit!']
        calls = [x[0] for x in walk(commit) if x and isinstance(x[0], str)]
        order = ['tr3-c-restore-i2-commit-tail-internal!', 'tr3-c-restore-o2-commit-tail-internal!',
                 'tr3-c-restore-o2-finalize-tail-internal!', 'tr3-d2-restore-commit-internal!',
                 'cpu4-trainer-controls-restore-commit-internal!', 'cpu4-trainer-control-end-internal!']
        self.assertEqual([x for x in calls if x in order], order)
        self.assertEqual(calls.count('tr3-c-restore-owned-release-tail-internal!'), 14)
        self.assertFalse(any(x in calls for x in ['make-vector', 'vector', 'cons', 'list']))

    def test_all_later_metadata_copies_precede_borrow_end(self):
        stage = self.text[self.text.index('(define (cpu4-checkpoint-stage-source!'):
                          self.text.index('(define (cpu4-checkpoint-release-cell!')]
        self.assertEqual(stage.count('o2-copy-config-data'), 2)
        self.assertEqual(stage.count('o2-copy-paths'), 2)
        self.assertEqual(stage.count('o2-copy-aliases'), 2)
        self.assertLess(stage.rindex('o2-copy-'), stage.index('c2-training-state-borrow-end-internal'))
        self.assertNotIn('vector-ref projection', stage[stage.index('c2-training-state-borrow-end-internal'):])
        prep = self.text[self.text.index('(define (cpu4-checkpoint-prepare!'):
                         self.text.index('(define (cpu4-checkpoint-abort!')]
        busy = prep.index('tr3-d2-restore-prepare-internal')
        for name in ['cpu4-trainer-controls-restore-copy-internal',
                     'cpu4-control-stage-internal', 'cpu4-control-parent-internal']:
            self.assertLess(prep.index(name), busy)

    def test_closed_custody_and_blocked_retry_are_explicit(self):
        abort = self.fn['cpu4-checkpoint-abort!']
        symbols = [x for node in walk(abort) for x in node if isinstance(x, str)]
        self.assertIn('aborted', symbols)
        self.assertIn('closed', symbols)
        self.assertIn('cpu4-checkpoint-drain-pending!', self.fn)
        self.assertIn('cpu4-snapshot-drain-pending!', self.fn)
        restore = self.fn['cpu4-trainer-restore-internal!']
        self.assertEqual(restore[2], ['cpu4-checkpoint-drain-pending!', 'trainer'])
        self.assertEqual(restore[3], ['cpu4-snapshot-drain-pending!', 'trainer'])

    def test_existing_trainer_only(self):
        self.assertNotIn('cpu4-candidate-model-create', self.text)
        self.assertNotIn('cpu4-model-binding-create', self.text)
        self.assertNotIn('o2-public-optimizer-create', self.text)
        self.assertNotIn('cpu4-trainer-create-internal', self.text)
        self.assertNotIn('cpu4-candidate-retire', self.text)


if __name__ == '__main__':
    unittest.main()
