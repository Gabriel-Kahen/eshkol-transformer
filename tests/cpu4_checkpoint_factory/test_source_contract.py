"""Bounded factory/guard source checks; no compiler or native execution."""
from pathlib import Path
import unittest
from tests.cpu4_trainer_control.test_source_contract import parse, walk

ROOT = Path(__file__).resolve().parents[2]


class FactoryContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (ROOT / 'native/cpu4_checkpoint_factory_extension.esk').read_text()
        cls.checkpoint = (ROOT / 'native/cpu4_checkpoint_extension.esk').read_text()
        cls.forms = parse(cls.text)
        cls.fn = {f[1][0]: f for f in cls.forms
                  if f[0] == 'define' and isinstance(f[1], list)}
        cls.cp = {f[1][0]: f for f in parse(cls.checkpoint)
                  if f[0] == 'define' and isinstance(f[1], list)}

    def test_real_nonregional_query_and_guard_first_in_all_entries(self):
        self.assertIn('(extern ptr cpu4-checkpoint-native-current-region :real region_current)', self.checkpoint)
        entries = [(self.fn, 'cpu4-checkpoint-factory-create-internal!'),
                   (self.fn, 'cpu4-checkpoint-factory-retire-internal!'),
                   (self.cp, 'cpu4-trainer-snapshot-internal'),
                   (self.cp, 'cpu4-trainer-restore-internal!')]
        for table, name in entries:
            self.assertEqual(table[name][2][0], 'cpu4-checkpoint-require-nonregional!')
        guard = self.cp['cpu4-checkpoint-require-nonregional!']
        self.assertTrue(any(f == ['null?', ['cpu4-checkpoint-native-current-region']]
                            for f in walk(guard)))

    def test_only_genuine_split_reader_and_k2_admission(self):
        load = self.fn['cpu4-checkpoint-factory-load!']
        calls = [f[0] for f in walk(load) if f and isinstance(f[0], str)]
        ordered = ['c2-checkpoint-load-stage-internal',
                   'c2-checkpoint-load-capability-descriptors-internal',
                   'c2-public-capability-admit-all', 'c2-checkpoint-load-reconstruct-internal']
        self.assertEqual([c for c in calls if c in ordered], ordered)
        self.assertNotIn('c2-public-checkpoint-load', calls)
        self.assertNotIn('c2-load-native-', self.text)

    def test_canonical_recovery_row_and_handler_precede_acquisition(self):
        body = self.text[self.text.index('(define (cpu4-checkpoint-factory-create-internal!'):]
        self.assertLess(body.index('(guard (caught'), body.index('(vector-set! cpu4-checkpoint-factories 0 next)'))
        self.assertLess(body.index('(car (vector-ref cpu4-checkpoint-factories 0))'),
                        body.index('(cpu4-checkpoint-parent #f)'))
        self.assertLess(body.index('(vector-set! recovery-cell 0 (vector-ref canonical 2))'),
                        body.index('(cpu4-checkpoint-factory-load! canonical)'))
        self.assertIn('(vector-set! canonical 14 child)', body)
        self.assertIn('(vector-set! canonical 15 trainer)', body)

    def test_full_detach_and_owned_image_release_before_children(self):
        body = self.text[self.text.index('(define (cpu4-checkpoint-factory-create-internal!'):]
        order = ['cpu4-checkpoint-stage-source!', 'cpu4-checkpoint-factory-config-check!',
                 'cpu4-checkpoint-factory-image-release!', 'cpu4-owned-candidate-create-internal',
                 'cpu4-owned-optimizer-create-internal!', 'cpu4-owned-binding-create-internal!',
                 'cpu4-owned-trainer-create-internal!', 'cpu4-checkpoint-prepare!', 'cpu4-checkpoint-commit!']
        positions = [body.index(f'({name}') for name in order]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('(vector-ref (vector-ref detached 1) 0)', body)
        config = self.fn['cpu4-checkpoint-factory-config-check!']
        calls = [f[0] for f in walk(config) if f and isinstance(f[0], str)]
        self.assertIn('cpu4-binding-profile!', calls)
        self.assertIn('d2-core-bytes=?', calls)
        self.assertIn('x1-private-config-fingerprint', calls)
        self.assertIn('x1-private-config-ref', calls)

    def test_retry_keeps_owned_image_until_success_or_registered_dead(self):
        release = self.fn['cpu4-checkpoint-factory-image-release!']
        calls = [f[0] for f in walk(release) if f and isinstance(f[0], str)]
        self.assertEqual(calls.count('cpu4-checkpoint-factory-image-dead?'), 2)
        self.assertIn('c2-training-state-release-internal!', calls)
        cleanup = self.text[self.text.index('(define (cpu4-checkpoint-factory-cleanup!'):
                            self.text.index('(define (cpu4-checkpoint-factory-retire-internal!')]
        order = ['cpu4-checkpoint-abort!', 'cpu4-checkpoint-factory-image-release!',
                 'cpu4-checkpoint-factory-stage-release!', 'cpu4-owned-candidate-retire-internal!']
        positions = [cleanup.index(f'({name}') for name in order]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("(vector-set! row 3 'cleanup-blocked)", self.text)

    def test_no_caller_component_destroy_or_model_substitute(self):
        for forbidden in ['m3t-model', 'tr3-auth', 'd2-token-dataset-close', 'tokenizer-release',
                          'module-load-state-dict!', 'cpu4-native-candidate-', 'eval-frame',
                          'generator-create', 'callback', '(length ']:
            self.assertNotIn(forbidden, self.text)
        entry = self.fn['cpu4-checkpoint-factory-create-internal!'][1]
        self.assertEqual(entry[1:], ['path', 'policy', 'report', 'resolved', 'tokenizer', 'dataset', 'recovery-cell'])

    def test_prepared_runtime_has_real_baseline_nested_witness_and_io_counts(self):
        fixture = (ROOT / 'tests/cpu4_checkpoint_factory/nonregional_runtime.esk').read_text()
        forms = parse(fixture)
        self.assertTrue(any(f == ['null?', ['cpu4-checkpoint-native-current-region']]
                            for f in walk(forms)))
        regions = [f for f in walk(forms) if f and f[0] == 'with-region']
        self.assertEqual(len(regions), 3)
        self.assertIn('(not (eq? outer inner))', fixture)
        self.assertIn('(define (cpu4-factory-region-check-held-ticket! ticket)', fixture)
        for symbol in ['et_c2_checkpoint_reader_test_event_count_v1',
                       'et_checkpoint_io_test_event_count_v1',
                       'et_c2_checkpoint_core_test_live_images_v1',
                       'et_c2_checkpoint_reader_test_live_count_v1',
                       'et_c2_checkpoint_reader_test_fd_live_count_v1']:
            self.assertIn(':real ' + symbol, fixture)


if __name__ == '__main__':
    unittest.main()
