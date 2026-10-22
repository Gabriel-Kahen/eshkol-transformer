"""Source guards for the closed typed handoff; supported execution is separate."""
from pathlib import Path
import unittest
from tests.cpu4_trainer_control.test_source_contract import parse, walk

ROOT = Path(__file__).resolve().parents[2]

class HandoffContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (ROOT / 'native/cpu4_generator_handoff_extension.esk').read_text()
        cls.base = (ROOT / 'native/cpu4_candidate_extension.esk').read_text()
        cls.forms = parse(cls.text)
        cls.fn = {f[1][0]: f for f in cls.forms if f[0] == 'define' and isinstance(f[1], list)}
        cls.base_forms = parse(cls.base)
        cls.base_functions = {f[1][0]: f for f in walk(cls.base_forms)
                              if len(f) > 2 and f[0] == 'define' and isinstance(f[1], list)}

    def test_first_nonregional_guards_and_exact_separate_tokenizer_signatures(self):
        names = ['cpu4-owned-generator-handoff-internal!', 'cpu4-generator-handoff-retry-internal!',
                 'cpu4-checkpoint-factory-generator-handoff-internal!']
        for name in names:
            self.assertEqual(self.fn[name][2][0], 'cpu4-checkpoint-require-nonregional!')
        for name in [names[0], names[2]]:
            self.assertEqual(self.fn[name][1][-3:], ['generation-tokenizer', 'config', 'recovery-cell'])
        self.assertIn('(g3c4-baseline-tokenizer-entry generation-tokenizer', self.text)
        self.assertIn('(vector-ref trainer-row 1)', self.text)

    def test_exact_i32_native_commit_only_no_native_or_shared_source_edits(self):
        self.assertIn('(extern i32 cpu4-handoff-native-o2-commit ptr ptr', self.text)
        self.assertIn(':real et_o2_private_candidate_retire_commit_v1', self.text)
        for forbidden in ['cpu4-native-candidate-retire-', 'module-load-state-dict!', 'm3t-model',
                          'tr3-auth', 'generator-generate', 'd2-token-dataset-close!', '(length ']:
            self.assertNotIn(forbidden, self.text)
        for f in walk(self.forms):
            if len(f) > 2 and f[0] == 'let' and isinstance(f[1], str):
                self.assertNotEqual(f[1], 'match')

    def test_plan_before_detach_and_fresh_plan_each_detached_attempt(self):
        resume = self.text[self.text.index('(define (cpu4-handoff-resume!'):
                           self.text.index('(define (cpu4-generator-handoff-retry-internal!')]
        self.assertLess(resume.index('(cpu4-owned-o2-unlink-plan'), resume.index('(cpu4-trainer-detach-internal!'))
        self.assertEqual(resume.count('(cpu4-owned-o2-unlink-plan'), 2)
        detached = resume[resume.index("(if (eq? (vector-ref row 3) 'detached)"):]
        self.assertLess(detached.index('(cpu4-owned-o2-unlink-plan'), detached.index('(cpu4-handoff-native-o2-commit'))
        self.assertIn('(set-cdr! (vector-ref plan 0) (vector-ref plan 2))', detached)
        self.assertIn('(vector-set! child 11 #t)', resume)
        for slot in range(6):
            self.assertIn(f'(vector-set! (vector-ref child 7) {slot} #f)', detached)
        self.assertLess(detached.index('(vector-set! child 8 #f)'), detached.index('(module-eval!'))

    def test_real_public_membership_immediately_precedes_typed_base(self):
        resume = self.fn['cpu4-handoff-resume!']
        auth = [f for f in walk(resume) if f and f[0] == 'let*'
                and isinstance(f[1], list) and f[1] and f[1][0][0] == 'profile'][0]
        self.assertEqual(auth[1][0][1][0], 'g3c4-public-g2-profile-live')
        self.assertEqual(auth[1][1][1][0], 'g3c4-generator-entry-call-ready')
        self.assertTrue(any(f == ['eq?', ['vector-ref', 'profile', '1'], 'entry'] for f in walk(auth)))
        for index, row_slot in [('5', '7'), ('6', '8')]:
            self.assertTrue(any(f == ['eq?', ['vector-ref', 'entry', index], ['vector-ref', 'row', row_slot]] for f in walk(auth)))
        self.assertEqual(auth[3][0], 'cpu4-candidate-adopt-public-g2-entry-internal!')
        for f in walk(auth[4:]):
            if f and isinstance(f[0], str):
                self.assertIn(f[0], {'vector-set!', 'vector-ref', 'quote'})

    def test_base_keeps_original_api_and_has_no_public_g2_dependency(self):
        original = self.base_functions['adopt']
        self.assertTrue(any(f and f[0] == 'if' and f[1] == 'generator' for f in walk(original)))
        typed = self.base_functions['adopt-public-g2-entry']
        self.assertTrue(any(f == ['=', ['vector-length', 'entry'], '12'] for f in walk(typed)))
        self.assertTrue(any(f == ['eq?', ['vector-ref', 'entry', '5'], 'model-entry'] for f in walk(typed)))
        self.assertTrue(any(f == ['native-adopt', ['vector-ref', 'row', '2'], ['vector-ref', 'entry', '3']] for f in walk(typed)))
        self.assertNotIn('g3c4-public-g2-profile-live', self.base)
        self.assertNotIn('g3c4-generator-entry-call-ready', self.base)
        self.assertIn('(vector create model adopt retire retire-owned-optimizer adopt-public-g2-entry)', self.base)
        self.assertEqual(typed[2][-2], ['vector-set!', 'row', '1', ['quote', 'adopted']])
        self.assertEqual(typed[2][-1], 'result')

    def test_recovery_and_config_words_precede_effects(self):
        self.assertIn('(g3c4-public-profile-config config operation)', self.text)
        self.assertIn('(g3c4-normalize-config (vector-ref parsed 0) operation)', self.text)
        self.assertIn('(g3c4-public-result-rng-entry rng-shell operation)', self.text)
        self.assertIn('(g3c4-rng-capture (vector-ref source 3) operation)', self.text)
        entry = self.text[self.text.index('(define (cpu4-owned-generator-handoff-internal!'):]
        self.assertLess(entry.index('(accepted (cpu4-handoff-config'), entry.index('(vector-set! child 12 #t)'))
        self.assertLess(entry.index('(result (vector model #f))'), entry.index('(vector-set! child 12 #t)'))
        self.assertLess(entry.index('(car (vector-ref cpu4-handoffs 0))'), entry.index('(vector-set! recovery-cell 0'))
        self.assertLess(entry.index('(vector-set! recovery-cell 0'), entry.index('(vector-set! child 12 #t)'))
        retry = self.fn['cpu4-generator-handoff-retry-internal!']
        for f in walk(retry):
            if f and f[0] == 'vector':
                self.fail('retry allocates a new result/error envelope')

    def test_prepared_runtime_checks_actual_region_read_and_same_values(self):
        text = (ROOT / 'tests/cpu4_generator_handoff/runtime_fragment.esk').read_text()
        forms = parse(text)
        self.assertFalse(any(f[0] in ('load', 'require', 'import') for f in forms))
        for required in ['(null? (cpu4-checkpoint-native-current-region))', 'with-region',
                         '(cpu4-control-native-enter lease 1)', '(cpu4-control-native-leave lease)',
                         'cpu4-handoff-runtime-values=?', '(i2-checkpoint-encode',
                         'cpu4-checkpoint-factory-retire-internal!', 'g3c4-public-g2-profile-live']:
            self.assertIn(required, text)
        self.assertNotIn('generator-generate', text)
        self.assertNotIn('g3c4-model-create', text)

if __name__ == '__main__':
    unittest.main()
