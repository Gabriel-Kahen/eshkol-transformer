"""Cheap body/inspector dependency guards; no native or Eshkol execution."""
import hashlib
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/cpu4_trainer_control'))
from test_source_contract import parse, walk

BODY = ROOT / 'tests/cpu4_heldout/body.esk'
NATIVE = ROOT / 'tests/cpu4_heldout/inspect.c'


class HeldoutSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = BODY.read_text()
        cls.native = NATIVE.read_text()
        cls.forms = parse(cls.text)
        cls.functions = {form[1][0]: form for form in cls.forms
                         if form[0] == 'define' and isinstance(form[1], list)}
        cls.nodes = [node for form in cls.forms for node in walk(form)]

    def test_no_aggregate_owner_training_or_new_e3_api(self):
        self.assertFalse(any(form[0] == 'load' for form in self.forms))
        heads = {node[0] for node in self.nodes if node and isinstance(node[0], str)}
        e3 = {name for name in heads if name.startswith('e3-') or name.startswith('cpu4-e3-')}
        self.assertEqual(e3, {'cpu4-e3-frame-bind-internal', 'e3-evaluate-into!',
                              'e3-counter-ref', 'e3-frame-unbind!'})
        source = subprocess.check_output(['git', 'show',
            '4c745a0f:native/e3_cpu4_private_extension.esk'], cwd=ROOT, text=True)
        for name in e3:
            self.assertIn('(define (' + name, source)
        for prefix in ('g3c4-model-create', 'cpu4-trainer-create', 'o2-public-optimizer-create',
                       'd2-token-dataset-open', 'cpu4-training-step', 'tr3-lease-'):
            self.assertFalse(any(name.startswith(prefix) for name in heads))
        self.assertNotIn('length', heads)
        self.assertNotIn('(load ', self.text)

    def test_externs_are_direct_and_fixture_owned_only(self):
        externs = {form[2] for form in self.forms if form[0] == 'extern'}
        self.assertEqual(externs, {'heldout-image-create', 'heldout-image-equal',
            'heldout-image-destroy', 'heldout-output-create', 'heldout-output-bits',
            'heldout-output-destroy', 'heldout-live-owned'})
        for form in self.forms:
            if form[0] != 'extern':
                for node in walk(form):
                    for index, item in enumerate(node):
                        if isinstance(item, str) and item in externs:
                            self.assertEqual(index, 0, 'captured bare extern')

    def test_exact_same_owner_sequence_and_counts(self):
        identity = list(walk(self.functions['heldout-phase-identity!']))
        for index, name in enumerate(('trainer', 'model', 'binding', 'dataset')):
            self.assertIn(['eq?', name, ['vector-ref', 'previous', str(index)]], identity)
        self.assertIn(["quote", 'new'], identity)
        self.assertIn(["quote", 'baseline'], identity)
        counts = list(walk(self.functions['heldout-counts!']))
        self.assertIn(['vector', '0', '0', '0'], counts)
        self.assertIn(['vector', '1000', '250', '249'], counts)
        self.assertIn(['not', ['eq?', 'dataset', ['vector-ref', 'record', '15']]], self.nodes)
        self.assertIn(['=', ['vector-ref', 'core', '14'], '1'], self.nodes)
        self.assertIn(['=', ['vector-ref', 'core', '15'], '0'], self.nodes)

    def test_full_preservation_and_actual_idle_probe(self):
        byte_calls = [node for node in walk(self.functions['heldout-controls=?'])
                      if node and node[0] == 'd2-core-bytes=?']
        self.assertEqual(len(byte_calls), 3)
        copy = list(walk(self.functions['heldout-copy-controls!']))
        self.assertIn(['cpu4-trainer-control-begin-internal', 'trainer', ['quote', 'snapshot']], copy)
        self.assertTrue(any(node and node[0] == 'cpu4-trainer-control-end-internal!' for node in copy))
        for label in ('evaluation changed controls/RNG/training cursor/config',
                      'evaluation changed one of the17 P1 modes',
                      'evaluation changed the held-out logical cursor',
                      'evaluation changed full14 values/28 moments/identities/absent gradients'):
            self.assertIn(label, self.text)
        for token in ('HELDOUT_WORDS = 1192', 'HELDOUT_VALUE_BYTES = 4768',
                      'moments[2][HELDOUT_WORDS]', 'et_o2_test_optimizer_moment_bits_v1',
                      'et_f32_parameter_value_tensor_v1', 'et_f32_parameter_validate_identity_v1'):
            self.assertIn(token, self.native)
        for forbidden in ('state_snapshot_v1', 'value_snapshot_v1', 'copy_bits_from_v1',
                          'fail_alloc', 'gradient_plan', 'model_owner_create'):
            self.assertNotIn(forbidden, self.native)

    def test_owned_cleanup_precedes_emission_and_return(self):
        form = self.functions['cpu4-heldout-evaluate-internal!']
        text = self.text[self.text.index('(define (cpu4-heldout-evaluate-internal!'):]
        self.assertLess(text.rindex('(heldout-cleanup! ledger)'), text.index('(heldout-emit phase report)'))
        cleanup = list(walk(self.functions['heldout-cleanup!']))
        for name in ('e3-frame-unbind!', 'heldout-output-destroy', 'heldout-image-destroy',
                     'heldout-live-owned'):
            self.assertTrue(any(node and node[0] == name for node in cleanup))
        self.assertIn(['vector-set!', 'heldout-fixture-session', '0', '#f'], list(walk(form)))
        self.assertIn(['vector-set!', 'heldout-fixture-session', '2', ['quote', 'done']], list(walk(form)))
        self.assertIn('while (value && value != candidate)', self.native)
        self.assertIn('while (*slot && *slot != candidate)', self.native)

    def test_closed_nonbuiltin_loop_names(self):
        names = {node[1] for node in self.nodes if len(node) > 2 and node[0] == 'let'
                 and isinstance(node[1], str)}
        self.assertEqual(names, {'capture-modes', 'destroy-outputs', 'emit-fields',
                                 'create-outputs', 'validate-words'})


if __name__ == '__main__':
    unittest.main()
