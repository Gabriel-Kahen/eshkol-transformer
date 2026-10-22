"""Host checks of the genuine detach fixture/recipe; no runtime admission."""
import hashlib
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/cpu4_trainer_control'))
from test_source_contract import parse, walk

FIXTURE = ROOT / 'tests/cpu4_trainer_detach/runtime_test.esk'
RECIPE = ROOT / 'scripts/test-cpu4-trainer-detach.sh'


class DetachFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = FIXTURE.read_text()
        cls.forms = parse(cls.text)
        cls.script = RECIPE.read_text()

    def test_exact_accepted_detach_leaf(self):
        data = (ROOT / 'native/cpu4_trainer_detach_extension.esk').read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),
                         '11c14d9cbefa931e83f38c0185a2c0c3a14a35d627e4868aef4a5923e03d4fc9')

    def test_one_real_identity_aggregate(self):
        loads = [form[1] for form in self.forms if form[0] == 'load']
        self.assertEqual(len(loads), len(set(loads)))
        self.assertEqual(loads[-2:], ['cpu4_trainer_control_extension.esk',
                                     'cpu4_trainer_detach_extension.esk'])
        for name in ('i2_wave2_extension.esk', 'o2_wave2_extension.esk',
                     'c2_d2_cursor_pair_extension.esk', 'g3c4_model_extension.esk',
                     'cpu4_model_binding_extension.esk', 'd2_dataset.esk'):
            self.assertIn(name, loads)
        self.assertNotIn('(extern ', self.text)
        self.assertIn('(t2-wave2-tokenizer-byte resolved)', self.text)
        self.assertNotIn('tokenizer-load', self.text)
        for form in self.forms:
            if form[0] == 'define' and isinstance(form[1], list):
                name = form[1][0]
                self.assertFalse(name.startswith(('cpu4-', 'o2-', 'd2-', 'g3c4-', 't2-', 'i2-')), name)

    def test_actual_exclusion_retry_and_stale_cases(self):
        for label in ('active snapshot prevents detach', 'staged restore prevents detach',
                      'prepared restore prevents detach', 'P1 EVAL preflight prevents detach',
                      'live D2 batch prevents detach', 'native READ contention prevents detach',
                      'failed native prepare preserves same trainer/lease',
                      'failed native prepare leaves READ held',
                      'detach same trainer after failed preflight', 'repeat detach is stale',
                      'dead row accessor rejected', 'old parent cannot regain authority',
                      'old stage cannot regain authority', 'only trainer state slot changed'):
            self.assertIn('"' + label + '"', self.text)
        self.assertNotRegex(self.text, r'\(length\s')
        self.assertNotIn('tr3-lease-', self.text)
        self.assertNotIn('m3t-public-model-create', self.text)

    def test_genuine_components_remain_usable(self):
        heads = {node[0] for form in self.forms for node in walk(form)
                 if node and isinstance(node[0], str)}
        for name in ('o2-public-optimizer-zero-grad!', 'o2-public-optimizer-state',
                     'o2-public-optimizer-state-release!', 'd2-token-dataset-next-batch',
                     'd2-token-batch-release!', 'cpu4-d2-cursor-restore-internal!',
                     'module-eval!', 'module-mode-internal', 'i2-native-borrow-checksum',
                     'i2-parameter-gradient-metadata-internal'):
            self.assertIn(name, heads)
        for label in ('same genuine O2 optimizer remains registered',
                      'same genuine D2 owner remains live', 'all14 canonical parameter values preserved',
                      'all14 gradients remain absent', 'model P1 modes remain usable'):
            self.assertIn('"' + label + '"', self.text)
        self.assertNotIn('optimizer-step!', self.text)

    def test_recipe_preserves_accepted_controls_closure(self):
        original = (ROOT / 'scripts/test-cpu4-trainer-control.sh').read_text()
        expected = original.replace('test-cpu4-trainer-control.sh', 'test-cpu4-trainer-detach.sh')
        expected = expected.replace('cpu4_controls', 'cpu4_detach')
        expected = expected.replace('tests/cpu4_trainer_control/runtime_test.esk',
                                    'tests/cpu4_trainer_detach/runtime_test.esk')
        expected = expected.replace('"$out/controls"', '"$out/detach"')
        expected = expected.replace('CPU4 genuine constructor/snapshot/restore controls PASS:',
                                    'CPU4 genuine unpublished trainer detach PASS:')
        self.assertEqual(self.script, expected)
        self.assertIn('for repeat in 1 2', self.script)
        self.assertIn('--strict-types --optimize 0 --no-stdlib', self.script)
        self.assertNotIn('flock ', self.script)


if __name__ == '__main__':
    unittest.main()
