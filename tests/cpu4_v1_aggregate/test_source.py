"""Cheap causal controls for the real source/TU union; never builds a binary."""
import copy
import hashlib
import unittest
from .closure import (ENTRY, ROOT, load_roster, manifest, source_closure,
                      native_closure, validate)


class SourceUnion(unittest.TestCase):
    def setUp(self):
        self.roster = load_roster()
        self.text = (ROOT / ENTRY).read_text()

    def reject_source(self, suffix):
        with self.assertRaises((AssertionError, ValueError)):
            validate(ROOT, self.roster, {ENTRY: self.text + suffix})

    def test_actual_transitive_union_and_pinned_inputs(self):
        closure, native = validate(ROOT, self.roster)
        self.assertEqual((ROOT / 'native/cpu4_v1_source_members.sha256').read_text(),
                         manifest(ROOT, self.roster))
        paths = {path for path, _ in closure}
        self.assertTrue({'native/m3_call_adapters.esk', 'native/m3t_transport_extension.esk',
                         'native/m3_model_extension.esk', 'native/g3c4_public_generation_dispatch_extension.esk',
                         'native/g3c4_public_g2_result_extension.esk',
                         'native/g3c4_p2g2_terminal_commit_extension.esk'} <= paths)
        self.assertEqual(native['native/f32_tensor.c'], 'src/eshkol_transformer/m3_call_f32_integration.c')
        self.assertEqual(native['native/i64_tensor.c'], 'src/eshkol_transformer/m3_i64_integration.c')
        self.assertEqual(native['src/eshkol_transformer/m3t_transport.c'], 'src/eshkol_transformer/g3t_transport.c')
        self.assertEqual(native['native/d2_native.c'], 'native/e3_d2_native.c')

    def test_copied_authority_and_fixture_roots_rejected(self):
        for name in ('i2_wave2_extension.esk', 'e3_cpu4_boundary.esk', 'e3_cpu4_private_root.esk',
                     'd2_wave2_root.esk', 'cpu4_checkpoint_root.esk',
                     'c2_wave2_root.esk', 'k2_wave2_root.esk'):
            with self.subTest(name=name):
                self.reject_source(f'\n(load "{name}")\n')

    def test_original_t1_and_common_state_cannot_be_rebound(self):
        for definition in ('(define t1-wave1-tensor-admitted? t2-wave2-tensor-admitted?)',
                           '(define m3-call-state (vector #f))',
                           '(define i2-module-registry-root (vector (quote ())))'):
            with self.subTest(definition=definition):
                self.reject_source('\n' + definition + '\n')

    def test_e3_seven_helpers_are_the_original_forms(self):
        from tests.cpu4_trainer_control.test_source_contract import parse
        actual = {}
        for path, forms in source_closure(ROOT, self.roster):
            for form in forms:
                if len(form) > 2 and form[0] in ('define', 'define-syntax'):
                    name = form[1][0] if isinstance(form[1], list) else form[1]
                    actual[name] = form
        copied = parse((ROOT / 'native/e3_cpu4_boundary.esk').read_text())
        definitions = [form for form in copied if len(form) > 2 and form[0] in ('define', 'define-syntax')]
        self.assertEqual(len(definitions), 7)
        for form in definitions:
            name = form[1][0] if isinstance(form[1], list) else form[1]
            self.assertEqual(actual[name], form, name)

    def test_g2_dead_release_uses_loaded_identity_helper(self):
        from tests.cpu4_trainer_control.test_source_contract import parse, walk
        forms = parse((ROOT / 'native/g3c4_public_g2_result_extension.esk').read_text())
        body = next(f for f in forms if f[0] == 'define' and isinstance(f[1], list) and f[1][0] == 'g3c4-public-g2-output-release!')
        self.assertIn(['i2-identity-member?', 'shell', ['vector-ref', 'g3c4-public-g2-released-outputs', '0']], list(walk(body)))
        self.assertFalse(any(f and f[0] == 'memq' for f in walk(body)))
        definitions = {f[1][0]: f for _, source_forms in source_closure(ROOT, self.roster)
                       for f in source_forms if len(f) > 2 and f[0] == 'define' and isinstance(f[1], list)}
        helper = definitions['i2-identity-member?']
        self.assertIn(['eq?', 'value', ['car', 'values']], list(walk(helper)))

    def test_explicit_original_tokenizer_leaves_cannot_be_omitted(self):
        for leaf in ('t1', 't2'):
            text = self.text.replace(f'(load "../internal/{leaf}/lib/transformer/tokenizer_internal.esk")', '')
            with self.subTest(leaf=leaf), self.assertRaises(AssertionError):
                validate(ROOT, self.roster, {ENTRY: text})
        # Explicit canonical preloads make this dependency independent of which
        # tokenizer_internal short spelling wins later include-path lookup.
        changed = copy.deepcopy(self.roster)
        a, b = changed['source_include_dirs'].index('internal/t1/lib'), changed['source_include_dirs'].index('internal/t2/lib')
        changed['source_include_dirs'][a], changed['source_include_dirs'][b] = changed['source_include_dirs'][b], changed['source_include_dirs'][a]
        validate(ROOT, changed)

    def test_transitive_duplicate_is_detected(self):
        path = 'native/g3c4_public_g2_result_extension.esk'
        text = (ROOT / path).read_text() + '\n(define i2-carrier-tag (vector #f))\n'
        with self.assertRaisesRegex(AssertionError, 'duplicate definition i2-carrier-tag'):
            validate(ROOT, self.roster, {path: text})

    def test_duplicate_native_registries_are_rejected(self):
        for source in ('native/f32_tensor.c', 'src/eshkol_transformer/m3t_f32_integration.c',
                       'native/i64_tensor.c', 'native/d2_native.c', 'native/o2_optimizer.c',
                       'native/i2_wave2_package_bridge.c', 'src/eshkol_transformer/m3_model.c'):
            with self.subTest(source=source):
                changed = copy.deepcopy(self.roster)
                changed['translation_units'].append({'source': source, 'defines': []})
                with self.assertRaises(AssertionError):
                    validate(ROOT, changed)
                # Exercise the actual .c identity walker too, independently of roster equality.
                with self.assertRaisesRegex(AssertionError, 'duplicate native identity'):
                    native_closure(ROOT, changed)

    def test_candidate_off_and_m3t_package_flags_are_rejected(self):
        for mode in ('off', 'on'):
            changed = copy.deepcopy(self.roster)
            changed['candidate_mode'] = mode
            changed['common_defines'].remove('ET_CPU4_CANDIDATE_PRIVATE')
            with self.subTest(mode=mode), self.assertRaises(AssertionError):
                validate(ROOT, changed)
        for flag in ('ET_M3T_PACKAGE_BUILD', 'ET_M3T_PACKAGE_BUILD=1', 'ET_CPU4_CANDIDATE_PRIVATE=0', 'ET_F32_TENSOR_TESTING'):
            changed = copy.deepcopy(self.roster)
            changed['translation_units'][0]['defines'].append(flag)
            with self.subTest(flag=flag), self.assertRaises(AssertionError):
                validate(ROOT, changed)

    def test_helper_only_and_generation_features_cannot_disappear(self):
        cases = {
            'native/i2_wave2_package_bridge.c': 'ET_I2_NATIVE_HELPERS_ONLY',
            'native/o2_wave2_package_bridge.c': 'ET_O2_NATIVE_HELPERS_ONLY',
            'src/eshkol_transformer/g3c4_model_owner.c': 'ET_G3C4_P2_G2_TYPED_RESULT_PRIVATE',
            'src/eshkol_transformer/g3t_transport.c': 'ET_G3T_GENERATOR_RNG_PRIVATE',
            'native/tr3_c_restore_bindings.c': 'ET_TR3_C_I2_RESTORE_PRIVATE',
        }
        for source, flag in cases.items():
            changed = copy.deepcopy(self.roster)
            next(u for u in changed['translation_units'] if u['source'] == source)['defines'].remove(flag)
            with self.subTest(source=source), self.assertRaises(AssertionError):
                validate(ROOT, changed)

    def test_factory_requires_real_reader_k2_and_child_order(self):
        leaves = ('k2_wave2_extension', 'c2_o2_encode_extension', 'c2_model_encode_extension',
                  'c2_persistence_policy_extension', 'c2_checkpoint_inspect_extension',
                  'c2_checkpoint_save_extension', 'c2_checkpoint_load_extension',
                  'c2_public_extension', 'cpu4_checkpoint_factory_extension')
        for leaf in leaves:
            changed = self.text.replace(f'(load "{leaf}.esk")', '')
            with self.subTest(missing=leaf), self.assertRaises(AssertionError):
                validate(ROOT, self.roster, {ENTRY: changed})
        factory = '(load "cpu4_checkpoint_factory_extension.esk")'
        changed = self.text.replace(factory, '').replace('(load "cpu4_candidate_children_extension.esk")',
                                                       factory + '\n(load "cpu4_candidate_children_extension.esk")')
        with self.assertRaisesRegex(AssertionError, 'leaf prerequisite order'):
            validate(ROOT, self.roster, {ENTRY: changed})

    def test_factory_guard_and_original_t2_policy_installation(self):
        checkpoint = 'native/cpu4_checkpoint_extension.esk'
        text = (ROOT / checkpoint).read_text().replace(
            "(cpu4-checkpoint-require-nonregional! 'cpu4-trainer-restore)", '#t')
        self.assertNotEqual(text, (ROOT / checkpoint).read_text())
        with self.assertRaisesRegex(AssertionError, 'nonregional guard must be first'):
            validate(ROOT, self.roster, {checkpoint: text})
        public = 'native/c2_public_extension.esk'
        text = (ROOT / public).read_text().replace(
            '(vector-set! t2-wave2-policy-projector 0 c2-policy-t2-entry-internal)',
            '(vector-set! t2-wave2-policy-projector 0 #f)')
        with self.assertRaisesRegex(AssertionError, 'original T2 policy installation changed'):
            validate(ROOT, self.roster, {public: text})
        for source, flag in (('native/k2_capabilities.c', 'ET_C2_CARRIER_FACTORIES'),
                             ('native/i2_wave2_package_bridge.c', 'ET_C2_I2_MODEL_COPY')):
            changed = copy.deepcopy(self.roster)
            next(u for u in changed['translation_units'] if u['source'] == source)['defines'].remove(flag)
            with self.subTest(flag=flag), self.assertRaises(AssertionError):
                validate(ROOT, changed)

    def test_current_sanitizer_helper_and_proof_are_pinned(self):
        for line in (ROOT / 'tests/e3_private/asan_ir_sources.sha256').read_text().splitlines():
            digest, path = line.split(None, 1)
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
        self.assertIn('tests/shared/prepare_eshkol_asan_ir.py', manifest(ROOT, self.roster))
        self.assertIn('tests/e3_private/check_asan_ir.py', manifest(ROOT, self.roster))

    def test_handoff_requires_actual_factory_and_public_generation_order(self):
        handoff = '(load "cpu4_generator_handoff_extension.esk")'
        with self.assertRaisesRegex(AssertionError, 'handoff prerequisite'):
            validate(ROOT, self.roster, {ENTRY: self.text.replace(handoff, '')})
        factory = '(load "cpu4_checkpoint_factory_extension.esk")'
        changed = self.text.replace(handoff, '').replace(factory, handoff + '\n' + factory)
        with self.assertRaisesRegex(AssertionError, 'leaf prerequisite order'):
            validate(ROOT, self.roster, {ENTRY: changed})

    def test_handoff_first_guard_and_original_public_authority(self):
        source = 'native/cpu4_generator_handoff_extension.esk'
        text = (ROOT / source).read_text().replace(
            "(cpu4-checkpoint-require-nonregional! 'cpu4-generator-handoff-retry)", '#t')
        with self.assertRaisesRegex(AssertionError, 'nonregional guard must be first'):
            validate(ROOT, self.roster, {source: text})
        source = 'native/g3c4_public_generation_dispatch_extension.esk'
        text = (ROOT / source).read_text().replace(
            '(define (g3c4-public-g2-profile-live ', '(define (replacement-profile-live ')
        with self.assertRaisesRegex(AssertionError, 'handoff prerequisite'):
            validate(ROOT, self.roster, {source: text})


if __name__ == '__main__':
    unittest.main()
