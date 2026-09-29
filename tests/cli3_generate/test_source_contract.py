from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'tests/cli3_generate'
NATIVE = ROOT / 'native'


def lines(path):
    return path.read_text().splitlines()


class SingleRegistryContract(unittest.TestCase):
    def test_accepts_exact_g3g_p1_g1_suffix(self):
        aggregate = lines(FIXTURE / 'aggregate_root.esk')
        g3 = lines(NATIVE / 'g3g_package_root.esk')
        self.assertEqual(aggregate[1], '(load "cli3_root.esk")')
        self.assertEqual(aggregate[2], '(load "t2_g3g_private_bridge.esk")')
        self.assertEqual(aggregate[3:-1], g3[3:])
        self.assertEqual(aggregate[-1], '(load "generate_extension.esk")')
        self.assertEqual(aggregate.count('(load "cli3_root.esk")'), 1)
        self.assertNotIn('(load "m3_package_root.esk")', aggregate)

    def test_g3g_uses_the_existing_t2_tokenizer_authority(self):
        bridge = (FIXTURE / 't2_g3g_private_bridge.esk').read_text()
        for alias in ('(define t1-private-entry t2-private-entry)',
                      '(t2-private-raise category operation message)',
                      '(define t1-wave1-native-fail t2-wave2-native-fail)'):
            self.assertIn(alias, bridge)
        self.assertNotIn('(load ', bridge)
        self.assertNotIn('(require ', bridge)
        self.assertNotIn('t1-private-tokenizer-registry', bridge)

        closure = lines(FIXTURE / 'aggregate_source_closure.txt')
        self.assertEqual(closure.count('tests/cli3_generate/t2_g3g_private_bridge.esk'), 1)
        self.assertEqual(closure.index('tests/cli3_generate/t2_g3g_private_bridge.esk') + 1,
                         closure.index('native/g3t_model_admission_extension.esk'))
        self.assertIn('internal/t2/lib/transformer/tokenizer_internal.esk', closure)
        self.assertNotIn('native/t1_wave1_root.esk', closure)
        self.assertNotIn('internal/t1/lib/transformer/tokenizer_internal.esk', closure)

    def test_public_exports_are_reviewed_union(self):
        expected = (set(lines(NATIVE / 'tr3_public_installed_exports.txt'))
                    | set(lines(NATIVE / 'cli3_public_exports.txt'))
                    | {name for name in lines(NATIVE / 'g3g_package_public_exports.txt')
                       if name.startswith(('et_e1b_public_g3_',
                                           'et_e1b_public_m3_'))})
        expected.add('et_e1b_public_cli3_generate_dispatch_v1')
        exports = lines(FIXTURE / 'aggregate_public_exports.txt')
        self.assertEqual(exports, sorted(name for name in expected
                                         if name.startswith('et_e1b_public_')))

        error_globals = {name for name in lines(NATIVE / 'cli3_public_strings.txt')
                         if name.startswith('et_e1b_error_')}
        self.assertEqual(len(error_globals), 6)
        self.assertEqual(lines(FIXTURE / 'aggregate_public_strings.txt'),
                         sorted(set(exports) | error_globals))

        build = (ROOT / 'scripts/build-e1b-consumer.sh').read_text()
        pattern = re.search(r"^export_pattern='([^']+)'$", build, re.M).group(1)
        for name in exports:
            self.assertIsNotNone(re.fullmatch(pattern, name), name)
        for name in error_globals:
            with self.subTest(rejected_export=name):
                self.assertIsNone(re.fullmatch(pattern, name))

    def test_bridge_keeps_one_predecessor(self):
        bridge = (FIXTURE / 'aggregate_bridge.c').read_text()
        m3 = (NATIVE / 'm3_package_bridge.c').read_text()
        g3 = (NATIVE / 'g3g_package_bridge.c').read_text()
        m3_wrappers = m3.split('#include "m3t_package_bridge.c"\n', 1)[1].strip()
        tr3 = (NATIVE / 'tr3_public_installed_bridge.c').read_text()
        g3_wrappers = g3.split('#include "m3_package_bridge.c"\n', 1)[1].strip()
        expected = (
            '/* Test-only same-registry CLI3/TR3 plus accepted G3 wrappers. */\n'
            '#include "cli3_package_bridge.c"\n\n'
            + g3_wrappers + '\n')
        self.assertIn(m3_wrappers, tr3)
        self.assertTrue(bridge.startswith(expected))
        self.assertNotIn(m3_wrappers, bridge)
        self.assertIn('void et_e1b_public_cli3_generate_dispatch_v1(',
                      bridge[len(expected):])

    def test_witness_uses_public_checkpoint_and_generation(self):
        witness = (FIXTURE / 'public_runtime.esk').read_text()
        for call in ('(capability-discover)', '(checkpoint-load ',
                     '(trainer-load-state! trainer loaded)',
                     '(module-eval! model)',
                     '(generator-create model tokenizer selected-policy)',
                     '(tokenizer-encode tokenizer (bytevector 65))',
                     '(generation-input-create encoded)',
                     '(generation-output-text output)', '(greedy-oracle '):
            self.assertIn(call, witness)
        self.assertNotIn('et_e1b_private_', witness)
        self.assertNotIn('g3t-', witness)

    def test_generate_command_is_a_bounded_successor(self):
        extension = (FIXTURE / 'generate_extension.esk').read_text()
        self.assertIn('(cli3-b-restore! trainer checkpoint)', extension)
        self.assertIn('(g3g-public-generator-create', extension)
        self.assertIn('(t2-wave2-tokenizer-encode tokenizer (bytevector prompt))',
                      extension)
        self.assertIn('(g3g-public-generation-output-text output)', extension)
        self.assertIn('(g3g-public-generator-close owned-generator)', extension)
        self.assertIn('(g3g-public-generation-output-release owned-output)', extension)
        self.assertIn('(g3g-public-generation-tensor-release owned-input)', extension)
        self.assertNotIn('g3c4-', extension)
        self.assertEqual(lines(FIXTURE / 'aggregate_source_closure.txt')[-1],
                         'tests/cli3_generate/generate_extension.esk')
        entry = (FIXTURE / 'cli.esk').read_text()
        original = (ROOT / 'src/eshkol_transformer/cli.esk').read_text()
        self.assertEqual(
            entry.replace('Test-installed CLI3/G3-G successor entry; the package owns dispatch.',
                          'Thin Eshkol-native executable entry. The CLI3 aggregate owns all dispatch.')
                 .replace('cli3-generate-dispatch', 'cli3-dispatch')
                 .replace('et_e1b_public_cli3_generate_dispatch_v1',
                          'et_e1b_public_cli3_dispatch_v1'),
            original)


if __name__ == '__main__':
    unittest.main()
