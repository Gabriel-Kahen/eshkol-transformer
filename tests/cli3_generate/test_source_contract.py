from pathlib import Path
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
        self.assertEqual(aggregate[2:-1], g3[3:])
        self.assertEqual(aggregate[-1], '(load "generate_extension.esk")')
        self.assertEqual(aggregate.count('(load "cli3_root.esk")'), 1)
        self.assertNotIn('(load "m3_package_root.esk")', aggregate)

    def test_public_exports_are_reviewed_union(self):
        expected = (set(lines(NATIVE / 'tr3_public_installed_exports.txt'))
                    | set(lines(NATIVE / 'cli3_public_exports.txt'))
                    | {name for name in lines(NATIVE / 'g3g_package_public_exports.txt')
                       if name.startswith(('et_e1b_public_g3_',
                                           'et_e1b_public_m3_'))})
        expected.add('et_e1b_public_cli3_generate_dispatch_v1')
        self.assertEqual(lines(FIXTURE / 'aggregate_public_exports.txt'),
                         sorted(expected))

    def test_bridge_keeps_one_predecessor(self):
        bridge = (FIXTURE / 'aggregate_bridge.c').read_text()
        m3 = (NATIVE / 'm3_package_bridge.c').read_text()
        g3 = (NATIVE / 'g3g_package_bridge.c').read_text()
        m3_wrappers = m3.split('#include "m3t_package_bridge.c"\n', 1)[1].strip()
        g3_wrappers = g3.split('#include "m3_package_bridge.c"\n', 1)[1].strip()
        expected = (
            '/* Test-only same-registry CLI3/TR3 plus reviewed M3 and G3 wrappers. */\n'
            '#include "cli3_package_bridge.c"\n\n'
            + m3_wrappers + '\n\n' + g3_wrappers + '\n')
        self.assertTrue(bridge.startswith(expected))
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
