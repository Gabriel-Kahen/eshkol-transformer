from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / 'native'
FIXTURE = ROOT / 'tests/cli3_generate'
PREFIX = NATIVE / 'cli3_generate'
INCLUDES = [ROOT / path for path in (
    'internal/p1/lib', 'internal/c1/lib', 'internal/t2/lib',
    'internal/t1/lib', 'internal/d2/lib', 'internal/e3/lib', 'src')]


def policy(inputs, includes=INCLUDES):
    source = (ROOT / 'scripts/build-e1b-consumer.sh').read_text()
    prelude = source.split('require_command realpath\n', 1)[1].split(
        'private_root="$(realpath -- "$1")"', 1)[0]
    script = (
        'set -euo pipefail\n'
        f'source "{ROOT}/scripts/common.sh"\n'
        f'source "{ROOT}/scripts/cli3-native-closure-policy.sh"\n'
        'require_command realpath\n' + prelude +
        'printf "%s %s %s %s %s\\n" '
        '"${cli3_production_exact_tuple}" "${cli3_generate_exact_tuple}" '
        '"${m3_tuple_requested}" "${m3t_tuple_requested}" '
        '"${cli3_tuple_requested}"\n')
    return subprocess.run(
        ['/usr/bin/bash', '-c', script, 'production-policy',
         *map(str, inputs), '/tmp/unused-production-policy.o',
         *map(str, includes)], cwd=ROOT, capture_output=True, text=True)


class ProductionGenerateContract(unittest.TestCase):
    def setUp(self):
        self.inputs = [Path(f'{PREFIX}_{name}') for name in (
            'root.esk', 'package_bridge.c', 'private_renames.txt',
            'public_exports.txt')]

    def test_exact_production_tuple_and_near_matches(self):
        accepted = policy(self.inputs)
        self.assertEqual((accepted.returncode, accepted.stdout.strip()),
                         (0, '1 0 0 0 1'), accepted.stderr)
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / 'public_exports.txt'
            copied.write_bytes(self.inputs[3].read_bytes())
            cases = [
                ([*self.inputs[:3], copied], INCLUDES),
                ([self.inputs[0], copied, *self.inputs[2:]], INCLUDES),
                (self.inputs, [*INCLUDES[:2], INCLUDES[3], INCLUDES[2],
                               *INCLUDES[4:]]),
                ([self.inputs[0].parent / '..' / 'native' /
                  self.inputs[0].name, *self.inputs[1:]], INCLUDES),
            ]
            for inputs, includes in cases:
                with self.subTest(inputs=inputs):
                    rejected = policy(inputs, includes)
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertIn('policy requires', rejected.stderr)

    def test_two_export_dispatch_and_reviewed_closures(self):
        self.assertEqual((NATIVE / 'cli3_generate_public_exports.txt').read_bytes(),
                         (NATIVE / 'cli3_public_exports.txt').read_bytes())
        self.assertEqual((NATIVE / 'cli3_generate_public_strings.txt').read_bytes(),
                         (NATIVE / 'cli3_public_strings.txt').read_bytes())
        self.assertEqual((NATIVE / 'cli3_generate_undefined_symbols.txt').read_bytes(),
                         (FIXTURE / 'aggregate_undefined_symbols.txt').read_bytes())
        source = (FIXTURE / 'aggregate_source_closure.txt').read_text()
        for old, new in (
            ('tests/cli3_generate/aggregate_root.esk',
             'native/cli3_generate_root.esk'),
            ('tests/cli3_generate/t2_g3g_private_bridge.esk',
             'native/cli3_generate_t2_bridge.esk'),
            ('tests/cli3_generate/generate_extension.esk',
             'native/cli3_generate_extension.esk')):
            source = source.replace(old, new)
        self.assertEqual((NATIVE / 'cli3_generate_source_closure.txt').read_text(),
                         source)
        native = (FIXTURE / 'aggregate_native_source_closure.txt').read_text()
        for old, new in (
            ('tests/cli3_generate/aggregate_bridge.c',
             'native/cli3_generate_package_bridge.c'),
            ('tests/cli3_generate/aggregate_native_owner.c',
             'native/cli3_generate_native_owner.c')):
            native = native.replace(old, new)
        self.assertEqual((NATIVE / 'cli3_generate_native_source_closure.txt').read_text(),
                         native)
        bridge = (NATIVE / 'cli3_generate_package_bridge.c').read_text()
        self.assertIn('#define cli3_private_dispatch_cabi_v1 '
                      'cli3_generate_private_dispatch_cabi_v1', bridge)
        self.assertNotIn('et_e1b_public_g3_', bridge)
        self.assertIn('native/cli3_generate_root.esk',
                      (ROOT / 'scripts/build-cli3.sh').read_text())

    def test_command_retains_reviewed_ownership_order(self):
        production = (NATIVE / 'cli3_generate_extension.esk').read_text()
        reviewed = (FIXTURE / 'generate_extension.esk').read_text()
        self.assertEqual(production.split('\n', 2)[2], reviewed.split('\n', 3)[3])
        sequence = ('(cli3-b-restore! trainer checkpoint)',
                    '(trainer-release! trainer)', '(set! trainer #f)',
                    '(module-eval! model)', '(g3g-public-generator-create')
        positions = []
        for item in sequence:
            positions.append(production.index(item, positions[-1] if positions else 0))
        self.assertEqual(positions, sorted(positions))
        self.assertIn('(if owned-trainer', production)


if __name__ == '__main__':
    unittest.main()
