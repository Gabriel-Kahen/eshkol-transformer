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

    def test_readonly_d1_exports_and_reviewed_closures(self):
        exports = (NATIVE / 'cli3_generate_public_exports.txt').read_text().splitlines()
        base = (NATIVE / 'cli3_public_exports.txt').read_text().splitlines()
        d1 = (NATIVE / 'd1_e1b_public_exports.txt').read_text().splitlines()
        readonly = [name for name in d1 if name !=
                    'et_e1b_public_d1_token_corpus_write_v1']
        self.assertEqual(exports, sorted(set(base) | set(readonly)))
        self.assertEqual(len(exports), 9)
        self.assertFalse(any(name.startswith('et_e1b_public_d2_') for name in exports))
        errors = [name for name in (NATIVE / 'cli3_defined_symbols.txt').read_text().splitlines()
                  if name.startswith('et_e1b_error_')]
        self.assertEqual((NATIVE / 'cli3_generate_defined_symbols.txt').read_text().splitlines(),
                         sorted(set(errors) | set(exports)))
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
        build = (ROOT / 'scripts/build-cli3.sh').read_text()
        self.assertIn('tests/cli3/stage-cli3-d1-readonly.py', build)
        self.assertNotIn('scripts/stage-cli3-d1-readonly.py', build)
        self.assertIn('ESHKOL_LIB_DIR="${public_dir}"', build)
        self.assertIn('"${public_dir}/transformer/data.esk"', build)
        self.assertIn('compile_app --emit-object --emit-depfile', build)
        self.assertIn('[[ -s "${temporary_dir}/eshkol-transformer.d" ]]', build)
        self.assertLess(build.index('compile_app --emit-object --emit-depfile'),
                        build.index('compile_app "${app_source}" -o'))
        self.assertIn('cli3_generate_defined_symbols.txt', build)
        self.assertIn('(require transformer.data)',
                      (ROOT / 'src/eshkol_transformer/cli.esk').read_text())

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

    def test_installed_validate_uses_shared_public_handler(self):
        standalone = (ROOT / 'cli/eshkol-transformer.esk').read_text()
        self.assertEqual(standalone, (
            ';;; Standalone public-D1 corpus validation entry.\n'
            '(require transformer.cli3_corpus_validate)\n\n'
            '(cli3-corpus-validate-command (cdr (command-line)))\n'))
        handler = (ROOT / 'lib/transformer/cli3_corpus_validate.esk').read_text()
        self.assertIn('(require transformer.data)', handler)
        self.assertIn('(provide cli3-corpus-validate-command)', handler)
        self.assertIn('(token-corpus-validate directory', handler)
        self.assertNotIn('et-d1-private-', handler)
        self.assertNotIn('et_e1b_public_d2_', handler)
        entry = (ROOT / 'src/eshkol_transformer/cli.esk').read_text()
        self.assertLess(entry.index('(cli3-corpus-validate-command arguments)'),
                        entry.index('(cli3-dispatch (vector arguments) output)'))
        self.assertIn('(require transformer.cli3_corpus_validate)', entry)
        build = (ROOT / 'scripts/build-cli3.sh').read_text()
        self.assertIn('"${public_dir}/transformer/cli3_corpus_validate.esk"', build)
        production = (ROOT / 'scripts/test-cli3-production-generate.sh').read_text()
        self.assertIn('tests.cli3.test_installed_corpus_validate', production)


if __name__ == '__main__':
    unittest.main()
