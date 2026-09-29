from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / 'scripts/build-e1b-consumer.sh'
PREFIX = ROOT / 'tests/cli3_generate/aggregate'


def dispatch(inputs, includes):
    source = BUILD.read_text()
    prelude = source.split('require_command realpath\n', 1)[1].split(
        'private_root="$(realpath -- "$1")"', 1)[0]
    script = (
        'set -euo pipefail\n'
        f'source "{ROOT}/scripts/common.sh"\n'
        f'source "{ROOT}/scripts/cli3-native-closure-policy.sh"\n'
        'require_command realpath\n'
        + prelude
        + 'printf "%s %s %s %s %s %s %s\\n" '
          '"${cli3_generate_exact_tuple}" "${m3_tuple_requested}" '
          '"${m3t_tuple_requested}" "${e3_tuple_requested}" '
          '"${g3g_tuple_requested}" "${g3g_g0_tuple_requested}" '
          '"${g3g_manual_tuple_requested}"\n'
    )
    return subprocess.run(
        ['/usr/bin/bash', '-c', script, 'policy-dispatch', *map(str, inputs),
         '/tmp/unused-policy-dispatch.o', *map(str, includes)],
        cwd=ROOT, text=True, capture_output=True, check=False)


class PolicyDispatch(unittest.TestCase):
    def setUp(self):
        self.inputs = [Path(f'{PREFIX}_{name}') for name in
                       ('root.esk', 'bridge.c', 'private_renames.txt',
                        'public_exports.txt')]
        self.includes = [ROOT / name for name in
                         ('internal/p1/lib', 'internal/c1/lib', 'internal/t2/lib',
                          'internal/t1/lib', 'internal/d2/lib', 'internal/e3/lib',
                          'src')]

    def test_exact_generate_tuple_dispatches_past_predecessor_policies(self):
        result = dispatch(self.inputs, self.includes)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '1 0 0 0 0 0 0')

    def test_near_match_copy_and_foreign_input_still_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / 'exports.txt'
            copied.write_bytes(self.inputs[3].read_bytes())
            cases = [
                ([self.inputs[0].parent / '..' / 'cli3_generate' /
                  self.inputs[0].name, *self.inputs[1:]], self.includes),
                ([*self.inputs[:3], copied], self.includes),
                ([self.inputs[0], copied, *self.inputs[2:]], self.includes),
                (self.inputs, [*self.includes[:2], self.includes[3],
                               self.includes[2], *self.includes[4:]]),
            ]
            for inputs, includes in cases:
                with self.subTest(inputs=inputs, includes=includes):
                    result = dispatch(inputs, includes)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('M3 policy requires the exact lexical repository tuple',
                                  result.stderr)

    def test_dedicated_predecessor_tuples_keep_their_policy(self):
        includes = [ROOT / name for name in
                    ('internal/p1/lib', 'internal/c1/lib', 'internal/t1/lib', 'src')]
        for stem, flags in (
            ('m3_package', '0 1 0 0 0 0 0'),
            ('m3t_package', '0 0 1 0 0 0 0'),
            ('g3g_package', '0 0 0 0 1 0 0'),
            ('g3g_g0_package', '0 0 0 0 0 1 0'),
            ('g3g_manual_package', '0 0 0 0 0 0 1'),
        ):
            with self.subTest(stem=stem):
                inputs = [ROOT / f'native/{stem}_{name}' for name in
                          ('root.esk', 'bridge.c', 'private_renames.txt',
                           'public_exports.txt')]
                result = dispatch(inputs, includes)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), flags)

        e3_inputs = [ROOT / f'native/{name}' for name in
                     ('e3_private_driver_root.esk', 'e3_private_bridge.c',
                      'e3_private_package_private_renames.txt',
                      'e3_private_package_public_exports.txt')]
        e3_includes = [ROOT / name for name in
                       ('internal/p1/lib', 'internal/c1/lib', 'internal/t1/lib',
                        'src', 'internal/d2/lib', 'internal/e3/lib')]
        result = dispatch(e3_inputs, e3_includes)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '0 0 0 1 0 0 0')


if __name__ == '__main__':
    unittest.main()
