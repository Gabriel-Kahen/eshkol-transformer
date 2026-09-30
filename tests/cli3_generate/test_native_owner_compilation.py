from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CLANG = shutil.which('clang')
NM = shutil.which('nm')
COMMON = [
    '-std=c11', '-Wall', '-Wextra', '-Werror', '-Wpedantic',
    '-fstack-protector-all', '-fPIC', '-fvisibility=hidden', '-fno-common',
    '-ffp-contract=off', '-fexcess-precision=standard', '-frounding-math',
    '-fno-fast-math', '-I', str(ROOT / 'include'),
    '-I', str(ROOT / 'native'), '-I', str(ROOT / 'src'),
]
G3_FLAGS = [
    '-DET_G3T_PREFILL_SAMPLE_PRIVATE', '-DET_G3T_OUTPUT_TEXT_PRIVATE',
    '-DET_G3T_FINAL_PUBLICATION_PRIVATE',
    '-DET_G3T_FULL_REQUEST_PREFLIGHT_PRIVATE',
    '-DET_G3T_OUTPUT_IDS_CLONE_PRIVATE',
    '-DET_G3T_OUTPUT_LENGTHS_CLONE_PRIVATE',
    '-DET_G3T_OUTPUT_CACHE_LENGTHS_CLONE_PRIVATE',
    '-DET_G3T_OUTPUT_RNG_CLONE_PRIVATE',
    '-DET_G3T_GENERATOR_RNG_PRIVATE',
    '-DET_G3T_INPUT_FROM_T1_PRIVATE',
    '-DET_G3T_OWNED_TOKEN_INPUT_PRIVATE',
    '-DET_I64_TENSOR_STORAGE_QUERY_PRIVATE',
    '-DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE',
]


@unittest.skipUnless(CLANG and NM, 'requires clang and nm')
class NativeOwnerCompilation(unittest.TestCase):
    def compile(self, source, output, flags=()):
        return subprocess.run(
            [CLANG, *COMMON, *flags, '-c', str(ROOT / source), '-o', str(output)],
            capture_output=True, text=True)

    def symbols(self, path, global_only=False):
        flags = ['-g', '--defined-only'] if global_only else []
        output = subprocess.check_output([NM, *flags, str(path)], text=True)
        return [line.split()[-1] for line in output.splitlines() if line.strip()]

    def test_one_shared_m3_owner_preserves_both_facets(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            e3 = directory / 'e3.o'
            g3 = directory / 'g3.o'
            aggregate = directory / 'aggregate.o'
            for source, output, flags in (
                ('src/eshkol_transformer/e3_frame.c', e3, ()),
                ('src/eshkol_transformer/g3t_transport.c', g3, G3_FLAGS),
                ('tests/cli3_generate/aggregate_native_owner.c', aggregate,
                 ['-DET_CLI3_E3_G3T_AGGREGATE_BUILD', *G3_FLAGS]),
            ):
                result = self.compile(source, output, flags)
                self.assertEqual(result.returncode, 0, result.stderr)
            e3_symbols = set(self.symbols(e3, True))
            g3_symbols = set(self.symbols(g3, True))
            aggregate_symbols = self.symbols(aggregate, True)
            self.assertTrue(any(name.startswith('et_m3t_')
                                for name in e3_symbols & g3_symbols))
            self.assertEqual(set(aggregate_symbols), e3_symbols | g3_symbols)
            self.assertEqual(len(aggregate_symbols), len(set(aggregate_symbols)))
            self.assertEqual(self.symbols(aggregate).count('m3_workspaces'), 1)

    def test_wrong_build_mode_or_owner_order_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'rejected.o'
            no_mode = self.compile('tests/cli3_generate/aggregate_native_owner.c',
                                   output, G3_FLAGS)
            self.assertNotEqual(no_mode.returncode, 0)
            self.assertIn('exact package build mode', no_mode.stderr)
            wrong_order = self.compile('src/eshkol_transformer/g3t_transport.c',
                                       output, [
                                           '-DET_G3T_REUSE_E3_M3_OWNER_PRIVATE',
                                           *G3_FLAGS,
                                       ])
            self.assertNotEqual(wrong_order.returncode, 0)
            self.assertIn('requires the CLI3 E3/G3-T aggregate TU',
                          wrong_order.stderr)

    def test_production_owner_matches_reviewed_single_registry(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fixture = directory / 'fixture.o'
            production = directory / 'production.o'
            flags = ['-DET_CLI3_E3_G3T_AGGREGATE_BUILD', *G3_FLAGS]
            for source, output in (
                ('tests/cli3_generate/aggregate_native_owner.c', fixture),
                ('native/cli3_generate_native_owner.c', production),
            ):
                result = self.compile(source, output, flags)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.symbols(production, True),
                             self.symbols(fixture, True))
            self.assertEqual(self.symbols(production).count('m3_workspaces'), 1)
            rejected = self.compile('native/cli3_generate_native_owner.c',
                                    directory / 'rejected.o', G3_FLAGS)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('exact package build mode', rejected.stderr)


if __name__ == '__main__':
    unittest.main()
