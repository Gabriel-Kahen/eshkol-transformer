"""Exact M3T surface and repository-only packaging policy, without a compiler."""
from pathlib import Path
import os
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / 'native/m3t_package'


def manifest(suffix):
    return Path(f'{PREFIX}_{suffix}.txt').read_text().splitlines()


def operations():
    proposal = (ROOT / 'docs/M3T_TRANSPORT_PROPOSAL.md').read_text()
    rows = [(n, int(a)) for n, a in re.findall(
        r'\| `(diagnostic-[^`]+)` \| ([012]) \|', proposal)]
    return rows + [(n[:-1] + '-vjp!', a) for n, a in rows[21:]] + [('diagnostic-sum!', 2)]


class PackageContract(unittest.TestCase):
    def test_exact_surface_and_arity(self):
        ops = operations()
        self.assertEqual(len(ops), 38)
        facade = (ROOT / 'lib/transformer/diagnostic_transport.esk').read_text()
        exported = re.search(r'\(provide\s+(.*?)\)', facade, re.S)[1].split()
        self.assertEqual(exported, [n for n, _ in ops])
        self.assertEqual(re.findall(r'\(require ([^)]+)\)', facade), ['transformer.error_consumer'])
        bridge = (ROOT / 'native/m3t_package_bridge.c').read_text()
        self.assertEqual(bridge.count('#include "i2_wave2_package_bridge.c"'), 1)
        for name, arity in ops:
            suffix = name.removeprefix('diagnostic-')
            native = suffix.rstrip('!').replace('-', '_')
            public = f'et_e1b_public_m3t_{native}_v1'
            private = f'et_e1b_private_m3t_{native}_cabi_v1'
            params = re.search(r'\(define \(' + re.escape(name) + r'([^)]*)\)', facade)[1].split()
            self.assertEqual(len(params), arity, name)
            ffi = re.search(r'\(extern void m3t-boxed-' + re.escape(suffix.rstrip('!')) + r'\s+((?:ptr\s+)+):real ' + public, facade)[1].split()
            self.assertEqual(len(ffi), arity + 1, name)
            cparams = re.search(r'void ' + public + r'\(([^)]*)\)', bridge)[1].split(',')
            self.assertEqual(len(cparams), arity + 1, name)
            self.assertIn(f'm3t-public-{suffix} {private}', manifest('private_renames'))
        exports = manifest('public_exports')
        defined = manifest('defined_symbols')
        self.assertEqual(len(exports), 79)
        self.assertEqual(len(defined), 85)
        self.assertEqual(defined, manifest('public_strings'))
        self.assertEqual(len(set(exports) - set((ROOT / 'native/i2_wave2_public_exports.txt').read_text().splitlines())), 38)
        for suffix in ('public_exports', 'defined_symbols', 'private_renames', 'undefined_symbols'):
            rows = manifest(suffix)
            self.assertEqual(rows, sorted(set(rows)))

    def test_installed_closure_and_single_registry(self):
        self.assertEqual(manifest('facades'), [f'transformer/{n}.esk' for n in (
            'config', 'diagnostic_transport', 'error_consumer', 'error_public', 'module', 'tokenizer')])
        self.assertEqual(manifest('archive_members'), ['m3t_package.o'])
        self.assertEqual(manifest('native_objects'), ['n2/n2_primitives_provider.o', 'n3k/n3k_primitives_provider.o', 'a2/a2_attention_provider.o'])
        root = (ROOT / 'native/m3t_package_root.esk').read_text()
        self.assertEqual(re.findall(r'\(load "([^"]+)"\)', root), ['i2_wave2_root.esk', 'm3t_transport_extension.esk'])
        for file in manifest('facades'):
            source = (ROOT / 'lib' / file).read_text()
            imports = re.findall(r'\(require transformer\.([^\s)]+)\)', source)
            self.assertTrue(all(f'transformer/{name}.esk' in manifest('facades') for name in imports))

    def test_ordinary_i2_conditional_helper_does_not_select_m3t_policy(self):
        script = """PROJECT_ROOT=$1
raw_private_root=$1/native/i2_wave2_root.esk
raw_package_bridge=$1/native/i2_wave2_package_bridge.c
raw_package_renames=$1/native/i2_wave2_private_renames.txt
raw_public_exports=$1/native/i2_wave2_public_exports.txt
raw_include_dirs=("$1/internal/p1/lib" "$1/internal/c1/lib" "$1/internal/t1/lib" "$1/src")
die() { echo "$*" >&2; exit 1; }
source "$2"
[[ "$m3t_tuple_requested" == 0 ]]
"""
        result = subprocess.run(['/usr/bin/bash', '-eu', '-c', script, 'policy-test',
            str(ROOT), str(ROOT / 'scripts/m3t-package-policy.sh')], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_source_closure_rejects_every_outside_or_aliased_dependency(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'repository'
            root.mkdir()
            source = root / 'reviewed.esk'
            source.write_text('; reviewed source\n')
            (root / 'alias.esk').symlink_to(source)
            outside = Path(temp) / 'unlisted.esk'
            outside.write_text('; outside source\n')
            script = """PROJECT_ROOT=$1
raw_private_root=$1/absent
raw_package_bridge=$1/absent
raw_package_renames=$1/absent
raw_public_exports=$1/absent
raw_include_dirs=()
die() { echo "$*" >&2; exit 1; }
source "$2"
m3t_normalize_source_dependencies
"""
            for added in ('', str(outside), str(root / 'alias.esk'),
                          str(root / '../repository/reviewed.esk'), 'reviewed.esk'):
                result = subprocess.run(['/usr/bin/bash', '-eu', '-c', script,
                    'source-closure-test', str(root), str(ROOT / 'scripts/m3t-package-policy.sh')],
                    input=f'\n{source}\n{added}\n', text=True, capture_output=True)
                self.assertEqual(result.returncode == 0, not added, result.stderr)
                if not added:
                    self.assertEqual(result.stdout, 'reviewed.esk\n')
                else:
                    self.assertIn('M3T policy', result.stderr)

    def test_native_depfile_alias_cannot_disappear_during_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reviewed = root / 'reviewed'
            reviewed.mkdir()
            (reviewed / 'header.h').write_text('/* reviewed */\n')
            (root / 'alias').symlink_to(reviewed, target_is_directory=True)
            self.assertEqual((root / 'alias/header.h').resolve(), reviewed / 'header.h')
            script = """PROJECT_ROOT=$1
raw_private_root=$1/absent
raw_package_bridge=$1/absent
raw_package_renames=$1/absent
raw_public_exports=$1/absent
raw_include_dirs=()
die() { echo "$*" >&2; exit 1; }
source "$2"
m3t_check_native_dependency_path "$3"
"""
            for spelling, success in [('reviewed/../reviewed/header.h', True),
                                      ('alias/header.h', False), ('alias/../reviewed/header.h', False)]:
                result = subprocess.run(['/usr/bin/bash', '-eu', '-c', script, 'depfile-test',
                    temp, str(ROOT / 'scripts/m3t-package-policy.sh'), str(root / spelling)],
                    text=True, capture_output=True)
                self.assertEqual(result.returncode == 0, success, result.stderr)
                if not success:
                    self.assertIn('M3T native depfile uses a symlink alias', result.stderr)

    def test_symlinked_native_closure_rejects_before_toolchain(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'native').mkdir()
            for source in (ROOT / 'native').glob('m3t_package*'):
                if source.is_file():
                    (root / 'native' / source.name).write_bytes(source.read_bytes())
            (root / 'native/f32_tensor.c').symlink_to(ROOT / 'native/f32_tensor.c')
            script = """PROJECT_ROOT=$1
raw_private_root=$1/native/m3t_package_root.esk
raw_package_bridge=$1/native/m3t_package_bridge.c
raw_package_renames=$1/native/m3t_package_private_renames.txt
raw_public_exports=$1/native/m3t_package_public_exports.txt
raw_include_dirs=("$1/internal/p1/lib" "$1/internal/c1/lib" "$1/internal/t1/lib" "$1/src")
die() { echo "$*" >&2; exit 1; }
source "$2"
"""
            result = subprocess.run(['/usr/bin/bash', '-eu', '-c', script, 'policy-test',
                temp, str(ROOT / 'scripts/m3t-package-policy.sh')], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('M3T policy', result.stderr)

    def test_copied_partial_relative_and_reordered_tuples_reject_before_compiler(self):
        exact = [str(PREFIX) + s for s in ('_root.esk', '_bridge.c', '_private_renames.txt', '_public_exports.txt')]
        includes = [str(ROOT / p) for p in ('internal/p1/lib', 'internal/c1/lib', 'internal/t1/lib', 'src')]
        with tempfile.TemporaryDirectory() as temp:
            copied = []
            for n, src in enumerate(exact):
                path = Path(temp) / f'copy-{n}'
                path.write_bytes(Path(src).read_bytes())
                copied.append(str(path))
            # Edited copies still cannot opt into the generic E1B tuple.
            Path(copied[0]).write_text(Path(copied[0]).read_text() + '; copied root\n')
            Path(copied[1]).write_text(Path(copied[1]).read_text() + '/* copied bridge */\n')
            Path(copied[2]).write_text(Path(copied[2]).read_text() + '\n')
            Path(copied[3]).write_text(Path(copied[3]).read_text() + '\n')
            alias = Path(temp) / 'root.esk'
            alias.symlink_to(exact[0])
            cases = [(copied, includes), ([copied[0], *exact[1:]], includes),
                     ([str(alias), *exact[1:]], includes),
                     ([os.path.relpath(p, ROOT) for p in exact], includes),
                     (exact, list(reversed(includes))), (exact, includes + [temp])]
            for inputs, roots in cases:
                result = subprocess.run(['/usr/bin/bash', str(ROOT / 'scripts/build-e1b-consumer.sh'),
                    *inputs, str(Path(temp) / 'out.o'), *roots], cwd=ROOT, text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('M3T policy', result.stderr)
                self.assertFalse((Path(temp) / 'out.o').exists())


if __name__ == '__main__':
    unittest.main()
