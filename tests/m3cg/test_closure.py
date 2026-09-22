"""Fail-closed source inventory admission, independent of the compiler."""
from pathlib import Path
import tempfile
import unittest
from tests.m3cg.check_closure import ROOT, normalized

class Closure(unittest.TestCase):
    def test_repository_and_pinned_compiler_are_only_roots(self):
        with tempfile.TemporaryDirectory() as temp:
            compiler = Path(temp) / 'compiler'
            compiler.mkdir()
            header = compiler / 'header.h'
            header.write_text('')
            self.assertEqual(normalized(str(header), compiler), '.deps/eshkol-src/header.h')
            outside = Path(temp) / 'outside.h'
            outside.write_text('')
            with self.assertRaises(ValueError):
                normalized(str(outside), compiler)
            with self.assertRaises(ValueError):
                normalized('native/f32_tensor.c', compiler)
            alias = compiler / 'alias.h'
            alias.symlink_to(header)
            with self.assertRaises(ValueError):
                normalized(str(alias), compiler)
            self.assertEqual(normalized(str(ROOT / 'native/f32_tensor.c'), compiler),
                             'native/f32_tensor.c')

    def test_complete_closures_and_mutations(self):
        from tests.m3cg.check_closure import validate
        with tempfile.TemporaryDirectory() as temp:
            compiler = Path(temp) / 'compiler'
            compiler.mkdir()
            directory = Path(temp) / 'fresh-1'
            directory.mkdir()
            source = ['tests/m3cg/private_root.esk'] + (
                ROOT / 'native/m3_package_source_closure.txt').read_text().splitlines() + [
                'native/m3_call_adapters.esk', 'tests/m3cg/witness.esk']
            native = (ROOT / 'native/m3_package_native_source_closure.txt').read_text().splitlines() + [
                'tests/m3cg/package_bridge.c', 'tests/m3cg/model_harness.c',
                'src/eshkol_transformer/m3_call_f32_integration.c',
                'src/eshkol_transformer/m3_call_pins.h']
            def absolute(name):
                if name.startswith('.deps/eshkol-src/'):
                    path = compiler / name.removeprefix('.deps/eshkol-src/')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                    return str(path)
                return str(ROOT / name)
            def write(source_rows, native_rows):
                (directory / 'private.d').write_text('private.o: ' + ' '.join(map(absolute, source_rows)) + '\n')
                (directory / 'error.d').write_text('error.o: ' + ' '.join(map(absolute, native_rows)) + '\n')
                (directory / 'bridge.d').write_text('bridge.o:\n')
            write(source, native)
            validate(directory, compiler)
            for malformed in (source[:-1], source + ['README.md'], source + source[-1:],
                              [source[1], source[0], *source[2:]]):
                with self.subTest(source=malformed):
                    write(malformed, native)
                    with self.assertRaises(ValueError):
                        validate(directory, compiler)
            for malformed in (native[:-1], native + ['README.md']):
                with self.subTest(native=malformed):
                    write(source, malformed)
                    with self.assertRaises(ValueError):
                        validate(directory, compiler)
            # Normal source and bridge cannot gain witness declarations/observers.
            normal = Path(temp) / 'fresh-normal'
            directory.rename(normal)
            directory = normal
            normal_source = ['tests/m3cg/normal_root.esk', *source[1:-1]]
            normal_native = [name for name in native if not name.startswith('tests/m3cg/')]
            write(normal_source, normal_native)
            validate(directory, compiler)
            write(normal_source + ['tests/m3cg/witness.esk'], normal_native)
            with self.assertRaises(ValueError):
                validate(directory, compiler)
