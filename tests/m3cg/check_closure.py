"""Reject every dependency outside the fixed private witness input inventory."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]

def dependencies(path):
    return path.read_text().replace('\\\n', ' ').split(':', 1)[1].split()

def normalized(raw, compiler):
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f'nonabsolute dependency: {raw}')
    current = Path('/')
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f'symlink dependency: {raw}')
    path = path.resolve(strict=True)
    if path.is_relative_to(ROOT):
        return str(path.relative_to(ROOT))
    if path.is_relative_to(compiler):
        return '.deps/eshkol-src/' + str(path.relative_to(compiler))
    raise ValueError(f'outside dependency: {raw}')

def validate(directory, compiler):
    normal = directory.name == 'fresh-normal'
    source = [normalized(x, compiler) for x in dependencies(directory / 'private.d')]
    expected = (ROOT / 'native/m3_package_source_closure.txt').read_text().splitlines()
    expected = [('tests/m3cg/normal_root.esk' if normal else 'tests/m3cg/private_root.esk')] + expected + ['native/m3_call_adapters.esk']
    if not normal:
        expected.append('tests/m3cg/witness.esk')
    if source != expected:
        raise ValueError(f'Eshkol closure drift: {source}')
    native = set()
    for depfile in [directory / 'error.d', directory / 'bridge.d',
                    *sorted(directory.glob('native-*.d'))]:
        native.update(normalized(x, compiler) for x in dependencies(depfile))
    expected_native = set((ROOT / 'native/m3_package_native_source_closure.txt').read_text().splitlines())
    if not normal:
        expected_native.update(['tests/m3cg/package_bridge.c', 'tests/m3cg/model_harness.c'])
    expected_native.update(['src/eshkol_transformer/m3_call_f32_integration.c',
        'src/eshkol_transformer/m3_call_pins.h'])
    if native != expected_native:
        raise ValueError(f'native closure missing={sorted(expected_native-native)}, extra={sorted(native-expected_native)}')
    (directory / 'source-closure.txt').write_text('\n'.join(source) + '\n')
    (directory / 'native-closure.txt').write_text('\n'.join(sorted(native)) + '\n')

if __name__ == '__main__':
    validate(Path(sys.argv[1]), Path(sys.argv[2]).resolve())
