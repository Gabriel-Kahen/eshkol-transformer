#!/usr/bin/env python3
"""Check exact real compiler/native closure for the private T1/D2 prerequisite."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PINS = {
    'native/d2_native.c': '645ec4c0166b274045c9dced0bd2ee023b3fd7d75f5c2ff1821571f12be5d08c',
    'native/d2_native.h': '165dc0184d288b5ea1eef0b3a6a4d5aee4c44e7a2acc1eb1d4d46908206163e8',
}


def dependencies(path):
    return [Path(value).resolve() for value in path.read_text().replace('\\\n', ' ').split(':', 1)[1].split()]


def same(actual, expected, label):
    if actual != expected:
        raise ValueError(f'{label} closure mismatch: actual={actual}, expected={expected}')


def check(repo, work, repetition):
    expected_dir = repo / 'tests/e3_d2'
    expected = []
    for entry in (expected_dir / 'expected_source_closure.txt').read_text().splitlines():
        expected.append(work / entry[6:] if entry.startswith('@work/') else repo / entry)
    actual = dependencies(work / 'identity.d')
    same(actual, expected, 'Eshkol source')
    native = json.loads((expected_dir / 'expected_native_closure.json').read_text())
    same(sorted(path.stem for path in (work / 'native').glob('*.d')), sorted(native), 'native objects')
    for name, paths in native.items():
        same(dependencies(work / 'native' / f'{name}.d'), [repo / path for path in paths], name)
    same(dependencies(work / 'e3_d2_native.d'),
         [repo / path for path in native['e3_d2_native']], 'production variant')
    same(dependencies(work / 'd2_native.d'),
         [repo / path for path in native['e3_d2_native']
          if path not in {'native/e3_d2_native.c', 'native/e3_d2_native.h'}], 'original D2')
    for source, digest in PINS.items():
        same(hashlib.sha256((repo / source).read_bytes()).hexdigest(), digest, source)
    members = subprocess.check_output(['ar', 't', work / 'native/libe3_d2_test_runtime.a'], text=True).splitlines()
    same(members, [name + '.o' for name in sorted(native)], 'archive members')
    symbols = subprocess.check_output(['nm', '-a', work / f'identity-{repetition}'], text=True)
    (work / f'identity-{repetition}.symbols').write_text(symbols)
    if 't2-private-tokenizer' in symbols or 't2_private_tokenizer' in symbols:
        raise ValueError('T2 authority in prerequisite executable')
    # Preserve normalized, complete inventories beside the raw depfiles.
    (work / 'source-closure.txt').write_text('\n'.join(str(path) for path in actual) + '\n')
    print('E3 D2 exact compiler/native closure and predecessor pins PASS')


if __name__ == '__main__':
    check(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3])
