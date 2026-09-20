"""Exact, source-tree-only L3S archive and compiler-input audit."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def output(*args: str) -> str:
    return subprocess.check_output(args, text=True)


def manifest(name: str) -> list[str]:
    return (ROOT / 'native' / f'l3s_{name}.txt').read_text().splitlines()


def check(artifact: Path) -> None:
    obj = artifact / 'l3s_masked_objective_provider.o'
    archive = artifact / 'libeshkol_transformer_l3s.a'
    assert output('ar', 't', str(archive)).splitlines() == manifest('archive_members')
    symbols = output('nm', '-g', '--defined-only', '--format=posix', str(obj))
    assert sorted(line.split()[0] for line in symbols.splitlines()) == manifest('public_exports'), symbols
    undefined = output('nm', '-u', '--format=posix', str(obj))
    assert sorted({line.split()[0] for line in undefined.splitlines()}) == manifest('undefined_symbols'), undefined
    depfile = (artifact / 'l3s_masked_objective_provider.d').read_text().replace('\\\n', '')
    target, inputs = depfile.split(':', 1)
    assert target == 'l3s_masked_objective_provider.o'
    closure = sorted({Path(name).resolve().relative_to(ROOT).as_posix() for name in shlex.split(inputs)})
    assert closure == manifest('compile_closure'), closure
    actual = []
    for directory in ('include', 'native'):
        for path in (ROOT / directory).rglob('*'):
            if path.suffix not in ('.h', '.c', '.esk'):
                continue
            if re.search(r'ET_L3S_MASKED_OBJECTIVE_ABI|et_l3s_kernel_provider_v1', path.read_text()):
                actual.append(path.relative_to(ROOT).as_posix())
    assert sorted(actual) == manifest('source_closure'), actual
    for name in manifest('source_closure'):
        source = (ROOT / name).read_text()
        assert not re.search(r'\b(python|pytorch|torch|dlopen|dlsym|getenv)\b|finite[-_ ]difference|scalar[-_ ]fallback', source, re.I), name
    for line in (ROOT / 'native/l3s_predecessor_sources.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    # Independently extract the archive member; an external stale .o is no proof.
    member = subprocess.check_output(['ar', 'p', str(archive), manifest('archive_members')[0]])
    assert member == obj.read_bytes(), 'archive member differs from audited normal object'
    print('L3S package PASS: exact archive, exports, undefined symbols, compiler/source closure and immutable predecessors')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('artifact', type=Path)
    check(parser.parse_args().artifact.resolve())
