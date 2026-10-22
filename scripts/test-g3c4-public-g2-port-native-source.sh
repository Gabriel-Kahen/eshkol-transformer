#!/usr/bin/env bash
# Source-only C11/feature isolation. No objects, runtime or sanitizer admission.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python3 - "$root" <<'PY'
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile
root = Path(sys.argv[1])
source = (root / 'scripts/test-g3c4-p2g2-typed-result.sh').read_text()
macros = re.search(r'owner_macros=\((.*?)\)\n', source, re.S).group(1).split()
flags = ['-std=c11', '-Wall', '-Wextra', '-Werror', '-Wpedantic']
flags += ['-I' + str(root / name) for name in ('include', 'native', 'src', 'src/eshkol_transformer')]
flags += ['-D' + name for name in ('ET_I64_TENSOR_TESTING', 'ET_I64_TENSOR_STORAGE_QUERY_PRIVATE',
    'ET_F32_TENSOR_STORAGE_QUERY_PRIVATE', 'ET_A2_KV_CACHE_TESTING',
    'ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE', 'ET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE',
    'ET_G3C4_T1_I1_EXACT_PAIR_PRIVATE')]
compiler = os.environ.get('CC', 'clang')
def run(path, extra):
    return subprocess.run([compiler, *flags, *extra, str(path)], cwd=root,
                          capture_output=True, check=False)
owner = root / 'src/eshkol_transformer/g3c4_model_owner.c'
enabled = [*macros, '-DET_G3C4_P2_G2_TYPED_RESULT_PRIVATE']
for name in ('src/eshkol_transformer/g3c4_model_owner.c', 'tests/g3c4/test_p2_g2_typed_result.c'):
    result = run(root / name, [*enabled, '-fsyntax-only'])
    assert result.returncode == 0 and not result.stderr, result.stderr.decode()
with tempfile.TemporaryDirectory() as temporary:
    old = Path(temporary) / 'g3c4_model_owner.c'
    old.write_bytes(subprocess.check_output(['git', 'show', 'c71d39a:src/eshkol_transformer/g3c4_model_owner.c'], cwd=root))
    a, b = [run(path, [*macros, '-E', '-P']) for path in (old, owner)]
    assert a.returncode == b.returncode == 0 and not a.stderr and not b.stderr
    assert a.stdout == b.stdout, 'feature-off native owner changed'
    assert b'et_g3c4_private_p2g2_typed_copy_v1' not in b.stdout
    isolated = run(owner, ['-DET_G3C4_P2_G2_TYPED_RESULT_PRIVATE', '-E', '-P'])
    assert isolated.returncode != 0
    assert b'typed P2/G2 results require the logical I1 clone prerequisite' in isolated.stderr
print('G3-C4 public G2 native source PASS: 2 strict C11 fixtures, exact feature-off preprocessing, isolated-feature reject')
PY
