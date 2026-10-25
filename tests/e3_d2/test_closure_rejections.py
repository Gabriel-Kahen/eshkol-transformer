#!/usr/bin/env python3
"""Mutate recorded depfiles to prove the exact-closure gate rejects additions."""
from pathlib import Path
import subprocess
import sys

repo, work = (Path(value).resolve() for value in sys.argv[1:3])
source = work / 'identity.d'
native = work / 'native/e3_d2_native.d'
checkpoint = work / 'native/checkpoint_io.d'
source_bytes = source.read_bytes()
native_bytes = native.read_bytes()
checkpoint_bytes = checkpoint.read_bytes()
resource_header = str(repo / 'native/cpuv1_resource_private.h').encode()
assert native_bytes.count(resource_header) == 1
assert checkpoint_bytes.count(resource_header) == 1
mutations = [
    (source, source_bytes + f' {repo}/native/m3_package_root.esk\n'.encode()),
    (source, source_bytes + f' {repo}/native/unknown_shared_frame.esk\n'.encode()),
    (source, source_bytes.replace(str(work / 'variant-a/source/e3_d2_dataset.esk').encode(),
                                  str(repo / 'internal/d2/lib/d2_dataset.esk').encode())),
    (source, source_bytes.replace(str(repo / 'internal/t1/lib/t1_tokenizer_core.esk').encode(), b'')),
    (native, native_bytes + f' {repo}/native/unexpected.h\n'.encode()),
    (native, native_bytes.replace(resource_header, b'')),
    (checkpoint, checkpoint_bytes.replace(resource_header, b'')),
]
try:
    for index, (path, data) in enumerate(mutations):
        path.write_bytes(data)
        result = subprocess.run([sys.executable, repo / 'tests/e3_d2/check_identity_closure.py',
                                 repo, work, 'a'], capture_output=True, text=True)
        if result.returncode == 0 or 'closure mismatch' not in result.stderr:
            raise AssertionError(f'closure mutation {index} did not fail for the intended reason')
        source.write_bytes(source_bytes)
        native.write_bytes(native_bytes)
        checkpoint.write_bytes(checkpoint_bytes)
finally:
    source.write_bytes(source_bytes)
    native.write_bytes(native_bytes)
    checkpoint.write_bytes(checkpoint_bytes)
print('E3 D2 closure mutation PASS: 7 missing/unexpected/predecessor source checks')
