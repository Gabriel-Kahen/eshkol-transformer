"""Prove the historical L3S F32 translation unit survives current feature-off source."""

import hashlib
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
HISTORIC = 'c4b3d935^:native/f32_tensor.c'
HISTORIC_SHA256 = 'c35e5d3d8999d0927b2111ebd0cbceb3d517b77171141e5a68dae76f18803a47'


def preprocessed(cc, source, testing):
    command = [str(cc), '-E', '-P', '-x', 'c', '-std=c11',
               '-I', str(ROOT / 'include'), '-I', str(ROOT / 'native')]
    if testing:
        command.append('-DET_F32_TENSOR_TESTING')
    output = subprocess.run(command + ['-'], input=source, stdout=subprocess.PIPE,
                            check=True, cwd=ROOT).stdout
    return output.lstrip(b'\n')


def check(cc):
    historic = subprocess.check_output(['git', 'show', HISTORIC], cwd=ROOT)
    if hashlib.sha256(historic).hexdigest() != HISTORIC_SHA256:
        raise AssertionError('historical L3S F32 source digest differs')
    current = (ROOT / 'native/f32_tensor.c').read_bytes()
    for testing in (False, True):
        if preprocessed(cc, historic, testing) != preprocessed(cc, current, testing):
            raise AssertionError(f'L3S F32 preprocessed lineage differs: testing={testing}')


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        raise SystemExit('usage: check_f32_lineage.py PINNED_CC')
    check(sys.argv[1])
    print('L3S F32 lineage: normal and sanitizer-test preprocessing equal')
