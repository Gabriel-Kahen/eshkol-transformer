"""Accepted G3-S/Philox oracle, applied to generated replay traces by custody."""
from __future__ import annotations

import math
from tests.g3s.reference import bits, f32, literal, philox, word


def check_trace(data: bytes) -> None:
    rows = data.decode().splitlines()
    pre = [line.split() for line in rows if line.startswith('G3R_ORACLE_PRE ')]
    post = [line.split() for line in rows if line.startswith('G3R_ORACLE_POST ')]
    if len(pre) != 1 or len(post) != 1:
        raise ValueError('one actual pre/post pair required')
    p, q = pre[0], post[0]
    if len(p) != 261 or p[:5] != ['G3R_ORACLE_PRE', 'categorical', '1729', '0', '0']:
        raise ValueError('P1 oracle header/shape')
    if len(q) != 10 or q[0] != 'G3R_ORACLE_POST':
        raise ValueError('output oracle shape')
    encoded = list(map(int, p[5:]))
    if not all(0 <= n <= 0xffffffff for n in encoded):
        raise ValueError('logit bits')
    logits = [word(n) for n in encoded]
    if not all(math.isfinite(f) and bits(f) == n for f, n in zip(logits, encoded)):
        raise ValueError('finite binary32 logits required')
    uniform = f32((philox(1729, 0, 0)[0] >> 8) * 2**-24)
    expected = literal(logits, 1.0, 256, 1.0, uniform)['token']
    if int(q[1]) != expected:
        raise ValueError('sampled ID')
    values = list(map(int, q[2:]))
    if values[::2] != [1, 1729, 1, 0] or values[1::2] != [1, 1729, 1, 0]:
        raise ValueError('native/output successor RNG')
