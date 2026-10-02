#!/usr/bin/env python3
"""Independent G3-S arithmetic/Philox check of genuine second-draw transcripts."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'g3s'))
import reference

MASK = (1 << 64) - 1

def signed(word):
    return word if word < 1 << 63 else word - (1 << 64)

def check(path):
    rows = [line.split() for line in path.read_text().splitlines() if line.startswith('ORACLE ')]
    assert len(rows) >= 8
    carry = final = singleton = False
    for row in rows:
        assert len(row) == 267
        mode, actual = row[1], int(row[2])
        before, after = tuple(map(int, row[3:7])), tuple(map(int, row[7:11]))
        logits = [reference.word(int(word, 16)) for word in row[11:]]
        assert before[0] == 1 and before[1] >= 0
        if mode == 'greedy':
            expected = max(range(256), key=lambda i: (logits[i], -i))
            assert after == before
        else:
            assert mode in ('categorical', 'categorical-singleton')
            lo, hi = before[2] & MASK, before[3] & MASK
            assert (lo, hi) != (MASK, MASK), 'exhausted state cannot draw'
            lanes = reference.philox(before[1], lo, hi)
            uniform = reference.f32((lanes[0] >> 8) * 2**-24)
            k = 1 if mode == 'categorical-singleton' else 17
            expected = reference.literal(logits, 1.0, k, 0.5, uniform)['token']
            lo = (lo + 1) & MASK
            hi = (hi + (lo == 0)) & MASK
            assert after == (1, before[1], signed(lo), signed(hi))
            carry |= before[2] == -1 and after[2] == 0 and after[3] == before[3] + 1
            final |= after[2:] == (-1, -1)
            singleton |= mode == 'categorical-singleton'
        assert actual == expected, (mode, actual, expected)
    assert carry and final and singleton
    print(f'G3-C4 P2/G2 independent second-draw oracle PASS: rows={len(rows)} carry/final/singleton=1')

if __name__ == '__main__':
    check(Path(sys.argv[1]))
