#!/usr/bin/env python3
"""Check facade-produced public values against fresh P2 logits and Philox."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "g3s"))
import reference  # development-only, never linked into generation

MASK = (1 << 64) - 1


def draw(logits, mode, policy, rng):
    if mode == "greedy":
        return max(range(256), key=lambda token: (logits[token], -token)), rng
    assert mode == "categorical"
    assert rng[2:] != (-1, -1)
    lanes = reference.philox(rng[1], rng[2] & MASK, rng[3] & MASK)
    uniform = reference.f32((lanes[0] >> 8) * 2**-24)
    token = reference.literal(logits, 1.0, policy[1], 0.5, uniform)["token"]
    low = ((rng[2] & MASK) + 1) & MASK
    high = ((rng[3] & MASK) + (low == 0)) & MASK
    signed = lambda word: word if word < 1 << 63 else word - (1 << 64)
    return token, (1, rng[1], signed(low), signed(high))


def check(path: Path):
    rows = [line.split() for line in path.read_text().splitlines()]
    first = [row for row in rows if row and row[0] == "PUBLIC_FIRST"]
    second = [row for row in rows if row and row[0] == "COMPOSER_SECOND"]
    cases = [row for row in rows if row and row[0] == "PUBLIC_CASE"]
    assert len(cases) >= 4 and len(first) == len(cases)
    assert sum(int(row[4]) == 2 for row in cases) == len(second)
    saw_first_eos = saw_second_eos = saw_greedy = saw_categorical = False
    second_index = 0
    for first_row, case in zip(first, cases):
        assert len(first_row) == 273, "fresh P2 logits must have 256 words"
        assert len(case) == 14, "public row must include four final RNG words"
        mode, seed, eos = case[1], int(case[2]), int(case[3])
        count, id0, id1, cache, text0, text1 = map(int, case[4:10])
        final_rng = tuple(map(int, case[10:14]))
        assert first_row[1] == mode
        assert tuple(map(int, first_row[2:4])) == (0, 255)
        policy = tuple(map(int, first_row[4:8]))
        assert policy == (1065353216, 256 if mode == "greedy" else 17,
                          1065353216 if mode == "greedy" else 1056964608,
                          eos)
        assert int(first_row[8]) == id0
        entry = tuple(map(int, first_row[9:13]))
        assert entry == (1, seed, 0, 0)
        logits = [reference.word(int(bits, 16)) for bits in first_row[17:]]
        assert len(logits) == 256
        expected0, successor = draw(logits, mode, policy, entry)
        assert id0 == expected0 and text0 == id0
        if mode == "greedy":
            saw_greedy = True
        else:
            saw_categorical = True
        if count == 1:
            saw_first_eos = True
            assert eos == id0 and id1 == 0 and cache == 3 and text1 == -1
            assert final_rng == successor
            assert tuple(map(int, first_row[13:17])) == successor
            continue
        assert count == 2 and cache == 4 and text1 == id1 and eos != id0
        row = second[second_index]
        second_index += 1
        assert len(row) == 268 and row[1] == mode
        assert tuple(map(int, row[2:4])) == (id0, id1)
        assert tuple(map(int, row[4:8])) == successor
        second_logits = [reference.word(int(bits, 16)) for bits in row[12:]]
        assert len(second_logits) == 256
        expected1, final = draw(second_logits, mode, policy, successor)
        assert id1 == expected1 and final_rng == final
        assert tuple(map(int, row[8:12])) == final
        assert tuple(map(int, first_row[13:17])) == final
        if eos == id1 and id0 != id1:
            saw_second_eos = True
    assert second_index == len(second)
    assert saw_first_eos and saw_second_eos and saw_greedy and saw_categorical
    print("G3-C4 public G2 independent Philox/T1 oracle PASS: "
          f"cases={len(cases)} first={len(first)} second={len(second)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_p2_g2_public_facade_harness_oracle.py RUN_STDOUT")
    check(Path(sys.argv[1]))
