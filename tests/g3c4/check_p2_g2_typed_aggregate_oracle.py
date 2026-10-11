#!/usr/bin/env python3
"""Independent G3-S/Philox check of one real typed aggregate call per case."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "g3s"))
import reference


MASK = (1 << 64) - 1


def signed(word: int) -> int:
    return word if word < 1 << 63 else word - (1 << 64)


def check(path: Path) -> None:
    rows = {}
    for line in path.read_text().splitlines():
        if not line.startswith("AGG_ORACLE "):
            continue
        fields = line.split()
        assert len(fields) == 273, f"truncated oracle row: {len(fields)}"
        case, step = map(int, fields[1:3])
        mode = fields[3]
        temp_bits, top_k, top_p_bits, eos, actual = map(int, fields[4:9])
        before = tuple(map(int, fields[9:13]))
        after = tuple(map(int, fields[13:17]))
        logits = [reference.word(int(bits, 16)) for bits in fields[17:]]
        assert case in (0, 1, 2, 3) and step in (0, 1)
        assert (case, step) not in rows
        assert before[0] == 1 and before[1] >= 0
        if mode == "greedy":
            expected = max(range(256), key=lambda i: (logits[i], -i))
            expected_rng = before
        else:
            assert mode == "categorical"
            lo, hi = before[2] & MASK, before[3] & MASK
            assert (lo, hi) != (MASK, MASK)
            lanes = reference.philox(before[1], lo, hi)
            uniform = reference.f32((lanes[0] >> 8) * 2**-24)
            expected = reference.literal(
                logits, reference.word(temp_bits), top_k,
                reference.word(top_p_bits), uniform)["token"]
            lo = (lo + 1) & MASK
            hi = (hi + (lo == 0)) & MASK
            expected_rng = (1, before[1], signed(lo), signed(hi))
        assert actual == expected, (case, step, actual, expected)
        assert after == expected_rng, (case, step, after, expected_rng)
        rows[case, step] = (actual, before, after, eos, mode)

    assert set(rows) == {(0, 0), (0, 1), (1, 0), (2, 0), (2, 1),
                         (3, 0), (3, 1)}
    assert rows[3, 0][4] == rows[3, 1][4] == "categorical"
    assert rows[0, 0][0] != rows[0, 1][0]
    assert rows[1, 0][0] == rows[0, 0][0] == rows[1, 0][3]
    assert rows[2, 0][0] == rows[0, 0][0]
    assert rows[2, 1][0] == rows[0, 1][0] == rows[2, 1][3]
    assert rows[0, 0][3] == -1
    assert rows[0, 1][3] == -1
    assert rows[0, 0][4] == rows[1, 0][4] == rows[2, 0][4]
    for case in (0, 2, 3):
        assert rows[case, 1][1] == rows[case, 0][2]
    counts = {}
    for line in path.read_text().splitlines():
        if line.startswith("AGG_A2 "):
            fields = line.split()
            assert len(fields) == 3
            case, count = map(int, fields[1:])
            assert case not in counts
            counts[case] = count
    assert counts == {0: 2, 1: 1, 2: 2, 3: 2}
    print("P2/G2 typed aggregate independent RNG/A2 oracle PASS: cases=4 draws=7")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_p2_g2_typed_aggregate_oracle.py RUN_STDOUT")
    check(Path(sys.argv[1]))
