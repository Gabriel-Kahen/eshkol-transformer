#!/usr/bin/env python3
"""Independent G3-S/Philox check of the C4 P2 last-logit witness."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "g3s"))
import reference  # development-only oracle; never linked into production


def check(path: Path) -> None:
    records = [line.split() for line in path.read_text().splitlines()
               if line.startswith("ORACLE ")]
    assert len(records) == 2, f"expected two oracle rows, got {len(records)}"
    assert [row[1] for row in records] == ["greedy", "categorical"]
    for row in records:
        assert len(row) == 267, f"{row[1]}: truncated last-logit row"
        mode, token = row[1], int(row[2])
        before = tuple(map(int, row[3:7]))
        after = tuple(map(int, row[7:11]))
        logits = [reference.word(int(bits, 16)) for bits in row[11:]]
        assert len(logits) == 256
        if mode == "greedy":
            assert before == (1, 7, 0, 0)
            expected = max(range(256), key=lambda index: (logits[index], -index))
            assert after == before, "greedy changed the successor RNG"
        else:
            assert before == (1, 1729, 0, 0)
            lanes = reference.philox(1729, 0, 0)
            uniform = reference.f32((lanes[0] >> 8) * 2**-24)
            expected = reference.literal(logits, 1.0, 17, 0.5, uniform)["token"]
            assert after == (1, 1729, 1, 0), "categorical RNG did not advance once"
        assert token == expected, f"{mode}: sampled {token}, oracle {expected}"
    print("G3-C4 P2/G1 independent G3-S/Philox oracle PASS: modes=2 logits=256")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_p2_g1_oracle.py NATIVE_STDOUT")
    check(Path(sys.argv[1]))
