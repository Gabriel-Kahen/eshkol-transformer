#!/usr/bin/env python3
"""Independent G3-S/Philox oracle for the two actual Eshkol P2/G2 calls."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "g3s"))
import reference  # development-only oracle, never linked into generation


def check(path: Path) -> None:
    rows = [line.split() for line in path.read_text().splitlines()
            if line.startswith("COORD_ORACLE ")]
    assert len(rows) == 2, f"expected two coordinator rows, got {len(rows)}"
    expected_cases = (("greedy", (0, 255), (1065353216, 256, 1065353216, -1),
                       (1, 7, 0, 0)),
                      ("categorical", (7, 11), (1065353216, 17, 1056964608, -1),
                       (1, 1729, 0, 0)))
    for row, (mode, prompt, policy, entry) in zip(rows, expected_cases):
        assert len(row) == 273, f"{mode}: truncated last-logit row"
        assert row[1] == mode, f"expected {mode} row"
        assert tuple(map(int, row[2:4])) == prompt, f"{mode}: wrong P2 prompt"
        assert tuple(map(int, row[4:8])) == policy, f"{mode}: wrong policy"
        selected = int(row[8])
        before = tuple(map(int, row[9:13]))
        after = tuple(map(int, row[13:17]))
        assert before == entry, f"{mode}: wrong entry RNG"
        logits = [reference.word(int(bits, 16)) for bits in row[17:]]
        assert len(logits) == 256
        if mode == "greedy":
            expected_id = max(range(256),
                              key=lambda index: (logits[index], -index))
            expected_rng = before
        else:
            lanes = reference.philox(before[1], before[2], before[3])
            uniform = reference.f32((lanes[0] >> 8) * 2**-24)
            expected_id = reference.literal(logits, 1.0, 17, 0.5,
                                            uniform)["token"]
            expected_rng = (1, 1729, 1, 0)
        assert selected == expected_id, (
            f"{mode}: coordinator selected {selected}, oracle {expected_id}")
        assert after == expected_rng, (
            f"{mode}: successor RNG {after}, oracle {expected_rng}")
    print("G3-C4 P2/G2 T1 coordinator independent G3-S/Philox oracle PASS: "
          "modes=2 logits=256")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_p2_g2_t1_prefix_oracle.py RUN_STDOUT")
    check(Path(sys.argv[1]))
