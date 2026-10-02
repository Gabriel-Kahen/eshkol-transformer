#!/usr/bin/env python3
"""Independent G3-S/Philox check of actual same-call second-frame logits."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "g3s"))
import reference  # development-only oracle


def check(path: Path) -> None:
    lines = path.read_text().splitlines()
    second = [line.split() for line in lines
              if line.startswith("COMPOSER_SECOND ")]
    terminal = [line.split() for line in lines
                if line.startswith("ESH_TERMINAL ")]
    assert len(second) == 3, f"expected three second-frame rows, got {len(second)}"
    assert len(terminal) == 5, f"expected five terminal rows, got {len(terminal)}"
    for index, (row, mode, entry) in enumerate(zip(
            second[:2], ("greedy", "categorical"), ((1, 7, 0, 0),
                                                      (1, 1729, 1, 0)))):
        assert len(row) == 268, f"{mode}: truncated second-logit row"
        assert row[1] == mode, f"{mode}: wrong second-frame mode"
        first, selected = map(int, row[2:4])
        before = tuple(map(int, row[4:8]))
        after = tuple(map(int, row[8:12]))
        assert before == entry, f"{mode}: wrong first-frame successor RNG"
        logits = [reference.word(int(bits, 16)) for bits in row[12:]]
        assert len(logits) == 256
        if mode == "greedy":
            expected = max(range(256), key=lambda token: (logits[token], -token))
            expected_rng = before
        else:
            lanes = reference.philox(before[1], before[2], before[3])
            uniform = reference.f32((lanes[0] >> 8) * 2**-24)
            expected = reference.literal(logits, 1.0, 17, 0.5,
                                         uniform)["token"]
            expected_rng = (1, 1729, 2, 0)
        assert selected == expected, (
            f"{mode}: actual second ID {selected}, independent oracle {expected}")
        assert after == expected_rng, f"{mode}: wrong second-frame successor RNG"
        g2, g1 = terminal[2 * index:2 * index + 2]
        assert g2 == ["ESH_TERMINAL", mode, "2", str(first), str(selected), "4"]
        assert g1 == ["ESH_TERMINAL", mode, "1", str(first), "0", "3"]
    terminal_eos = second[2]
    matching = second[0] if terminal_eos[1] == "greedy" else second[1]
    assert terminal_eos == matching, "second-token EOS changed the authentic draw"
    assert terminal[4] == ["ESH_TERMINAL", matching[1], "2",
                           matching[2], matching[3], "4"]
    print("G3-C4 P2/G2 terminal independent second-frame oracle PASS: "
          "modes=2 logits=256 terminal=5 second-eos=1")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_p2_g2_terminal_composer_oracle.py RUN_STDOUT")
    check(Path(sys.argv[1]))
