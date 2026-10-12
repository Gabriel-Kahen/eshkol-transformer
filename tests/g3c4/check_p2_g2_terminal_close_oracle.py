#!/usr/bin/env python3
"""Independent G3-S/Philox check of actual same-call second-frame logits."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "g3s"))
import reference  # development-only oracle

MASK = (1 << 64) - 1


def signed(word):
    return word if word < 1 << 63 else word - (1 << 64)


def check(path: Path) -> None:
    lines = path.read_text().splitlines()
    second = [line.split() for line in lines
              if line.startswith("COMPOSER_SECOND ")]
    terminal = [line.split() for line in lines
                if line.startswith("ESH_TERMINAL ")]
    assert len(second) == 6, f"expected six second-frame rows, got {len(second)}"
    assert len(terminal) == 8, f"expected eight terminal rows, got {len(terminal)}"
    modes = ("greedy", "categorical", None, "categorical-singleton",
             "categorical-carry", "categorical-final")
    entries = ((1, 7, 0, 0), (1, 1729, 1, 0), None,
               (1, 7, 1, 0), (1, 7, -1, 0), (1, 7, -2, -1))
    for index, (row, mode, entry) in enumerate(zip(second, modes, entries)):
        if index == 2:
            matching = second[0] if row[1] == "greedy" else second[1]
            assert row == matching, "second-token EOS changed authentic draw"
            continue
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
            assert before[2:] != (-1, -1), "exhausted state cannot draw"
            lanes = reference.philox(before[1], before[2] & MASK, before[3] & MASK)
            uniform = reference.f32((lanes[0] >> 8) * 2**-24)
            expected = reference.literal(logits, 1.0,
                                         1 if mode == "categorical-singleton" else 17,
                                         0.5,
                                         uniform)["token"]
            lo = (before[2] + 1) & MASK
            hi = ((before[3] & MASK) + (lo == 0)) & MASK
            expected_rng = (1, before[1], signed(lo), signed(hi))
        assert selected == expected, (
            f"{mode}: actual second ID {selected}, independent oracle {expected}")
        assert after == expected_rng, f"{mode}: wrong second-frame successor RNG"
        if index < 2:
            g2, g1 = terminal[2 * index:2 * index + 2]
            assert g2 == ["ESH_TERMINAL", mode, "2", str(first), str(selected), "4"]
            assert g1 == ["ESH_TERMINAL", mode, "1", str(first), "0", "3"]
        else:
            assert index >= 3
            assert terminal[index + 2] == [
                "ESH_TERMINAL", "categorical", "2", str(first), str(selected), "4"]
    terminal_eos = second[2]
    matching = second[0] if terminal_eos[1] == "greedy" else second[1]
    assert terminal_eos == matching, "second-token EOS changed the authentic draw"
    assert terminal[4] == ["ESH_TERMINAL", matching[1], "2",
                           matching[2], matching[3], "4"]
    assert second[4][4:8] == ["1", "7", "-1", "0"]
    assert second[4][8:12] == ["1", "7", "0", "1"]
    assert second[5][8:12] == ["1", "7", "-1", "-1"]
    print("G3-C4 P2/G2 terminal independent second-frame oracle PASS: "
          "rows=6 logits=256 terminal=8 second-eos/singleton/carry/final=1")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_p2_g2_terminal_close_oracle.py RUN_STDOUT")
    check(Path(sys.argv[1]))
