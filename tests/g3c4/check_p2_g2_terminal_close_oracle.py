#!/usr/bin/env python3
"""Independent Philox/selection check of actual private Eshkol terminal rows.

The C witness also compares complete committed K/V and binding snapshots to
separate P3/T4 native forwards. This development oracle never enters runtime.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "g3s"))
import reference

MASK = (1 << 64) - 1
PROMPTS = {(0, 255), (7, 11)}
MODES = {"greedy": 7, "categorical": 1729}


def successor(before: tuple[int, ...], mode: str,
              logits: list[float]) -> tuple[int, tuple[int, ...]]:
    assert len(before) == 4 and before[0] == 1
    if mode == "greedy":
        return max(range(256), key=lambda i: (logits[i], -i)), before
    assert mode == "categorical"
    lo, hi = before[2] & MASK, before[3] & MASK
    assert (lo, hi) != (MASK, MASK), "exhausted draw cannot succeed"
    lanes = reference.philox(before[1], lo, hi)
    uniform = reference.f32((lanes[0] >> 8) * 2**-24)
    selected = reference.literal(logits, 1.0, 17, 0.5, uniform)["token"]
    lo = (lo + 1) & MASK
    hi = (hi + (lo == 0)) & MASK
    signed = lambda word: word if word < 1 << 63 else word - (1 << 64)
    return selected, (1, before[1], signed(lo), signed(hi))


def check_lines(lines: list[str]) -> int:
    seen: set[tuple[str, tuple[int, int], str]] = set()
    pending: list[list[str]] = []
    cases = 0
    second_eos = 0
    for line in lines:
        if line.startswith("TERM_ORACLE "):
            pending.append(line.split())
            continue
        if not line.startswith("TERM_CACHE "):
            continue
        row = line.split()
        assert len(row) == 9, "truncated terminal cache witness"
        mode = row[1]
        prompt = (int(row[2]), int(row[3]))
        count, cache, first, second, eos = map(int, row[4:9])
        assert mode in MODES and prompt in PROMPTS
        assert count in (1, 2) and cache == count + 2
        assert 0 <= first < 256
        assert second == 0 if count == 1 else 0 <= second < 256
        assert len(pending) == count, "missing or extra genuine draw"
        first_after = None
        for draw_index, draw in enumerate(pending):
            assert len(draw) == 270, "truncated 256-logit draw"
            assert draw[1] == mode and tuple(map(int, draw[2:4])) == prompt
            assert draw[4] == ("first" if draw_index == 0 else "second")
            selected = int(draw[5])
            before = tuple(map(int, draw[6:10]))
            after = tuple(map(int, draw[10:14]))
            logits = [reference.word(int(bits, 16)) for bits in draw[14:]]
            assert len(logits) == 256
            assert before == ((1, MODES[mode], 0, 0) if draw_index == 0
                              else first_after), "wrong entry/second RNG"
            expected, expected_after = successor(before, mode, logits)
            assert (selected, after) == (expected, expected_after), (
                mode, prompt, draw_index, selected, expected, after,
                expected_after)
            assert selected == (first if draw_index == 0 else second)
            if draw_index == 0:
                first_after = after
        if count == 1:
            assert eos == first, "first EOS was not the selected first ID"
            branch = "first-eos"
        else:
            assert eos != first, "second draw after first EOS"
            branch = "second-eos" if eos == second else "non-eos"
            second_eos += branch == "second-eos"
        assert (mode, prompt, branch) not in seen, "duplicate matrix case"
        seen.add((mode, prompt, branch))
        pending.clear()
        cases += 1
    assert not pending, "draw without terminal/cache witness"
    required = {(mode, prompt, branch)
                for mode in MODES for prompt in PROMPTS
                for branch in ("first-eos", "non-eos")}
    assert required <= seen, f"missing matrix cases: {required - seen}"
    assert second_eos > 0, "no genuine second EOS"
    assert cases == len(seen)
    return cases


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_p2_g2_terminal_close_oracle.py RUN_STDOUT")
    count = check_lines(Path(sys.argv[1]).read_text().splitlines())
    print(f"P2/G2 terminal independent Philox/P3/T4 oracle PASS: cases={count}")
