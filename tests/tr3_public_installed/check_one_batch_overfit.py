"""Predeclared quantitative gate for the installed one-batch TR3 witness."""

from __future__ import annotations

import math
from pathlib import Path
import struct
import sys


UPDATES = 32
MIN_LOSS_REDUCTION = 0.25


def f32(word: int) -> float:
    if not 0 < word < 0x7F800000:
        raise AssertionError(f"loss is not finite positive binary32: {word}")
    return struct.unpack("<f", struct.pack("<I", word))[0]


def check(lines: list[str]) -> tuple[float, float]:
    if len(lines) != 4:
        raise AssertionError(f"expected four public observations, got {len(lines)}")
    observations = []
    for line, label in zip(lines[:2] + lines[3:], ("before", "repeat", "after")):
        fields = line.split()
        if len(fields) != 5 or fields[:2] != ["TR3-OVERFIT-OBS", label]:
            raise AssertionError(f"unexpected observation: {line}")
        word, tokens, batches = map(int, fields[2:])
        if (tokens, batches) != (1, 1):
            raise AssertionError("fixture must emit exactly one masked-token batch")
        observations.append(word)
    fields = lines[2].split()
    if fields != ["TR3-OVERFIT-TRAIN", str(UPDATES), str(UPDATES)]:
        raise AssertionError("trainer did not report 32 authentic updates and tokens")
    before, repeated, after = observations
    if repeated != before:
        raise AssertionError("same-model evaluation changed without training")
    baseline, final = f32(before), f32(after)
    if not math.isfinite(final) or final > baseline * (1 - MIN_LOSS_REDUCTION):
        raise AssertionError(
            f"one-batch loss reduction below {MIN_LOSS_REDUCTION:.0%}: "
            f"{baseline:.9g} -> {final:.9g}"
        )
    return baseline, final


def main() -> None:
    baseline, final = check(Path(sys.argv[1]).read_text().splitlines())
    print(f"TR3-ONE-BATCH-OVERFIT-PASS initial={baseline:.9g} final={final:.9g}")


if __name__ == "__main__":
    main()
