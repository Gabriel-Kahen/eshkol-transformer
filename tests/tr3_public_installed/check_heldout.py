"""Strict checker for the installed finite-D2 held-out learning witness."""

from __future__ import annotations

import math
from pathlib import Path
import struct
import sys


MIN_DROP_F32_BITS = 0x3E800000  # 0.25f; frozen before observing this caller.
WEIGHT_F32_BITS = 0x40400000  # 3.0f, the two held-out mask weights [2, 1].


def f32(word: int) -> float:
    if not 0 <= word <= 0xFFFFFFFF:
        raise ValueError("f32 word out of range")
    value = struct.unpack(">f", struct.pack(">I", word))[0]
    if not math.isfinite(value):
        raise ValueError("nonfinite f32 metric")
    return value


def check(output: str) -> tuple[int, int]:
    lines = output.splitlines()
    if len(lines) != 5 or lines[-1] != "TR3-HELDOUT-AOT-PASS":
        raise ValueError("missing or extra witness records")
    observations = []
    for ordinal, index in ((0, 0), (1, 2), (0, 3)):
        fields = lines[index].split()
        if len(fields) != 6 or fields[:2] != ["TR3-HELDOUT-OBS", str(ordinal)]:
            raise ValueError("malformed observation")
        try:
            loss, weight, tokens, batches = map(int, fields[2:])
        except ValueError as exc:
            raise ValueError("noninteger observation") from exc
        if weight != WEIGHT_F32_BITS or (tokens, batches) != (3, 2):
            raise ValueError("held-out weight or count drift")
        if f32(loss) <= 0:
            raise ValueError("nonpositive held-out loss")
        observations.append(loss)
    if lines[1] != "TR3-HELDOUT-TRAIN 32 64 31":
        raise ValueError("missing fixed genuine update count")
    before, after, retained = observations
    if retained != before:
        raise ValueError("earlier public metric changed after training")
    drop = f32(before) - f32(after)
    if not math.isfinite(drop) or drop < f32(MIN_DROP_F32_BITS):
        raise ValueError("held-out loss did not clear predeclared f32 threshold")
    return before, after


def main() -> None:
    before, after = check(Path(sys.argv[1]).read_text())
    print(f"TR3-HELDOUT-CHECK-PASS before-bits={before} after-bits={after} "
          "weight-bits=1077936128 tokens=3 batches=2 updates=32 "
          "train-tokens=64 epochs=31 minimum-drop-f32-bits=1048576000")


if __name__ == "__main__":
    main()
