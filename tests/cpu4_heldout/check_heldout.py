"""Check actual E3 words against the frozen synthetic held-out criterion.

Development only: no model, evaluator, training, or reference metric computation.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
import struct
import sys

WEIGHT_BITS = 0x40800000  # exact 4.0f
COUNTS = {"baseline": (0, 0, 0), "post": (1000, 250, 249)}


def f32(word: int) -> float:
    if not 0 <= word <= 0xFFFFFFFF:
        raise ValueError("metric word outside uint32")
    value = struct.unpack("<f", struct.pack("<I", word))[0]
    if not math.isfinite(value):
        raise ValueError("nonfinite E3 metric")
    return value


def check(text: str) -> tuple[int, int]:
    lines = text.splitlines()
    if len(lines) != 2 or text != "\n".join(lines) + "\n":
        raise ValueError("expected exactly two canonical observation lines")
    means = []
    words = []
    for phase, line in zip(("baseline", "post"), lines):
        fields = line.split(" ")
        if len(fields) != 11 or fields[:2] != ["CPU4-HELDOUT-OBS", phase]:
            raise ValueError("missing, duplicate, reordered or malformed phase")
        if any(re.fullmatch(r"0|[1-9][0-9]*", item) is None for item in fields[2:]):
            raise ValueError("observation requires canonical unsigned decimal integers")
        mean, weight, perplexity, accuracy, physical, batches, tokens, updates, epochs = map(int, fields[2:])
        loss, total, ppl, acc = map(f32, (mean, weight, perplexity, accuracy))
        if weight != WEIGHT_BITS or total != 4 or (physical, batches) != (4, 1):
            raise ValueError("frozen full-row weight/counts differ")
        if (tokens, updates, epochs) != COUNTS[phase]:
            raise ValueError("frozen initial/final training counters differ")
        if (phase == "baseline" and loss <= 0) or loss < 0:
            raise ValueError("baseline must be positive and post nonnegative")
        if ppl < 1 or not 0 <= acc <= 1:
            raise ValueError("perplexity or accuracy outside frozen bounds")
        means.append(loss)
        words.append(mean)
    # f32 values times these small integers are exact in binary64. Compare the
    # frozen rational 9/10 without rounding a decimal 0.90 multiplier.
    if 10 * means[1] > 9 * means[0]:
        raise ValueError("post mean exceeds frozen 0.90 baseline criterion")
    return words[0], words[1]


def main() -> None:
    before, after = check(Path(sys.argv[1]).read_text())
    print(f"CPU4-HELDOUT-CHECK-PASS baseline-bits={before} post-bits={after} "
          "relative-drop>=1/10 weight=4 physical=4 batches=1 updates=250 "
          "scope=tiny-synthetic-perturbation")


if __name__ == "__main__":
    main()
