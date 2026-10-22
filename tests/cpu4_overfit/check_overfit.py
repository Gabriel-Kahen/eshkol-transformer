"""Development-only validation of actual Eshkol CE words; no training code."""
from __future__ import annotations
import math
from pathlib import Path
import struct
import sys

UPDATES = 250
SAMPLES = (1, 2, 10, 25, 50, 100, 150, 200, 250)
MIN_REDUCTION = 0.75
MAX_FINAL_MEAN = 1.0
MAX_F32_RETIRED_BYTES = 64 * 1024 * 1024
MAX_O2_RETIRED_BYTES = 256 * 1024
EXPECTED_CHECKS = 2010


def f32(word: int) -> float:
    if not 0 <= word < 0x7F800000:
        raise AssertionError(f"nonfinite/negative binary32 word {word}")
    return struct.unpack("<f", struct.pack("<I", word))[0]


def fields(line: str, tag: str, count: int) -> list[int]:
    parts = line.split()
    if len(parts) != count + 1 or parts[0] != tag:
        raise AssertionError(f"unexpected record: {line}")
    try:
        return [int(part) for part in parts[1:]]
    except ValueError as error:
        raise AssertionError("record fields must be integers") from error


def check(lines: list[str]) -> tuple[float, float]:
    if len(lines) != len(SAMPLES) + 4:
        raise AssertionError("missing/extra overfit observations")
    means = []
    for update, line in zip(SAMPLES, lines):
        actual, prior, numerator, weight, mean, active, completed, epochs = fields(
            line, "CPU4-OVERFIT-OBS", 8)
        if (actual, prior, weight, active, completed, epochs) != (
                update, update - 1, 0x40800000, 4, update, update - 1):
            raise AssertionError("observation mask/count/pre-update timing differs")
        numerator_value, mean_value = f32(numerator), f32(mean)
        if mean_value != numerator_value / 4.0:
            raise AssertionError("actual L3S full-mask numerator/mean disagree")
        means.append(mean_value)
    index = len(SAMPLES)
    if fields(lines[index], "CPU4-OVERFIT-TRAIN", 3) != [UPDATES, UPDATES * 4, UPDATES - 1]:
        raise AssertionError("trainer did not complete250 updates/1000 tokens/249 epochs")
    changed, values, byte_count = fields(lines[index + 1], "CPU4-OVERFIT-PARAM", 3)
    if not 1 <= changed <= 14 or (values, byte_count) != (1192, 4768):
        raise AssertionError("canonical parameter change/complete byte count missing")
    count, tensors, start_retired, end_retired, live, committed, aborted, stages, o2_retired = fields(
        lines[index + 2], "CPU4-OVERFIT-RETENTION", 9)
    if (count, tensors, live, committed, aborted, stages) != (UPDATES, 56, 0, UPDATES, 0, 0):
        raise AssertionError("unexpected frame/payload/O2 lifecycle counts")
    if not 0 <= start_retired <= end_retired or end_retired - start_retired > MAX_F32_RETIRED_BYTES:
        raise AssertionError("F32 retired controls exceed predeclared finite250 bound")
    if not 0 < o2_retired <= MAX_O2_RETIRED_BYTES:
        raise AssertionError("O2 retired controls exceed predeclared finite250 bound")
    if fields(lines[index + 3], "CPU4-OVERFIT-RUNTIME-PASS", 1) != [EXPECTED_CHECKS]:
        raise AssertionError("runtime checks did not complete")
    initial, final = means[0], means[-1]
    if not math.isfinite(initial) or initial <= 0:
        raise AssertionError("initial CE must be finite positive")
    if final > initial * (1.0 - MIN_REDUCTION) or final > MAX_FINAL_MEAN:
        raise AssertionError(f"predeclared >=75% drop and final<=1.0 unmet: {initial:.9g}->{final:.9g}")
    return initial, final


def main() -> None:
    initial, final = check(Path(sys.argv[1]).read_text().splitlines())
    print(f"CPU4-ONE-BATCH-OVERFIT-PASS updates=250 initial-pre0={initial:.9g} "
          f"final-pre249={final:.9g} drop={1.0-final/initial:.9g}")


if __name__ == "__main__":
    main()
