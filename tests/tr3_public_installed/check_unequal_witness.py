"""Reduce separately observed installed step words to an exact train oracle."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct


REFERENCE_NUMERATORS = (1093788954, 1093813638, 1085364985, 1093109014)
WEIGHT_BITS = (1073741824, 1073741824, 1065353216, 1073741824)


def unpack(word: int) -> float:
    if not 0 <= word <= 0xFFFFFFFF:
        raise ValueError("not a binary32 word")
    value = struct.unpack("<f", struct.pack("<I", word))[0]
    if not math.isfinite(value):
        raise ValueError("nonfinite physical observation")
    return value


def rounded(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def check(path: Path) -> None:
    lines = path.read_text().splitlines()
    if len(lines) != 7 or lines[-1] != (
        "TR3-PUBLIC-STEP-AOT-PASS loss-bits=1085403699 weight-bits=1077936128"
    ):
        raise ValueError("incomplete installed step/train witness")
    observed: list[float] = []
    weights: list[float] = []
    for ordinal, (line, reference, expected_weight) in enumerate(
        zip(lines[:4], REFERENCE_NUMERATORS, WEIGHT_BITS)
    ):
        fields = line.split()
        if len(fields) != 4 or fields[:2] != ["TR3-UNEQUAL-OBS", str(ordinal)]:
            raise ValueError("physical observations are missing or reordered")
        numerator_word, weight_word = map(int, fields[2:])
        numerator = unpack(numerator_word)
        weight = unpack(weight_word)
        if weight_word != expected_weight:
            raise ValueError("D2 physical mask weight differs")
        wanted = unpack(reference)
        # The accepted TR3 step trajectory oracle compares post-update
        # objectives at this numerical bound; PyTorch is not an exact O2 oracle.
        if abs(numerator - wanted) > 2e-5 + 3e-5 * abs(wanted):
            raise ValueError("installed objective differs from PyTorch reference")
        observed.append(numerator)
        weights.append(weight)

    step_fields = lines[4].split()
    if len(step_fields) != 7 or step_fields[0] != "TR3-UNEQUAL-STEP-OBS":
        raise ValueError("missing independently published step trajectory")
    first_mean, first_weight, first_tokens, second_mean, second_weight, second_tokens = (
        map(int, step_fields[1:])
    )
    if (first_weight, first_tokens, second_weight, second_tokens) != (
        bits(4.0), 4, bits(3.0), 3
    ):
        raise ValueError("step weight or token counters differ")
    for offset, published_mean, expected_weight in (
        (0, first_mean, 4.0), (2, second_mean, 3.0)
    ):
        numerator = rounded(observed[offset] + observed[offset + 1])
        if bits(rounded(numerator / expected_weight)) != published_mean:
            raise ValueError("published step mean differs from physical observations")

    summary_fields = lines[5].split()
    if len(summary_fields) != 3 or summary_fields[0] != "TR3-UNEQUAL-SUMMARY":
        raise ValueError("missing installed train summary")
    published_loss, published_weight = map(int, summary_fields[1:])
    running_numerator = 0.0
    running_weight = 0.0
    for numerator, weight in zip(observed, weights):
        running_numerator = rounded(running_numerator + numerator)
        running_weight = rounded(running_weight + weight)
    expected_loss = bits(rounded(running_numerator / running_weight))
    if published_loss != expected_loss or published_weight != bits(running_weight):
        raise ValueError("train summary differs from ordered physical reduction")
    mean_of_means = bits(rounded(
        rounded(unpack(first_mean) / 2.0)
        + rounded(unpack(second_mean) / 2.0)
    ))
    if expected_loss == mean_of_means:
        raise ValueError("unequal fixture does not distinguish mean-of-means")
    print("TR3-TRAIN-UNEQUAL-EXACT-PASS"
          f" loss-bits={expected_loss} mean-of-means-bits={mean_of_means}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("witness", type=Path)
    check(parser.parse_args().witness)
