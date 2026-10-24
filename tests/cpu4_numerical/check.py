#!/usr/bin/env python3
"""Independent binary64 CE/finite-difference oracle for the genuine Eshkol C4 row."""
import argparse
import math
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpu4_transport"))
import reference as model


def f32(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    sizes = {"ID": [4], "TARGET": [4], "MASK": [4],
             "VALUE": [math.prod(s) for _, s in model.PARAMETERS],
             "AFTER": [math.prod(s) for _, s in model.PARAMETERS],
             "GRAD": [math.prod(s) for _, s in model.PARAMETERS],
             "LOGIT": [1024], "OBS": [3]}
    rows = {name: [[None] * width for width in widths] for name, widths in sizes.items()}
    for line in path.read_text().splitlines():
        if not line.startswith("NUM "):
            continue
        parts = line.split()
        if len(parts) != 5 or parts[1] not in rows:
            raise AssertionError(f"malformed numerical row: {line}")
        kind, ordinal, index, word = parts[1], *map(int, parts[2:])
        if not (0 <= ordinal < len(rows[kind]) and
                0 <= index < len(rows[kind][ordinal]) and
                0 <= word < 2**32 and rows[kind][ordinal][index] is None):
            raise AssertionError(f"duplicate or out-of-range numerical row: {line}")
        rows[kind][ordinal][index] = word
    if any(value is None for planes in rows.values() for plane in planes for value in plane):
        raise AssertionError("incomplete C4 numerical transcript")
    return rows


def near(actual, expected, atol, rtol, label):
    error = abs(actual - expected)
    tolerance = atol + rtol * abs(expected)
    if (not all(map(math.isfinite, (actual, expected, error, tolerance))) or
            error > tolerance):
        raise AssertionError(f"{label}: {actual} versus {expected}; error={error}")
    return error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("transcript", type=Path)
    args = parser.parse_args()
    data = read(args.transcript)
    ids, targets, mask = (data[kind][0] for kind in ("ID", "TARGET", "MASK"))
    require(ids == [3, 197, 3, 41], "wrong genuine D2 IDs")
    require(targets == [197, 3, 41, 7], "wrong genuine D2 targets")
    require(mask == [1, 1, 1, 1], "wrong genuine D2 mask")
    require(data["AFTER"] == data["VALUE"], "canonical parameter values changed")
    p = [[f32(word) for word in plane] for plane in data["VALUE"]]
    measured = [f32(word) for word in data["LOGIT"][0]]
    expected = model.forward(p, ids)
    forward_error = max(near(a, b, model.FORWARD_ATOL, model.FORWARD_RTOL,
                             f"logit {i}")
                        for i, (a, b) in enumerate(zip(measured, expected)))

    def ce(parameters):
        flat = model.forward(parameters, ids)
        total = 0.0
        for t, target in enumerate(targets):
            if mask[t]:
                row = flat[t * 256:(t + 1) * 256]
                largest = max(row)
                total += largest + math.log(sum(math.exp(x - largest) for x in row)) - row[target]
        return total

    # The reference's central-difference routine calls this independent CE.
    model.objective = lambda parameters, seed: ce(parameters)
    gradients = model.vjp(p, [0.0] * 1024)
    require(len(gradients) == 14 and sum(map(len, gradients)) == 1192,
            "incomplete independent gradients")
    gradient_error = max(
        near(f32(actual), expected, model.GRADIENT_ATOL, model.GRADIENT_RTOL,
             f"gradient {ordinal}:{index}")
        for ordinal, (actuals, expected_values) in
        enumerate(zip(data["GRAD"], gradients))
        for index, (actual, expected) in enumerate(zip(actuals, expected_values)))
    require(all(any(abs(value) > 1e-7 for value in plane) for plane in gradients),
            "one canonical parameter has no numerical gradient")
    observation = [f32(word) for word in data["OBS"][0]]
    numerator = ce(p)
    near(observation[0], numerator, 3e-4, 5e-4, "CE numerator")
    require(observation[1] == 4.0, "wrong T4 active count")
    near(observation[2], numerator / 4, 3e-4, 5e-4, "CE mean")
    print("CPU4-ESHKOL-NUMERICAL-PASS values=1192 logits=1024 gradients=1192 "
          f"max-logit-error={forward_error:.9g} max-gradient-error={gradient_error:.9g}")


if __name__ == "__main__":
    main()
