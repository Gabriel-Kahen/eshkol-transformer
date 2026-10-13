"""Check the entire genuine E3 numerical trace for the fixed overflow witness.

Expected words are frozen from the sealed independent math audit
e3-finite-overflow-model-reachability-20261004-b (SHA256SUMS-AUDIT
09bf22e779a880df504b681e77fe6884eb37ad6484d4f922f8b02ce69890296a).
This parser never derives an expected value from the compiled observation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.e3_private.parse_numerical_observation import ROLE_NAMES, _pair, _words


SHAPES = (
    *((1, 2, 4),) * 7,
    *((1, 2, 2, 2),) * 4,
    *((1, 2, 4),) * 4,
    *((1, 2, 8),) * 2,
    *((1, 2, 4),) * 3,
    (1, 2, 256),
)
LOSS = {"finite": "42b0ffc6", "overflow": "42b1ffc6"}
NORM = {"finite": "41b0ffc6", "overflow": "41b1ffc6"}
NEG_NORM = {"finite": "c1b0ffc6", "overflow": "c1b1ffc6"}
PPL = "7f4cc590"
ZERO = "00000000"
ONE = "3f800000"
PADDED_CE = "40b17218"


def expected_role(name: str, case: str) -> list[str]:
    if name in {"ET", "X", "R", "Y"}:
        return [ONE, "bf800000", ONE, "bf800000", *([ZERO] * 4)]
    if name == "NF":
        return [NORM[case], NEG_NORM[case], NORM[case],
                NEG_NORM[case], *([ZERO] * 4)]
    if name == "Z":
        words = [ZERO] * 512
        words[3] = LOSS[case]
        return words
    return [ZERO] * _product(SHAPES[ROLE_NAMES.index(name)])


def _product(shape: tuple[int, ...]) -> int:
    size = 1
    for dimension in shape:
        size *= dimension
    return size


def parse(raw: bytes, case: str, gradient: str, *, mutant: str = "normal") -> int:
    if case not in LOSS or gradient not in {"absent", "present"}:
        raise ValueError("unknown overflow fixture case")
    if mutant not in {"normal", "clamp"}:
        raise ValueError("unknown overflow mutant")
    if not raw.endswith(b"\n") or len(raw) > 262144:
        raise ValueError("overflow observation is incomplete or oversized")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("overflow observation is not ASCII") from error
    killed = mutant == "clamp" and case == "overflow"
    calls = 1 if case == "finite" or killed else 2
    batch_tags = ["E3ROW", *(["E3ROLE"] * 21), "E3CE", "E3BATCH", "E3STATE"]
    tags = batch_tags * calls + (["E3FINAL"] if case == "finite" or killed else [])
    tags += (["E3-RETENTION-FAIL: overflow-stage12-error"] if killed else
             ["E3-FINITE-OVERFLOW-PASS gradient=" + gradient])
    if len(lines) != len(tags):
        raise ValueError("overflow observation omitted or duplicated a record")
    for index, (line, tag) in enumerate(zip(lines, tags, strict=True)):
        fields = line.split("\t")
        if tag.startswith("E3-FINITE-") or tag.startswith("E3-RETENTION-FAIL:"):
            if line != tag:
                raise ValueError("overflow runtime did not complete exactly")
            continue
        if fields[0] != tag:
            raise ValueError("overflow record order differs")
        if tag == "E3FINAL":
            expected_ppl = "7f7fffff" if killed else PPL
            if len(fields) != 3 or _words(fields[1], 4) != \
                    [LOSS[case], ONE, expected_ppl, ZERO] or \
                    _pair(fields[2]) != [1, 1]:
                raise ValueError("final publication differs")
            continue
        # Every fresh E3 acquire zeros its native batch counts. The retry is
        # the second record set in this process, but its native ordinal is 0.
        if len(fields) < 3 or fields[1] != "0":
            raise ValueError("overflow ordinal differs")
        if tag == "E3ROW":
            if len(fields) != 5 or _pair(fields[2]) != [3, 0] or \
                    _pair(fields[3]) != [4, 0] or \
                    _pair(fields[4], mask=True) != [True, False]:
                raise ValueError("authentic D2 row differs")
        elif tag == "E3ROLE":
            role_index = index % len(batch_tags) - 1
            if len(fields) != 5 or fields[2] != ROLE_NAMES[role_index]:
                raise ValueError("E3 role order differs")
            shape = SHAPES[role_index]
            if fields[3] != ",".join(map(str, shape)) or \
                    _words(fields[4], _product(shape)) != expected_role(fields[2], case):
                raise ValueError("E3 role geometry or value differs: " + fields[2])
        elif tag == "E3CE":
            if len(fields) != 3 or _words(fields[2], 2) != [LOSS[case], PADDED_CE]:
                raise ValueError("E3 CE differs")
        elif tag == "E3BATCH":
            if len(fields) != 4 or _words(fields[2], 4) != \
                    [LOSS[case], ONE, ZERO, ZERO] or fields[3] != "1":
                raise ValueError("E3 reduction differs")
        elif tag == "E3STATE":
            if len(fields) != 4 or _words(fields[2], 3) != \
                    [LOSS[case], ONE, ZERO] or _pair(fields[3]) != [1, 1]:
                raise ValueError("E3 accumulator differs")
        else:
            raise AssertionError("unexpected parser tag")
    return calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=tuple(LOSS))
    parser.add_argument("--gradient", required=True, choices=("absent", "present"))
    parser.add_argument("--mutant", choices=("normal", "clamp"), default="normal")
    arguments = parser.parse_args()
    print("verified_calls=" + str(parse(arguments.stdout.read_bytes(),
                                       arguments.case, arguments.gradient,
                                       mutant=arguments.mutant)))


if __name__ == "__main__":
    main()
