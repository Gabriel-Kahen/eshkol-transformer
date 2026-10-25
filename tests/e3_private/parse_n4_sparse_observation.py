"""Verify a complete compiled C1/D2/M3/E3 sparse-subnormal transcript."""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.e3_private.parse_numerical_observation import ROLE_NAMES, _pair, _words
from tests.e3_private.parse_overflow_observation import SHAPES
from tests.e3_reference.n4_sparse_bits import AXIS, _norm_axis, bits, forward_bits


def _finite(words: list[str]) -> bool:
    return all(int(word, 16) & 0x7F800000 != 0x7F800000 for word in words)


def _expected_role(name: str, count: int) -> list[str]:
    axis = [f"{word:08x}" for word in AXIS]
    zero = ["00000000"] * 4
    if name == "ET":
        return axis + zero
    if name == "EP":
        return zero + axis
    if name in {"X", "R", "Y"}:
        return axis * 2
    if name == "NF":
        return [f"{bits(value):08x}" for value in _norm_axis()] * 2
    if name == "Z":
        return [f"{word:08x}" for word in forward_bits()]
    return ["00000000"] * count


def parse(raw: bytes, gradient: str) -> None:
    if gradient not in {"absent", "present"}:
        raise ValueError("unknown gradient mode")
    if not raw.endswith(b"\n") or len(raw) > 262144:
        raise ValueError("N4 transcript incomplete or oversized")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("N4 transcript is not ASCII") from error
    tags = ["E3N4K1", "E3ROW", *(["E3ROLE"] * 21), "E3CE", "E3BATCH",
            "E3STATE", "E3FINAL", f"E3-N4-SPARSE-PASS gradient={gradient}"]
    if len(lines) != len(tags):
        raise ValueError("N4 transcript omitted or duplicated a record")
    for index, (line, tag) in enumerate(zip(lines, tags, strict=True)):
        fields = line.split("\t")
        if fields[0] != tag:
            raise ValueError("N4 transcript record order differs")
        if tag == "E3N4K1":
            if len(fields) != 4 or len(fields[1]) != 4 or len(fields[2]) != 8:
                raise ValueError("N4 K1 input/fenv record is malformed")
            try:
                control, mxcsr = int(fields[1], 16), int(fields[2], 16)
            except ValueError as error:
                raise ValueError("N4 K1 fenv is malformed") from error
            if (control & 0x003F) != 0x003F or control & 0x0C00 or \
                    (mxcsr & 0x1F80) != 0x1F80 or mxcsr & 0xE040:
                raise ValueError("N4 K1 floating environment is unsupported")
            if _words(fields[3], 512) != _expected_role("Z", 512):
                raise ValueError("N4 K1 input differs from independent M3 oracle")
            continue
        if tag == "E3-N4-SPARSE-PASS gradient=" + gradient:
            if line != tag:
                raise ValueError("N4 runtime footer differs")
            continue
        if tag == "E3FINAL":
            if len(fields) != 3 or _pair(fields[2]) != [1, 1] or \
                    not _finite(_words(fields[1], 4)):
                raise ValueError("N4 final output differs")
            continue
        if len(fields) < 3 or fields[1] != "0":
            raise ValueError("N4 batch ordinal differs")
        if tag == "E3ROW":
            if len(fields) != 5 or _pair(fields[2]) != [3, 0] or \
                    _pair(fields[3]) != [4, 0] or \
                    _pair(fields[4], mask=True) != [True, False]:
                raise ValueError("N4 row is not authentic D2 padded row")
        elif tag == "E3ROLE":
            ordinal = index - 2
            shape = SHAPES[ordinal]
            count = 1
            for dimension in shape:
                count *= dimension
            if len(fields) != 5 or fields[2] != ROLE_NAMES[ordinal] or \
                    fields[3] != ",".join(map(str, shape)) or \
                    _words(fields[4], count) != _expected_role(fields[2], count):
                raise ValueError("N4 M3 role shape or exact bits differ: " + ROLE_NAMES[ordinal])
        elif tag == "E3CE":
            if len(fields) != 3 or not _finite(_words(fields[2], 2)):
                raise ValueError("N4 CE is missing or nonfinite")
        elif tag == "E3BATCH":
            if len(fields) != 4 or fields[3] != "1":
                raise ValueError("N4 K1 active count differs")
            values = _words(fields[2], 4)
            if not _finite(values) or values[1] != "3f800000" or \
                    values[3] != "00000000":
                raise ValueError("N4 K1/reduction output differs")
        elif tag == "E3STATE":
            if len(fields) != 4 or _pair(fields[3]) != [1, 1] or \
                    not _finite(_words(fields[2], 3)):
                raise ValueError("N4 accumulator output differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--gradient", required=True, choices=("absent", "present"))
    args = parser.parse_args()
    parse(args.stdout.read_bytes(), args.gradient)
    print("verified_sparse_m3_k1=1")


if __name__ == "__main__":
    main()
