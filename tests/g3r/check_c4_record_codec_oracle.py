#!/usr/bin/env python3
"""Independent byte layout/SHA oracle for the private C4 companion."""

from hashlib import sha256
from pathlib import Path
import struct
import sys


FINGERPRINT = b"sha256:eshkol-byte-tokenizer-v1:" + b"a" * 64
C2_DIGEST = bytes(range(32))  # Opaque codec input, not a loaded image.
GREEDY = (0, 1065353216, 256, 1065353216, 2, -1)
CATEGORICAL = (1, 1065353216, 17, 1056964608, 2, 65)
RNG = (1, 1729, -1, 0)
EXHAUSTED = (1, 1729, -1, -1)
HIGH_BIT = (1, 9223372036854775807, -9223372036854775808,
            9223372036854775807)
CASES = {
    "prompt": (GREEDY, RNG, bytes((7, 11)), 0, 0),
    "prefix": (CATEGORICAL, RNG, bytes((7, 11, 31)), 1, 1),
    "first-eos": (CATEGORICAL, EXHAUSTED, bytes((7, 11, 65)), 1, 2),
    "second": (CATEGORICAL, HIGH_BIT, bytes((7, 11, 31, 41)), 2, 3),
    "second-eos": (CATEGORICAL, RNG, bytes((7, 11, 31, 65)), 2, 3),
    "prompt-eos": (CATEGORICAL, RNG, bytes((65, 11, 31)), 1, 1),
}


def expected_record(policy, rng, history, generated, phase):
    image = bytearray(288)
    image[:6] = b"G3RCV2"
    struct.pack_into("<HHIIIIIII", image, 8, 2, 0, 288, 2, 1,
                     len(history), 2, generated, 0)
    struct.pack_into("<6q", image, 40, *policy)
    struct.pack_into("<4q", image, 88, *rng)
    image[120:120 + len(history)] = history
    struct.pack_into("<I", image, 124, phase)
    image[128:224] = FINGERPRINT
    image[224:256] = C2_DIGEST
    image[256:288] = sha256(image[:256]).digest()
    return bytes(image)


def main():
    observed = {}
    lines = Path(sys.argv[1]).read_text().splitlines()
    for line in lines:
        if line.startswith("G3RC4_VECTOR "):
            _, name, raw = line.split(" ")
            assert name in CASES and name not in observed
            observed[name] = bytes.fromhex(raw)
    assert set(observed) == set(CASES), observed.keys()
    for name, args in CASES.items():
        assert observed[name] == expected_record(*args), name
    assert len([line for line in lines if line.startswith(
        "G3-R C4 codec PASS: checks=")]) == 1
    print("G3-R C4 independent SHA/layout oracle PASS: 6 vectors")


if __name__ == "__main__":
    main()
