#!/usr/bin/env python3
"""Independent fixed-layout SHA-256 vectors for the private G3RCV1 codec."""

from hashlib import sha256
from pathlib import Path
import struct
import sys


FINGERPRINT = b"sha256:eshkol-byte-tokenizer-v1:" + b"a" * 64
C2_DIGEST = bytes(range(32))  # Opaque codec-only input, not a C2 checkpoint.


def expected_record(policy, rng, history, prompt, generated, manual):
    image = bytearray(288)
    image[:6] = b"G3RCV1"
    struct.pack_into("<HHIIIIIII", image, 8, 1, 0, 288, 1, 1,
                     len(history), prompt, generated, manual)
    struct.pack_into("<6q", image, 40, *policy)
    struct.pack_into("<4q", image, 88, *rng)
    image[120:120 + len(history)] = history
    image[128:224] = FINGERPRINT
    image[224:256] = C2_DIGEST
    image[256:288] = sha256(image[:256]).digest()
    return bytes(image)


CASES = {
    "greedy": ((0, 1065353216, 256, 1065353216, 1, -1),
               (1, 7, 0, 0), bytes((7, 11)), 2, 0, 0),
    "terminal": ((1, 1065353216, 17, 1056964608, 1, 255),
                 (1, 1729, -1, -1), bytes((7, 255)), 1, 1, 0),
    "carry": ((1, 1065353216, 17, 1056964608, 1, 255),
              (1, 1729, -1, 0), bytes((0,)), 1, 0, 0),
    "high-bit": ((1, 1065353216, 17, 1056964608, 1, 255),
                 (1, 9223372036854775807, -9223372036854775808,
                  9223372036854775807), bytes((0,)), 1, 0, 0),
    "manual": ((1, 1065353216, 17, 1056964608, 0, 11),
               (1, 7, 0, 0), bytes((7, 11)), 1, 0, 1),
}


def main():
    lines = Path(sys.argv[1]).read_text().splitlines()
    observed = {}
    for line in lines:
        if line.startswith("G3R_VECTOR "):
            _, name, encoded = line.split(" ")
            assert name in CASES and name not in observed
            observed[name] = bytes.fromhex(encoded)
    assert set(observed) == set(CASES), observed.keys()
    for name, args in CASES.items():
        assert len(observed[name]) == 288, name
        assert observed[name] == expected_record(*args), name
    assert len([line for line in lines if line.startswith(
        "G3-R record codec PASS: checks=")]) == 1
    print("G3-R record codec independent SHA/layout oracle PASS: 5 vectors")


if __name__ == "__main__":
    main()
