#!/usr/bin/env python3
"""Independent byte-for-byte oracle for the real P1/G1 adapter output."""
import hashlib
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.c2.prepare_checkpoint_operational_fixtures import make_m3_artifact

lines = Path(sys.argv[1]).read_text().splitlines()
def one(prefix):
    matches = [line[len(prefix):] for line in lines if line.startswith(prefix)]
    assert len(matches) == 1, (prefix, len(matches))
    return matches[0]
def many(prefix, count):
    matches = [int(line[len(prefix):]) for line in lines if line.startswith(prefix)]
    assert len(matches) == count, (prefix, len(matches))
    return matches

record = bytes.fromhex(one("G3R_ADAPTER_RECORD "))
sampled = int(one("G3R_ADAPTER_SAMPLE "))
policy = many("G3R_ADAPTER_POLICY ", 6)
rng = many("G3R_ADAPTER_RNG ", 4)
image, _ = make_m3_artifact((2, 4))
digest = hashlib.sha256(image).digest()
assert len(record) == 288 and 0 <= sampled < 256
expected = bytearray(288)
expected[:6] = b"G3RCV1"
struct.pack_into("<HHIIIII", expected, 8, 1, 0, 288, 1, 1, 2, 1)
struct.pack_into("<II", expected, 32, 1, 0)
struct.pack_into("<6q", expected, 40, *policy)
struct.pack_into("<4q", expected, 88, *rng)
expected[120:122] = bytes((65, sampled))
expected[128:224] = (b"sha256:eshkol-byte-tokenizer-v1:"
                     b"aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704")
expected[224:256] = digest
expected[256:288] = hashlib.sha256(expected[:256]).digest()
assert record == expected
assert policy[5] == -1 and policy[4] == 1
assert rng[0] == 1
assert record[6:8] == bytes(2) and record[122:128] == bytes(6)
print("G3-R snapshot codec adapter oracle PASS")
