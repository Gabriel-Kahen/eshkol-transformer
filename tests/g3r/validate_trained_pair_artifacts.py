"""Independent byte-level check for a future compiled trained-C2 producer run.

This checks the recorded pair's binding and deterministic P1 fields. It does
not prove that the Eshkol producer ran or that C2 semantic LOAD succeeds.
"""
import argparse
import hashlib
from pathlib import Path
import struct


FINGERPRINT = (b"sha256:eshkol-byte-tokenizer-v1:"
               b"aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704")
MAGIC = bytes.fromhex("894553484b54524e53544154450d0a00")


def verify(trained: bytes, saved: bytes, record: bytes, prompt: int) -> None:
    if not 0 <= prompt <= 255:
        raise ValueError("prompt is outside raw V256")
    if trained != saved:
        raise ValueError("saved C2 differs from the exact trained checkpoint")
    if (len(saved) < 352 or saved[:16] != MAGIC or
            struct.unpack_from("<I", saved, 20)[0] != 256 or
            struct.unpack_from("<Q", saved, 48)[0] != 256 or
            struct.unpack_from("<I", saved, 136)[0] != 96 or
            struct.unpack_from("<Q", saved, 184)[0] == 0 or
            struct.unpack_from("<Q", saved, 192)[0] == 0 or
            saved[256:352] != FINGERPRINT):
        raise ValueError("C2 lacks a trained state or canonical T1 field")
    if len(record) != 288 or record[:8] != b"G3RCV1\0\0":
        raise ValueError("companion size or magic differs")
    if struct.unpack_from("<HHI", record, 8) != (1, 0, 288):
        raise ValueError("companion version or extent differs")
    if struct.unpack_from("<6I", record, 16) != (1, 1, 1, 1, 0, 0):
        raise ValueError("companion profile or history partition differs")
    if struct.unpack_from("<6q", record, 40) != (
            1, 1065353216, 256, 1065353216, 0, -1):
        raise ValueError("companion policy differs")
    if struct.unpack_from("<4q", record, 88) != (1, 1729, 0, 0):
        raise ValueError("companion RNG differs")
    if record[120] != prompt or record[121:128] != bytes(7):
        raise ValueError("companion prompt or padding differs")
    if record[128:224] != FINGERPRINT:
        raise ValueError("companion tokenizer differs from C2")
    if record[224:256] != hashlib.sha256(saved).digest():
        raise ValueError("companion does not bind the saved C2 image")
    if record[256:288] != hashlib.sha256(record[:256]).digest():
        raise ValueError("companion checksum differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--saved", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--prompt", type=int, required=True)
    args = parser.parse_args()
    verify(args.trained.read_bytes(), args.saved.read_bytes(),
           args.companion.read_bytes(), args.prompt)
    print("G3-R TRAINED PAIR BYTES PASS")


if __name__ == "__main__":
    main()
