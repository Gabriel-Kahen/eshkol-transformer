#!/usr/bin/env python3
"""Generate small independent G3RCV1 file-read fixtures in a fresh directory."""
import argparse
import hashlib
import os
from pathlib import Path
import struct


def record(c2_digest: bytes, fingerprint: bytes = b"sha256:eshkol-byte-tokenizer-v1:" + b"a" * 64) -> bytes:
    if len(fingerprint) != 96 or not fingerprint.startswith(b"sha256:eshkol-byte-tokenizer-v1:"):
        raise ValueError("fixture requires an exact raw-V256 tokenizer fingerprint")
    data = bytearray(288)
    data[:8] = b"G3RCV1\0\0"
    struct.pack_into("<HHI", data, 8, 1, 0, 288)
    struct.pack_into("<IIIIII", data, 16, 1, 1, 1, 1, 0, 0)
    struct.pack_into("<6q", data, 40, 0, 1065353216, 256,
                     1065353216, 1, -1)
    struct.pack_into("<4q", data, 88, 1, 1729, -1, -(1 << 63))
    data[120] = 7
    data[128:224] = fingerprint
    data[224:256] = c2_digest
    data[256:] = hashlib.sha256(data[:256]).digest()
    return bytes(data)


def prepare(c2_path: Path, output: Path) -> None:
    image = c2_path.read_bytes()
    digest = hashlib.sha256(image).digest()
    if (image[:16] != bytes.fromhex("894553484b54524e53544154450d0a00") or
            struct.unpack_from("<I", image, 20)[0] != 256 or
            struct.unpack_from("<Q", image, 48)[0] != 256 or
            struct.unpack_from("<I", image, 136)[0] != 96):
        raise ValueError("fixture C2 tokenizer extent is unsupported")
    fingerprint = image[256:352]
    output.mkdir()
    valid = record(digest)
    (output / "valid.g3r").write_bytes(valid)
    (output / "matched.g3r").write_bytes(record(digest, fingerprint))
    (output / "wrong-digest.g3r").write_bytes(
        record(hashlib.sha256(b"independent wrong C2 image").digest()))
    (output / "short.g3r").write_bytes(valid[:-1])
    (output / "long.g3r").write_bytes(valid + b"\0")
    torn = bytearray(valid)
    torn[120] ^= 1
    (output / "torn.g3r").write_bytes(torn)
    (output / "link.g3r").symlink_to("valid.g3r")
    os.mkfifo(output / "fifo.g3r", 0o600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c2-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.c2_image, args.output)
