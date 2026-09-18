#!/usr/bin/env python3
"""Independent exact-byte evidence for the private C2 save artifact."""

from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path


C2_MAGIC = bytes.fromhex("894553484b54524e53544154450d0a00")
C1_MAGIC = bytes.fromhex("894553484b4f4c434b50540d0a1a0a00")
C2_DOMAIN = b"eshkol-training-state-container-v1\0"
C1_DOMAIN = b"eshkol-checkpoint-container-v1\0"
MOMENT_DOMAIN = b"eshkol-training-state-moment-v1\0"


def u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def u64(data: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


def verify(path: Path) -> bytes:
    data = path.read_bytes()
    assert len(data) >= 288 and data[:16] == C2_MAGIC
    assert (u16(data, 16), u16(data, 18), u32(data, 20)) == (1, 0, 256)
    assert u64(data, 40) == len(data)
    assert data[-32:] == hashlib.sha256(C2_DOMAIN + data[:-32]).digest()
    assert (u32(data, 80), u32(data, 84), u32(data, 88), u32(data, 92)) == (
        1,
        1,
        2,
        3,
    )
    assert (u64(data, 184), u64(data, 192), u64(data, 200), u64(data, 208)) == (
        99,
        17,
        3,
        1,
    )
    assert (u64(data, 216), u64(data, 224), u64(data, 232)) == (
        1729,
        0xFFFFFFFFFFFFFFFF,
        0x8000000000000000,
    )

    metadata_at = u64(data, 48)
    payload_at = u64(data, 64)
    model_n, optimizer_n, moments_n = (u64(data, off) for off in (96, 104, 112))
    tokenizer_n, config_n, provider_n, library_n, compiler_n = (
        u32(data, off) for off in (136, 140, 144, 148, 152)
    )
    x1_n, current_n, epoch_n = u64(data, 120), u32(data, 128), u32(data, 132)
    optimizer_at = metadata_at + sum(
        (tokenizer_n, config_n, provider_n, library_n, compiler_n,
         x1_n, current_n, epoch_n)
    )
    assert optimizer_at + optimizer_n == payload_at
    assert payload_at + model_n + moments_n + 32 == len(data)

    model = data[payload_at:payload_at + model_n]
    assert model[:16] == C1_MAGIC and u64(model, 40) == len(model)
    assert model[-32:] == hashlib.sha256(C1_DOMAIN + model[:-32]).digest()

    optimizer = data[optimizer_at:optimizer_at + optimizer_n]
    count, records_at = u32(optimizer, 52), u64(optimizer, 104)
    assert count == 1
    moments_at = payload_at + model_n
    record_at = records_at
    for index in range(count):
        record_n = u64(optimizer, record_at)
        tensor_n = u64(optimizer, record_at + 32)
        avg_at = u64(optimizer, record_at + 40)
        square_at = u64(optimizer, record_at + 48)
        record = bytearray(optimizer[record_at:record_at + record_n])
        expected_avg = bytes(record[56:88])
        expected_square = bytes(record[88:120])
        record[56:120] = bytes(64)
        avg = data[moments_at + avg_at:moments_at + avg_at + tensor_n]
        square = data[moments_at + square_at:moments_at + square_at + tensor_n]
        assert len(avg) == tensor_n and len(square) == tensor_n
        assert expected_avg == hashlib.sha256(
            MOMENT_DOMAIN + struct.pack("<II", index, 1) + record + avg
        ).digest()
        assert expected_square == hashlib.sha256(
            MOMENT_DOMAIN + struct.pack("<II", index, 2) + record + square
        ).digest()
        record_at += record_n
    assert record_at == len(optimizer)
    return data


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: test_checkpoint_save_output.py A B [POSTCOMMIT]")
    images = [verify(Path(value)) for value in sys.argv[1:]]
    assert all(image == images[0] for image in images[1:])
    print(f"C2 checkpoint save independent bytes PASS: {len(images)} images")


if __name__ == "__main__":
    main()
