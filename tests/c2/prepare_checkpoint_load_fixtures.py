#!/usr/bin/env python3
"""Deterministic C2 LOAD fixtures, including fully rechecksummed dtype cases."""

from __future__ import annotations

import sys
import hashlib
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.c2.test_checkpoint_format import (
    C1_DOMAIN,
    C1_TENSOR_DOMAIN,
    make_fixture,
    p64,
    resign_outer,
    u64,
)


def resign_c1_tensor(data: bytearray, record: int, payload: int) -> None:
    record_bytes = u64(data, record)
    payload_offset = u64(data, record + 8)
    payload_bytes = u64(data, record + 16)
    digest = hashlib.sha256(
        C1_TENSOR_DOMAIN
        + data[record:record + 48]
        + b"\0" * 32
        + data[record + 80:record + record_bytes]
        + data[payload + payload_offset:payload + payload_offset + payload_bytes]
    ).digest()
    data[record + 48:record + 80] = digest


def resign_c1(data: bytearray, model: int) -> None:
    length = u64(data, model + 40)
    data[model + length - 32:model + length] = hashlib.sha256(
        C1_DOMAIN + data[model:model + length - 32]
    ).digest()


def main() -> None:
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    valid, _ = make_fixture()
    (root / "valid.c2").write_bytes(valid)

    # make_fixture's optional buffer is a canonical C1 bool buffer with fully
    # valid tensor, nested, and outer digests. C2 must reject it as a dtype
    # mismatch before invoking the fixed f32 I2 decoder.
    hostile, _ = make_fixture(4, 1)
    (root / "nonf32-buffer.c2").write_bytes(hostile)
    hostile_i64, _ = make_fixture(8, 2)
    (root / "nonf32-i64-buffer.c2").write_bytes(hostile_i64)

    # Convert that second record to one f32 element without changing physical
    # size, and recompute all three digest layers. This is a larger, valid C2
    # image used to exercise a measure/stage component-size race.
    larger, _ = make_fixture(4, 3)
    (root / "larger-valid.c2").write_bytes(larger)

    # Strong reconstruction fixture: two C1 entries (one parameter and one
    # f32 buffer), exact negative-zero buffer bits, and high u64 RNG counter
    # words that must round-trip through signed Eshkol i64 controls.
    exact, offsets = make_fixture(4, 3)
    first_record = offsets["c1_record"]
    buffer_record = first_record + u64(exact, first_record)
    model_payload = offsets["c1_payload"]
    exact[model_payload + 8:model_payload + 12] = struct.pack("<I", 0x80000000)
    resign_c1_tensor(exact, buffer_record, model_payload)
    resign_c1(exact, offsets["model"])
    p64(exact, 224, (1 << 64) - 1)
    p64(exact, 232, 1 << 63)
    resign_outer(exact)
    (root / "exact-valid.c2").write_bytes(exact)


if __name__ == "__main__":
    main()
