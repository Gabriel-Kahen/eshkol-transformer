"""Independent test-only parser/mutator for the D1 v1.0 binary format."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

SHARD_MAGIC = b"ESHKTSH1"
MANIFEST_MAGIC = b"ESHKTCM1"
SHARD_FIXED = 80
MANIFEST_FIXED = 96
RECORD_BYTES = 56
CHECKSUM_BYTES = 32


class FormatError(ValueError):
    pass


def u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def u64(data: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


def put_u16(data: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<H", data, offset, value)


def put_u32(data: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", data, offset, value)


def put_u64(data: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<Q", data, offset, value)


def resign(data: bytearray) -> None:
    data[-CHECKSUM_BYTES:] = hashlib.sha256(data[:-CHECKSUM_BYTES]).digest()


def _fingerprint(data: bytes, fixed: int, count: int) -> str:
    encoded = data[fixed : fixed + count]
    text = encoded.decode("utf-8", errors="strict")
    if not encoded or len(encoded) > 192 or text.encode("utf-8") != encoded:
        raise FormatError("noncanonical fingerprint")
    return text


def parse_shard(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    if len(data) < SHARD_FIXED + CHECKSUM_BYTES:
        raise FormatError("short shard")
    if data[:8] != SHARD_MAGIC or (u16(data, 8), u16(data, 10)) != (1, 0):
        raise FormatError("shard identity/version")
    if u32(data, 16) != 0 or u32(data, 20) != 1 or u32(data, 24) != 1:
        raise FormatError("shard features/encoding/checksum")
    if data[28:32] != bytes(4) or data[68:72] != bytes(4):
        raise FormatError("shard reserved")
    header_bytes = u32(data, 12)
    token_count = u64(data, 40)
    payload_bytes = u64(data, 48)
    fingerprint_bytes = u32(data, 64)
    if header_bytes != SHARD_FIXED + fingerprint_bytes:
        raise FormatError("shard header size")
    if payload_bytes != token_count * 8:
        raise FormatError("shard payload size")
    if u64(data, 72) != len(data) or len(data) != header_bytes + payload_bytes + 32:
        raise FormatError("shard file size")
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise FormatError("shard digest")
    tokens = list(struct.unpack_from(f"<{token_count}Q", data, header_bytes))
    return {
        "bytes": data,
        "digest": data[-32:],
        "fingerprint": _fingerprint(data, SHARD_FIXED, fingerprint_bytes),
        "index": u64(data, 32),
        "tokens": tokens,
        "vocab_size": u64(data, 56),
    }


def parse_corpus(directory: Path) -> dict[str, object]:
    data = (directory / "manifest.etm").read_bytes()
    if len(data) < MANIFEST_FIXED + CHECKSUM_BYTES:
        raise FormatError("short manifest")
    if data[:8] != MANIFEST_MAGIC or (u16(data, 8), u16(data, 10)) != (1, 0):
        raise FormatError("manifest identity/version")
    if u32(data, 16) != 0 or u32(data, 20) != 1 or u32(data, 24) != 1:
        raise FormatError("manifest features/encoding/checksum")
    if data[28:32] != bytes(4) or data[88:96] != bytes(8):
        raise FormatError("manifest reserved")
    shard_count = u64(data, 32)
    total_tokens = u64(data, 40)
    total_shard_bytes = u64(data, 48)
    vocab_size = u64(data, 56)
    shard_limit = u64(data, 64)
    fingerprint_bytes = u32(data, 72)
    header_bytes = u32(data, 12)
    if header_bytes != MANIFEST_FIXED + fingerprint_bytes:
        raise FormatError("manifest header size")
    if u32(data, 76) != RECORD_BYTES:
        raise FormatError("record size")
    if u64(data, 80) != len(data):
        raise FormatError("manifest declared size")
    if len(data) != header_bytes + shard_count * RECORD_BYTES + 32:
        raise FormatError("manifest table size")
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise FormatError("manifest digest")
    fingerprint = _fingerprint(data, MANIFEST_FIXED, fingerprint_bytes)
    records = []
    tokens: list[int] = []
    for index in range(shard_count):
        offset = header_bytes + index * RECORD_BYTES
        record = {
            "index": u64(data, offset),
            "tokens": u64(data, offset + 8),
            "bytes": u64(data, offset + 16),
            "digest": data[offset + 24 : offset + 56],
            "offset": offset,
        }
        shard_path = directory / f"shard-{index:016d}.ets"
        shard = parse_shard(shard_path)
        if record["index"] != index or shard["index"] != index:
            raise FormatError("index mismatch")
        if record["tokens"] != len(shard["tokens"]):
            raise FormatError("count mismatch")
        if record["bytes"] != len(shard["bytes"]):
            raise FormatError("byte mismatch")
        if record["digest"] != shard["digest"]:
            raise FormatError("digest mismatch")
        if shard["fingerprint"] != fingerprint or shard["vocab_size"] != vocab_size:
            raise FormatError("identity mismatch")
        tokens.extend(shard["tokens"])
        records.append(record)
    if len(tokens) != total_tokens:
        raise FormatError("total token mismatch")
    if sum(int(record["bytes"]) for record in records) != total_shard_bytes:
        raise FormatError("total byte mismatch")
    return {
        "bytes": data,
        "fingerprint": fingerprint,
        "records": records,
        "shard_count": shard_count,
        "shard_limit": shard_limit,
        "tokens": tokens,
        "total_shard_bytes": total_shard_bytes,
        "vocab_size": vocab_size,
    }
