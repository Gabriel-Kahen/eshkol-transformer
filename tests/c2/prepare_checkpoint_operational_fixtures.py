#!/usr/bin/env python3
"""Generate deterministic, fully valid C2 operational-boundary artifacts."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.c2.test_checkpoint_format import (
    C1_DOMAIN,
    C1_MAGIC,
    C1_TENSOR_DOMAIN,
    C2_DOMAIN,
    C2_MAGIC,
    COMPILER,
    CONFIG,
    LIBRARY,
    MOMENT_DOMAIN,
    PROVIDER,
    TOKENIZER,
    X1,
    cursor,
    make_optimizer,
    make_fixture as make_format_fixture,
    p16,
    p32,
    p64,
)

FILE_LIMIT = 16 * 1024 * 1024
METADATA_LIMIT = 512 * 1024
TENSOR_LIMIT = 8 * 1024 * 1024
TENSOR_COUNT_LIMIT = 64

M3_PARAMETERS = (
    ((b"blocks", b"0", b"attention", b"key", b"weight"), (4, 4)),
    ((b"blocks", b"0", b"attention", b"output", b"weight"), (4, 4)),
    ((b"blocks", b"0", b"attention", b"query", b"weight"), (4, 4)),
    ((b"blocks", b"0", b"attention", b"value", b"weight"), (4, 4)),
    ((b"blocks", b"0", b"ffn", b"down", b"weight"), (4, 8)),
    ((b"blocks", b"0", b"ffn", b"up", b"weight"), (8, 4)),
    ((b"blocks", b"0", b"norm1", b"bias"), (4,)),
    ((b"blocks", b"0", b"norm1", b"weight"), (4,)),
    ((b"blocks", b"0", b"norm2", b"bias"), (4,)),
    ((b"blocks", b"0", b"norm2", b"weight"), (4,)),
    ((b"head", b"weight"), (256, 4)),
    ((b"norm_final", b"bias"), (4,)),
    ((b"norm_final", b"weight"), (4,)),
    ((b"position_embedding", b"weight"), (2, 4)),
    ((b"token_embedding", b"weight"), (256, 4)),
)


def m3_x1() -> bytes:
    """Return the canonical resolved configuration for the fixed M3 profile."""
    document = json.loads(X1)
    document["resolved"].update({
        "model.context-length": 2,
        "model.head-size": 2,
        "model.hidden-size": 4,
        "model.kv-head-count": 2,
        "model.layer-count": 1,
        "model.query-head-count": 2,
    })
    return (json.dumps(document, separators=(",", ":"), sort_keys=True) + "\n").encode()


M3_X1 = m3_x1()
M3_CONFIG = (b"sha256:eshkol-config-json-v1:" +
             hashlib.sha256(M3_X1).hexdigest().encode())


def encoded_path(segment: bytes) -> bytes:
    return struct.pack("<I", len(segment)) + segment


def encoded_segments(segments: tuple[bytes, ...]) -> bytes:
    return b"".join(encoded_path(segment) for segment in segments)


def element_count(shape: tuple[int, ...]) -> int:
    result = 1
    for extent in shape:
        result *= extent
    return result


def m3_tensor_record(segments: tuple[bytes, ...], shape: tuple[int, ...],
                     payload_offset: int, payload: bytes) -> bytes:
    encoded = encoded_segments(segments)
    record = bytearray(80 + 8 * len(shape) + len(encoded))
    p64(record, 0, len(record))
    p64(record, 8, payload_offset)
    p64(record, 16, len(payload))
    p32(record, 24, len(encoded))
    p16(record, 28, len(segments))
    p16(record, 30, len(shape))
    record[32:36] = bytes((1, 3, 1, 1))
    p64(record, 40, element_count(shape))
    for dimension, extent in enumerate(shape):
        p64(record, 80 + 8 * dimension, extent)
    record[80 + 8 * len(shape):] = encoded
    record[48:80] = hashlib.sha256(
        C1_TENSOR_DOMAIN + record[:48] + b"\0" * 32 + record[80:] + payload
    ).digest()
    return bytes(record)


def make_m3_c1(position_shape: tuple[int, int]) -> tuple[bytes, int]:
    parameters = list(M3_PARAMETERS)
    parameters[13] = (parameters[13][0], position_shape)
    records: list[bytes] = []
    payloads: list[bytes] = []
    payload_offset = 0
    head_payload = b"\0" * (4 * element_count(parameters[10][1]))
    for index, (segments, shape) in enumerate(parameters):
        payload = head_payload if index in (10, 14) else b"\0" * (
            4 * element_count(shape))
        records.append(m3_tensor_record(segments, shape, payload_offset, payload))
        payloads.append(payload)
        payload_offset += len(payload)
    aliases = struct.pack("<IIII", 2, 0, 10, 14)
    metadata = PROVIDER + b"".join(records) + aliases
    payload = b"".join(payloads)
    header = bytearray(128)
    header[:16] = C1_MAGIC
    p16(header, 16, 1)
    p32(header, 20, 128)
    p32(header, 24, 1)
    p32(header, 28, 1)
    p64(header, 40, 128 + len(metadata) + len(payload) + 32)
    p64(header, 48, 128)
    p64(header, 56, len(metadata))
    p64(header, 64, 128 + len(metadata))
    p64(header, 72, len(payload))
    p32(header, 80, len(parameters))
    p32(header, 84, 1)
    p32(header, 88, len(PROVIDER))
    p16(header, 92, 1)
    p16(header, 96, 2)
    unsigned = bytes(header) + metadata + payload
    unique_bytes = sum(4 * element_count(shape)
                       for _, shape in parameters[:14])
    return unsigned + hashlib.sha256(C1_DOMAIN + unsigned).digest(), unique_bytes


def make_m3_optimizer(position_shape: tuple[int, int]) -> tuple[bytes, bytes]:
    parameters = list(M3_PARAMETERS[:14])
    parameters[13] = (parameters[13][0], position_shape)
    group = bytearray(40 + 4 * len(parameters))
    p64(group, 0, len(group))
    p32(group, 8, len(parameters))
    for off, value in ((16, 0x3A83126F), (20, 0x3F666666),
                       (24, 0x3F7D70A4), (28, 0x322BCC77), (32, 0)):
        p32(group, off, value)
    for index in range(len(parameters)):
        p32(group, 40 + 4 * index, index)
    config = bytearray(64)
    p32(config, 0, 64)
    p32(config, 16, 0x3F800000)
    p32(config, 40, 1)
    p32(config, 44, len(parameters))
    p64(config, 48, len(group))
    config += group

    records: list[bytes] = []
    payloads: list[bytes] = []
    payload_offset = 0
    for ordinal, (segments, shape) in enumerate(parameters):
        encoded = encoded_segments(segments)
        record = bytearray(120 + 8 * len(shape) + len(encoded))
        elements = element_count(shape)
        moment_bytes = 4 * elements
        avg = b"\0" * moment_bytes
        sq = b"\0" * moment_bytes
        p64(record, 0, len(record))
        p32(record, 8, ordinal)
        p32(record, 12, len(encoded))
        p16(record, 16, len(segments))
        p16(record, 18, len(shape))
        p64(record, 24, elements)
        p64(record, 32, moment_bytes)
        p64(record, 40, payload_offset)
        p64(record, 48, payload_offset + moment_bytes)
        for dimension, extent in enumerate(shape):
            p64(record, 120 + 8 * dimension, extent)
        record[120 + 8 * len(shape):] = encoded
        zero_record = record[:56] + b"\0" * 64 + record[120:]
        record[56:88] = hashlib.sha256(
            MOMENT_DOMAIN + struct.pack("<II", ordinal, 1) + zero_record + avg
        ).digest()
        record[88:120] = hashlib.sha256(
            MOMENT_DOMAIN + struct.pack("<II", ordinal, 2) + zero_record + sq
        ).digest()
        records.append(bytes(record))
        payloads.extend((avg, sq))
        payload_offset += 2 * moment_bytes

    header = bytearray(128)
    header[:8] = b"ESHKOPT1"
    p16(header, 8, 1)
    p32(header, 12, 128)
    p32(header, 24, 1)
    p32(header, 28, 3)
    p32(header, 32, 1)
    p32(header, 36, 1)
    p16(header, 40, 2)
    p32(header, 44, len(PROVIDER))
    p32(header, 48, 1)
    p32(header, 52, len(parameters))
    p32(header, 56, 2 * len(parameters))
    p64(header, 72, len(config))
    p64(header, 80, sum(map(len, records)))
    p64(header, 88, payload_offset)
    p64(header, 96, 128 + len(PROVIDER))
    p64(header, 104, 128 + len(PROVIDER) + len(config))
    p64(header, 112, 128 + len(PROVIDER) + len(config) + sum(map(len, records)))
    return (bytes(header) + PROVIDER + bytes(config) + b"".join(records),
            b"".join(payloads))


def make_m3_artifact(position_shape: tuple[int, int]) -> tuple[bytes, int]:
    model, unique_model_bytes = make_m3_c1(position_shape)
    optimizer, moments = make_m3_optimizer(position_shape)
    current, epoch = cursor(1), cursor(0)
    metadata = (TOKENIZER + M3_CONFIG + PROVIDER + LIBRARY + COMPILER + M3_X1 +
                current + epoch + optimizer)
    payload = model + moments
    header = bytearray(256)
    header[:16] = C2_MAGIC
    p16(header, 16, 1)
    p32(header, 20, 256)
    p32(header, 24, 1)
    p32(header, 28, 1)
    p64(header, 40, 256 + len(metadata) + len(payload) + 32)
    p64(header, 48, 256)
    p64(header, 56, len(metadata))
    p64(header, 64, 256 + len(metadata))
    p64(header, 72, len(payload))
    p32(header, 80, 15)
    p32(header, 84, 14)
    p32(header, 88, 28)
    p32(header, 92, 43)
    p64(header, 96, len(model))
    p64(header, 104, len(optimizer))
    p64(header, 112, len(moments))
    p64(header, 120, len(M3_X1))
    p32(header, 128, len(current))
    p32(header, 132, len(epoch))
    p32(header, 136, len(TOKENIZER))
    p32(header, 140, len(M3_CONFIG))
    p32(header, 144, len(PROVIDER))
    p32(header, 148, len(LIBRARY))
    p32(header, 152, len(COMPILER))
    p32(header, 156, 1)
    for offset, value in zip(range(160, 184, 2),
                             (1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0),
                             strict=True):
        p16(header, offset, value)
    p64(header, 184, 2)
    p64(header, 208, 1)
    p64(header, 216, 1729)
    unsigned = bytes(header) + metadata + payload
    return unsigned + hashlib.sha256(C2_DOMAIN + unsigned).digest(), unique_model_bytes


def buffer_names(count: int, total_bytes: int) -> list[bytes]:
    names = [f"z{index:04d}".encode() for index in range(count)]
    minimum = sum(map(len, names))
    if total_bytes < minimum or total_bytes > count * 65536:
        raise ValueError("requested C1 path budget is not representable")
    remaining = total_bytes - minimum
    for index, name in enumerate(names):
        add = min(65536 - len(name), remaining)
        names[index] = name + b"a" * add
        remaining -= add
    if remaining:
        raise AssertionError("path budget distribution failed")
    return names


def payload_sizes(buffer_count: int, total_bytes: int,
                  require_exact_max: bool) -> list[int]:
    if total_bytes < 8 + 4 * buffer_count or total_bytes % 4:
        raise ValueError("requested model payload budget is not representable")
    sizes = [4] * buffer_count
    remaining = total_bytes - 8 - sum(sizes)
    for index in range(buffer_count):
        add = min(TENSOR_LIMIT - sizes[index], remaining)
        add -= add % 4
        sizes[index] += add
        remaining -= add
    if remaining or (require_exact_max and max(sizes) != TENSOR_LIMIT):
        raise ValueError("requested model payload exceeds tensor limit")
    return sizes


def tensor_record(name: bytes, kind: int, payload_offset: int,
                  payload: bytes) -> bytes:
    payload_bytes = len(payload)
    path = encoded_path(name)
    record = bytearray(80 + 8 + len(path))
    p64(record, 0, len(record))
    p64(record, 8, payload_offset)
    p64(record, 16, payload_bytes)
    p32(record, 24, len(path))
    p16(record, 28, 1)
    p16(record, 30, 1)
    record[32:36] = bytes((kind, 3, 1, 1))
    p64(record, 40, payload_bytes // 4)
    p64(record, 80, payload_bytes // 4)
    record[88:] = path
    record[48:80] = hashlib.sha256(
        C1_TENSOR_DOMAIN + record[:48] + b"\0" * 32 + record[80:] + payload
    ).digest()
    return bytes(record)


def make_c1(entry_count: int, aggregate_metadata: int | None,
            outer_metadata: int, model_payload_bytes: int | None,
            exact_max_tensor: bool = False,
            single_buffer_bytes: int | None = None) -> bytes:
    buffer_count = entry_count - 1
    if buffer_count < 1:
        raise ValueError("operational fixtures require one parameter and a buffer")
    if aggregate_metadata is None:
        names = buffer_names(buffer_count, 5 * buffer_count)
    else:
        c1_metadata = aggregate_metadata - outer_metadata
        names_total = c1_metadata - 117 - 92 * buffer_count
        names = buffer_names(buffer_count, names_total)
    if single_buffer_bytes is not None:
        sizes = [single_buffer_bytes] + [4] * (buffer_count - 1)
    else:
        if model_payload_bytes is None:
            raise ValueError("model payload size is required")
        sizes = payload_sizes(buffer_count, model_payload_bytes,
                              exact_max_tensor)

    parameter_payload = struct.pack("<II", 0x3F800000, 0x40000000)
    records = [tensor_record(b"weight", 1, 0, parameter_payload)]
    payloads = [parameter_payload]
    offset = len(parameter_payload)
    for name, size in zip(names, sizes, strict=True):
        buffer_payload = b"\0" * size
        records.append(tensor_record(name, 2, offset, buffer_payload))
        payloads.append(buffer_payload)
        offset += size
    metadata = PROVIDER + b"".join(records)
    payload = b"".join(payloads)
    header = bytearray(128)
    header[:16] = C1_MAGIC
    p16(header, 16, 1)
    p32(header, 20, 128)
    p32(header, 24, 1)
    p32(header, 28, 1)
    p64(header, 40, 128 + len(metadata) + len(payload) + 32)
    p64(header, 48, 128)
    p64(header, 56, len(metadata))
    p64(header, 64, 128 + len(metadata))
    p64(header, 72, len(payload))
    p32(header, 80, entry_count)
    p32(header, 88, len(PROVIDER))
    p16(header, 92, 1)
    p16(header, 96, 2)
    unsigned = bytes(header) + metadata + payload
    return unsigned + hashlib.sha256(C1_DOMAIN + unsigned).digest()


def outer_metadata_prefix() -> tuple[bytes, bytes, bytes]:
    optimizer, moments, _ = make_optimizer()
    current, epoch = cursor(1), cursor(0)
    fixed = (TOKENIZER + CONFIG + PROVIDER + LIBRARY + COMPILER + X1 +
             current + epoch + optimizer)
    return fixed, optimizer, moments


def make_artifact(entry_count: int, *, aggregate_metadata: int | None = None,
                  file_bytes: int | None = None,
                  single_buffer_bytes: int | None = None,
                  exact_max_tensor: bool = False) -> bytes:
    metadata, optimizer, moments = outer_metadata_prefix()
    if file_bytes is not None:
        if aggregate_metadata is None:
            raise ValueError("exact file construction needs a metadata budget")
        model_payload = file_bytes - aggregate_metadata - 464
    else:
        model_payload = None
    model = make_c1(entry_count, aggregate_metadata, len(metadata),
                    model_payload, exact_max_tensor, single_buffer_bytes)
    payload = model + moments
    header = bytearray(256)
    header[:16] = C2_MAGIC
    p16(header, 16, 1)
    p32(header, 20, 256)
    p32(header, 24, 1)
    p32(header, 28, 1)
    total = 256 + len(metadata) + len(payload) + 32
    p64(header, 40, total)
    p64(header, 48, 256)
    p64(header, 56, len(metadata))
    p64(header, 64, 256 + len(metadata))
    p64(header, 72, len(payload))
    p32(header, 80, entry_count)
    p32(header, 84, 1)
    p32(header, 88, 2)
    p32(header, 92, entry_count + 2)
    p64(header, 96, len(model))
    p64(header, 104, len(optimizer))
    p64(header, 112, len(moments))
    p64(header, 120, len(X1))
    p32(header, 128, 304)
    p32(header, 132, 304)
    p32(header, 136, len(TOKENIZER))
    p32(header, 140, len(CONFIG))
    p32(header, 144, len(PROVIDER))
    p32(header, 148, len(LIBRARY))
    p32(header, 152, len(COMPILER))
    p32(header, 156, 1)
    for offset, value in zip(range(160, 184, 2),
                             (1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0),
                             strict=True):
        p16(header, offset, value)
    p64(header, 184, 2)
    p64(header, 208, 1)
    p64(header, 216, 1729)
    unsigned = bytes(header) + metadata + payload
    artifact = unsigned + hashlib.sha256(C2_DOMAIN + unsigned).digest()
    if file_bytes is not None and len(artifact) != file_bytes:
        raise AssertionError((len(artifact), file_bytes))
    return artifact


def artifact_metadata(data: bytes) -> int:
    outer = struct.unpack_from("<Q", data, 56)[0]
    model = struct.unpack_from("<Q", data, 64)[0]
    nested = struct.unpack_from("<Q", data, model + 56)[0]
    return outer + nested


def main() -> None:
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    m3, m3_unique_bytes = make_m3_artifact((2, 4))
    c4, c4_unique_bytes = make_m3_artifact((4, 4))
    if m3_unique_bytes != 4736 or c4_unique_bytes != 4768:
        raise AssertionError((m3_unique_bytes, c4_unique_bytes))
    artifacts = {
        "joint-exact.c2": make_artifact(
            62, aggregate_metadata=METADATA_LIMIT, file_bytes=FILE_LIMIT,
            exact_max_tensor=True),
        "file-one.c2": make_artifact(
            62, aggregate_metadata=METADATA_LIMIT - 3,
            file_bytes=FILE_LIMIT + 1, exact_max_tensor=True),
        "metadata-exact.c2": make_artifact(
            9, aggregate_metadata=METADATA_LIMIT, single_buffer_bytes=4),
        "metadata-one.c2": make_artifact(
            9, aggregate_metadata=METADATA_LIMIT + 1, single_buffer_bytes=4),
        "tensor-exact.c2": make_artifact(
            2, single_buffer_bytes=TENSOR_LIMIT),
        "tensor-one.c2": make_artifact(
            2, single_buffer_bytes=TENSOR_LIMIT + 4),
        "count-exact.c2": make_artifact(62, single_buffer_bytes=4),
        "count-one.c2": make_artifact(63, single_buffer_bytes=4),
        "alias-shape.c2": bytes(
            make_format_fixture(alias_shape_mismatch=True)[0]),
        "rank-two.c2": bytes(make_format_fixture(model_shape=(1, 2))[0]),
        "m3-model.c2": m3,
        "c4-schema.c2": c4,
    }
    corrupt = bytearray(artifacts["joint-exact.c2"])
    corrupt[-1] ^= 1
    artifacts["joint-corrupt.c2"] = bytes(corrupt)

    manifest: dict[str, dict[str, int | str]] = {}
    for name, data in artifacts.items():
        (root / name).write_bytes(data)
        manifest[name] = {
            "bytes": len(data),
            "artifact_metadata_bytes": artifact_metadata(data),
            "model_tensors": struct.unpack_from("<I", data, 80)[0],
            "total_tensors": struct.unpack_from("<I", data, 92)[0],
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    manifest["m3-model.c2"]["unique_model_bytes"] = m3_unique_bytes
    manifest["c4-schema.c2"]["unique_model_bytes"] = c4_unique_bytes
    manifest["m3-model.c2"]["config_fingerprint"] = M3_CONFIG.decode()
    manifest["c4-schema.c2"]["config_fingerprint"] = M3_CONFIG.decode()
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
