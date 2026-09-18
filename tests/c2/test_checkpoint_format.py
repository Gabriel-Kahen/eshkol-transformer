#!/usr/bin/env python3
"""Independent fixture builder and hostile-input gate for the native C2 parser."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest


C2_DOMAIN = b"eshkol-training-state-container-v1\0"
C1_DOMAIN = b"eshkol-checkpoint-container-v1\0"
C1_TENSOR_DOMAIN = b"eshkol-checkpoint-tensor-v1\0"
MOMENT_DOMAIN = b"eshkol-training-state-moment-v1\0"
CURSOR_DOMAIN = b"eshkol-token-dataset-cursor-checksum-v1\n"
C2_MAGIC = bytes.fromhex("894553484b54524e53544154450d0a00")
C1_MAGIC = bytes.fromhex("894553484b4f4c434b50540d0a1a0a00")
PROVIDER = b"i2-dense-cpu-f32-v1"
LIBRARY = b"eshkol-transformer\0" + b"0.1.0-draft\0" + b"eshkol-training-state:1.0\0"
COMPILER = b"Eshkol Compiler v1.3.4-evolve\0" + b"90cbd7130f47b8184bcc77b8d5c1b0026da980de\0"
TOKENIZER = b"sha256:eshkol-byte-tokenizer-v1:" + b"0" * 64
X1 = (Path(__file__).resolve().parents[1] / "x1" / "fixtures" /
      "resolved_minimal_v1.json").read_bytes()
CONFIG = b"sha256:eshkol-config-json-v1:" + hashlib.sha256(X1).hexdigest().encode()


def p16(buf: bytearray, off: int, value: int) -> None:
    struct.pack_into("<H", buf, off, value)


def p32(buf: bytearray, off: int, value: int) -> None:
    struct.pack_into("<I", buf, off, value)


def p64(buf: bytearray, off: int, value: int) -> None:
    struct.pack_into("<Q", buf, off, value)


def u64(buf: bytes | bytearray, off: int) -> int:
    return struct.unpack_from("<Q", buf, off)[0]


def path(value: bytes) -> bytes:
    return struct.pack("<I", len(value)) + value


def c1_parameter_record(name: bytes, shape: tuple[int, ...],
                        payload_offset: int, payload: bytes) -> bytes:
    encoded_path = path(name)
    record = bytearray(80 + 8 * len(shape) + len(encoded_path))
    p64(record, 0, len(record))
    p64(record, 8, payload_offset)
    p64(record, 16, len(payload))
    p32(record, 24, len(encoded_path))
    p16(record, 28, 1)
    p16(record, 30, len(shape))
    record[32:36] = bytes((1, 3, 1, 1))
    elements = 1
    for dimension, extent in enumerate(shape):
        p64(record, 80 + 8 * dimension, extent)
        elements *= extent
    p64(record, 40, elements)
    record[80 + 8 * len(shape):] = encoded_path
    record[48:80] = hashlib.sha256(
        C1_TENSOR_DOMAIN + record[:48] + b"\0" * 32 + record[80:] + payload
    ).digest()
    return bytes(record)


def cursor(ordinal: int) -> bytes:
    fp = TOKENIZER
    header_bytes = 176 + len(fp)
    result = bytearray(header_bytes + 32)
    result[:8] = b"ESHKDCU1"
    p16(result, 8, 1)
    p32(result, 12, header_bytes)
    p32(result, 20, 1)
    p32(result, 36, len(fp))
    for off, value in ((40, 256), (48, 1), (56, 1), (64, 1), (72, 1),
                       (80, 0), (88, 17), (104, 1), (112, 2), (120, ordinal)):
        p64(result, off, value)
    p64(result, 160, len(result))
    result[176:header_bytes] = fp
    result[header_bytes:] = hashlib.sha256(CURSOR_DOMAIN + result[:header_bytes]).digest()
    return bytes(result)


def make_c1(buffer_bytes: int = 0, buffer_dtype: int = 1) -> tuple[bytes, dict[str, int]]:
    encoded_path = path(b"weight")
    record = bytearray(80 + 8 + len(encoded_path))
    payload = struct.pack("<II", 0x3F800000, 0x40000000)
    p64(record, 0, len(record))
    p64(record, 16, len(payload))
    p32(record, 24, len(encoded_path))
    p16(record, 28, 1)
    p16(record, 30, 1)
    record[32:36] = bytes((1, 3, 1, 1))
    p64(record, 40, 2)
    p64(record, 80, 2)
    record[88:] = encoded_path
    record[48:80] = hashlib.sha256(C1_TENSOR_DOMAIN + record[:48] + b"\0" * 32 +
                                   record[80:] + payload).digest()
    records = bytes(record)
    if buffer_bytes:
        width = 1 if buffer_dtype == 1 else (8 if buffer_dtype == 2 else 4)
        if buffer_dtype not in (1, 2, 3) or buffer_bytes % width:
            raise ValueError("buffer dtype/byte count is not canonical")
        buffer_path = path(b"zz-buffer")
        buffer_record = bytearray(80 + 8 + len(buffer_path))
        p64(buffer_record, 0, len(buffer_record))
        p64(buffer_record, 8, len(payload))
        p64(buffer_record, 16, buffer_bytes)
        p32(buffer_record, 24, len(buffer_path))
        p16(buffer_record, 28, 1)
        p16(buffer_record, 30, 1)
        buffer_record[32:36] = bytes((2, buffer_dtype, 1, 1))
        elements = buffer_bytes // width
        p64(buffer_record, 40, elements)
        p64(buffer_record, 80, elements)
        buffer_record[88:] = buffer_path
        buffer_payload = b"\0" * buffer_bytes
        buffer_record[48:80] = hashlib.sha256(
            C1_TENSOR_DOMAIN + buffer_record[:48] + b"\0" * 32 +
            buffer_record[80:] + buffer_payload).digest()
        records += bytes(buffer_record)
        payload += buffer_payload
    metadata = PROVIDER + records
    header = bytearray(128)
    header[:16] = C1_MAGIC
    p16(header, 16, 1)
    p32(header, 20, 128)
    p32(header, 24, 1)
    p32(header, 28, 1)
    total = 128 + len(metadata) + len(payload) + 32
    p64(header, 40, total)
    p64(header, 48, 128)
    p64(header, 56, len(metadata))
    p64(header, 64, 128 + len(metadata))
    p64(header, 72, len(payload))
    p32(header, 80, 1 + bool(buffer_bytes))
    p32(header, 88, len(PROVIDER))
    p16(header, 92, 1)
    p16(header, 96, 2)
    unsigned = bytes(header) + metadata + payload
    return unsigned + hashlib.sha256(C1_DOMAIN + unsigned).digest(), {
        "record": 128 + len(PROVIDER),
        "payload": 128 + len(metadata),
    }


def make_alias_shape_mismatch_c1() -> tuple[bytes, dict[str, int]]:
    payload = struct.pack("<I", 0x3F800000)
    first = c1_parameter_record(b"weight", (), 0, payload)
    second = c1_parameter_record(b"weight_tied", (1,), len(payload), payload)
    aliases = struct.pack("<IIII", 2, 0, 0, 1)
    metadata = PROVIDER + first + second + aliases
    header = bytearray(128)
    header[:16] = C1_MAGIC
    p16(header, 16, 1)
    p32(header, 20, 128)
    p32(header, 24, 1)
    p32(header, 28, 1)
    total = 128 + len(metadata) + 2 * len(payload) + 32
    p64(header, 40, total)
    p64(header, 48, 128)
    p64(header, 56, len(metadata))
    p64(header, 64, 128 + len(metadata))
    p64(header, 72, 2 * len(payload))
    p32(header, 80, 2)
    p32(header, 84, 1)
    p32(header, 88, len(PROVIDER))
    p16(header, 92, 1)
    p16(header, 96, 2)
    unsigned = bytes(header) + metadata + payload + payload
    return unsigned + hashlib.sha256(C1_DOMAIN + unsigned).digest(), {
        "record": 128 + len(PROVIDER),
        "payload": 128 + len(metadata),
    }


def make_shaped_c1(shape: tuple[int, ...]) -> tuple[bytes, dict[str, int]]:
    elements = 1
    for extent in shape:
        elements *= extent
    payload = b"\0" * (4 * elements)
    record = c1_parameter_record(b"weight", shape, 0, payload)
    metadata = PROVIDER + record
    header = bytearray(128)
    header[:16] = C1_MAGIC
    p16(header, 16, 1)
    p32(header, 20, 128)
    p32(header, 24, 1)
    p32(header, 28, 1)
    total = 128 + len(metadata) + len(payload) + 32
    p64(header, 40, total)
    p64(header, 48, 128)
    p64(header, 56, len(metadata))
    p64(header, 64, 128 + len(metadata))
    p64(header, 72, len(payload))
    p32(header, 80, 1)
    p32(header, 88, len(PROVIDER))
    p16(header, 92, 1)
    p16(header, 96, 2)
    unsigned = bytes(header) + metadata + payload
    return unsigned + hashlib.sha256(C1_DOMAIN + unsigned).digest(), {
        "record": 128 + len(PROVIDER),
        "payload": 128 + len(metadata),
    }


def make_optimizer(shape: tuple[int, ...] = (2,)
                   ) -> tuple[bytes, bytes, dict[str, int]]:
    group = bytearray(44)
    p64(group, 0, len(group))
    p32(group, 8, 1)
    for off, value in ((16, 0x3A83126F), (20, 0x3F666666),
                       (24, 0x3F7D70A4), (28, 0x322BCC77), (32, 0)):
        p32(group, off, value)
    config = bytearray(64)
    p32(config, 0, 64)
    p32(config, 16, 0x3F800000)
    p32(config, 40, 1)
    p32(config, 44, 1)
    p64(config, 48, len(group))
    config += group
    encoded_path = path(b"weight")
    record = bytearray(120 + 8 * len(shape) + len(encoded_path))
    p64(record, 0, len(record))
    p32(record, 12, len(encoded_path))
    p16(record, 16, 1)
    p16(record, 18, len(shape))
    elements = 1
    for dimension, extent in enumerate(shape):
        p64(record, 120 + 8 * dimension, extent)
        elements *= extent
    moment_bytes = 4 * elements
    p64(record, 24, elements)
    p64(record, 32, moment_bytes)
    p64(record, 48, moment_bytes)
    record[120 + 8 * len(shape):] = encoded_path
    payload = b"\0" * (2 * moment_bytes)
    zero_record = record[:56] + b"\0" * 64 + record[120:]
    record[56:88] = hashlib.sha256(MOMENT_DOMAIN + struct.pack("<II", 0, 1) +
                                   zero_record + payload[:moment_bytes]).digest()
    record[88:120] = hashlib.sha256(MOMENT_DOMAIN + struct.pack("<II", 0, 2) +
                                    zero_record + payload[moment_bytes:]).digest()
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
    p32(header, 52, 1)
    p32(header, 56, 2)
    p64(header, 72, len(config))
    p64(header, 80, len(record))
    p64(header, 88, len(payload))
    p64(header, 96, 128 + len(PROVIDER))
    p64(header, 104, 128 + len(PROVIDER) + len(config))
    p64(header, 112, 128 + len(PROVIDER) + len(config) + len(record))
    metadata = bytes(header) + PROVIDER + bytes(config) + bytes(record)
    return metadata, payload, {
        "config": 128 + len(PROVIDER),
        "group": 128 + len(PROVIDER) + 64,
        "record": 128 + len(PROVIDER) + len(config),
    }


def make_fixture(buffer_bytes: int = 0, buffer_dtype: int = 1,
                 alias_shape_mismatch: bool = False,
                 model_shape: tuple[int, ...] | None = None
                 ) -> tuple[bytearray, dict[str, int]]:
    if alias_shape_mismatch:
        model, c1 = make_alias_shape_mismatch_c1()
    elif model_shape is not None:
        model, c1 = make_shaped_c1(model_shape)
    else:
        model, c1 = make_c1(buffer_bytes, buffer_dtype)
    optimizer_shape = (
        () if alias_shape_mismatch
        else (2,) if model_shape is None
        else model_shape
    )
    optimizer, moments, o2 = make_optimizer(optimizer_shape)
    x1 = X1
    current, epoch = cursor(1), cursor(0)
    metadata = TOKENIZER + CONFIG + PROVIDER + LIBRARY + COMPILER + x1 + current + epoch + optimizer
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
    p32(header, 80, 2 if alias_shape_mismatch else 1 + bool(buffer_bytes))
    p32(header, 84, 1)
    p32(header, 88, 2)
    p32(header, 92, 4 if alias_shape_mismatch else 3 + bool(buffer_bytes))
    p64(header, 96, len(model))
    p64(header, 104, len(optimizer))
    p64(header, 112, len(moments))
    p64(header, 120, len(x1))
    p32(header, 128, len(current))
    p32(header, 132, len(epoch))
    p32(header, 136, len(TOKENIZER))
    p32(header, 140, len(CONFIG))
    p32(header, 144, len(PROVIDER))
    p32(header, 148, len(LIBRARY))
    p32(header, 152, len(COMPILER))
    p32(header, 156, 1)
    for off, value in zip(range(160, 184, 2), (1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0)):
        p16(header, off, value)
    p64(header, 184, 2)
    p64(header, 192, 0)
    p64(header, 200, 0)
    p64(header, 208, 1)
    p64(header, 216, 1729)
    unsigned = bytes(header) + metadata + payload
    data = bytearray(unsigned + hashlib.sha256(C2_DOMAIN + unsigned).digest())
    base_metadata = 256
    x1_off = base_metadata + len(TOKENIZER + CONFIG + PROVIDER + LIBRARY + COMPILER)
    current_off = x1_off + len(x1)
    epoch_off = current_off + len(current)
    optimizer_off = epoch_off + len(epoch)
    model_off = 256 + len(metadata)
    moment_off = model_off + len(model)
    offsets = {
        "tokenizer": 256,
        "config_fp": 256 + len(TOKENIZER),
        "provider": 256 + len(TOKENIZER + CONFIG),
        "library": 256 + len(TOKENIZER + CONFIG + PROVIDER),
        "compiler": 256 + len(TOKENIZER + CONFIG + PROVIDER + LIBRARY),
        "x1": x1_off,
        "current": current_off,
        "epoch": epoch_off,
        "optimizer": optimizer_off,
        "model": model_off,
        "moments": moment_off,
    }
    offsets.update({f"c1_{k}": model_off + v for k, v in c1.items()})
    offsets.update({f"o2_{k}": optimizer_off + v for k, v in o2.items()})
    return data, offsets


def resign_outer(data: bytearray) -> None:
    data[-32:] = hashlib.sha256(C2_DOMAIN + data[:-32]).digest()


def resign_cursor(data: bytearray, off: int) -> None:
    header = struct.unpack_from("<I", data, off + 12)[0]
    data[off + header:off + header + 32] = hashlib.sha256(
        CURSOR_DOMAIN + data[off:off + header]).digest()


def resign_c1(data: bytearray, off: int) -> None:
    length = u64(data, off + 40)
    data[off + length - 32:off + length] = hashlib.sha256(
        C1_DOMAIN + data[off:off + length - 32]).digest()


def resign_moment(data: bytearray, record: int, payload: int, kind: int) -> None:
    record_bytes = u64(data, record)
    moment_bytes = u64(data, record + 32)
    relative = u64(data, record + (40 if kind == 1 else 48))
    zero_record = data[record:record + 56] + b"\0" * 64 + data[record + 120:record + record_bytes]
    digest = hashlib.sha256(MOMENT_DOMAIN + struct.pack("<II", 0, kind) +
                            zero_record + data[payload + relative:payload + relative + moment_bytes]).digest()
    start = record + (56 if kind == 1 else 88)
    data[start:start + 32] = digest


def resign_config_fingerprint(data: bytearray, x1: int, config_fp: int) -> None:
    length = u64(data, 120)
    digest = hashlib.sha256(data[x1:x1 + length]).hexdigest().encode()
    data[config_fp + 29:config_fp + 93] = digest


class ParserTests(unittest.TestCase):
    driver: Path

    def invoke(self, data: bytes, *args: int) -> tuple[int, int, int, int]:
        with tempfile.NamedTemporaryFile() as f:
            f.write(data)
            f.flush()
            run = subprocess.run([str(self.driver), f.name, *(str(a) for a in args)],
                                 text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, check=False)
        fields = run.stdout.strip().split()
        self.assertGreaterEqual(len(fields), 3, run.stderr)
        return int(fields[0]), int(fields[1]), int(fields[2]), run.returncode

    def test_valid_and_limits(self) -> None:
        data, _ = make_fixture()
        self.assertEqual(self.invoke(data)[:2], (0, 0))
        self.assertEqual(self.invoke(data, 1, 0, 7)[0], 2)
        self.assertEqual(self.invoke(data, 1, 0, 8, 2)[0], 2)
        self.assertEqual(self.invoke(data, 1, 0, 8, 3, 100)[0], 2)
        exact, _ = make_fixture(8 * 1024 * 1024, 3)
        self.assertEqual(self.invoke(exact)[0], 0)
        one_over, _ = make_fixture(8 * 1024 * 1024 + 4, 3)
        self.assertEqual(self.invoke(one_over)[0], 4)
        one_over, one_over_off = make_fixture(8 * 1024 * 1024 + 4, 3)
        p32(one_over, one_over_off["o2_group"] + 40, 1)
        resign_outer(one_over)
        self.assertEqual(self.invoke(one_over)[0], 2)

    def test_header_and_checksum_matrix(self) -> None:
        data, _ = make_fixture()
        cases: list[tuple[bytearray, int]] = []
        for off, expected in ((0, 2), (20, 2), (240, 2)):
            item = data.copy(); item[off] ^= 1; cases.append((item, expected))
        item = data.copy(); p16(item, 16, 2); cases.append((item, 3))
        item = data.copy(); item[-1] ^= 1; cases.append((item, 2))
        item = data[:-1]; cases.append((item, 2))
        item = data + b"x"; cases.append((bytearray(item), 2))
        item = data.copy(); p64(item, 56, (1 << 64) - 1); cases.append((item, 2))
        for raw, expected in cases:
            self.assertEqual(self.invoke(raw)[0], expected)

    def test_identity_classification(self) -> None:
        data, off = make_fixture()
        item = data.copy(); item[off["provider"]:off["provider"] + 19] = b"x2-dense-cpu-f32-v1"; resign_outer(item)
        self.assertEqual(self.invoke(item)[0], 4)
        item = data.copy(); item[off["provider"]] = 0xff; resign_outer(item)
        self.assertEqual(self.invoke(item)[0], 2)
        item = data.copy(); item[off["library"]] = ord("E"); resign_outer(item)
        self.assertEqual(self.invoke(item)[0], 3)
        item = data.copy(); item[off["compiler"] + 17] = ord("9"); resign_outer(item)
        self.assertEqual(self.invoke(item, 1)[0], 4)
        self.assertEqual(self.invoke(item, 2)[0], 5)

    def test_cursor_semantics_before_success(self) -> None:
        data, off = make_fixture()
        item = data.copy(); p64(item, off["epoch"] + 120, 2); resign_cursor(item, off["epoch"]); resign_outer(item)
        self.assertEqual(self.invoke(item)[0], 2)
        item = data.copy(); item[off["current"] + 128] ^= 1; resign_cursor(item, off["current"]); resign_outer(item)
        self.assertEqual(self.invoke(item)[0], 2)
        item = data.copy(); item[off["current"] + 200] ^= 1; resign_outer(item)
        self.assertEqual(self.invoke(item)[0], 2)
        item = data.copy(); p64(item, off["current"] + 72, (1 << 64) - 1)
        p64(item, off["epoch"] + 72, (1 << 64) - 1)
        resign_cursor(item, off["current"]); resign_cursor(item, off["epoch"]); resign_outer(item)
        self.assertEqual(self.invoke(item)[0], 2)
        item = data.copy(); p64(item, off["current"] + 40, 257); p64(item, off["epoch"] + 40, 257)
        resign_cursor(item, off["current"]); resign_cursor(item, off["epoch"]); resign_outer(item)
        self.assertEqual(self.invoke(item)[0], 2)

    def test_x1_classification_adapter(self) -> None:
        data, off = make_fixture()

        item = data.copy(); item[off["x1"]] ^= 1; resign_outer(item)
        self.assertEqual(self.invoke(item)[:3], (2, 10, off["x1"]))

        item = data.copy(); item[off["x1"]] = 0xff; resign_outer(item)
        self.assertEqual(self.invoke(item)[:3], (2, 11, off["x1"]))

        for marker, new, expected in (
                (b'"model.device":"cpu"', b"gpu", (8, 5)),
                (b'"model.dtype":"f32"', b"f64", (7, 5))):
            item = data.copy(); start = item.index(marker, off["x1"]) + len(marker) - 4
            item[start:start + 3] = new
            resign_config_fingerprint(item, off["x1"], off["config_fp"])
            resign_outer(item)
            self.assertEqual(self.invoke(item)[:3], (*expected, off["x1"]))

        item = data.copy(); version = item.index(b'"format-version":[1,0]', off["x1"])
        item[version + 18] = ord("2")
        resign_config_fingerprint(item, off["x1"], off["config_fp"]); resign_outer(item)
        self.assertEqual(self.invoke(item)[:3], (3, 4, off["x1"]))

        item = data.copy(); version = item.index(b'"format-version":[1,0]', off["x1"])
        item[version + 17:version + 22] = b'"2.0"'
        resign_config_fingerprint(item, off["x1"], off["config_fp"]); resign_outer(item)
        self.assertEqual(self.invoke(item)[:3], (2, 5, off["x1"]))

        item = data.copy(); hidden = item.index(b'"model.hidden-size":64', off["x1"])
        item[hidden + 20:hidden + 22] = b"65"
        resign_config_fingerprint(item, off["x1"], off["config_fp"]); resign_outer(item)
        self.assertEqual(self.invoke(item)[:3], (2, 5, off["x1"]))

    def test_nested_and_optimizer_adversaries(self) -> None:
        data, off = make_fixture()
        alias_shape, alias_off = make_fixture(alias_shape_mismatch=True)
        self.assertEqual(
            self.invoke(alias_shape)[:3],
            (2, 15, alias_off["c1_record"] + 205),
        )
        item = data.copy(); item[off["c1_record"] + 36] = 1
        resign_c1(item, off["model"]); resign_outer(item)
        self.assertEqual(self.invoke(item)[0], 2)
        item = data.copy(); p32(item, off["o2_group"] + 40, 1); resign_outer(item)
        self.assertEqual(self.invoke(item)[0], 2)
        item = data.copy(); item[off["o2_record"] + 132] = ord("X"); resign_outer(item)
        self.assertEqual(self.invoke(item)[0], 2)
        item = data.copy(); item[off["o2_record"] + 56] ^= 1; resign_outer(item)
        self.assertEqual(self.invoke(item)[0], 2)
        item = data.copy(); p32(item, off["moments"], 0x7F800000)
        resign_moment(item, off["o2_record"], off["moments"], 1); resign_outer(item)
        self.assertEqual(self.invoke(item)[0], 2)
        item = data.copy(); p32(item, off["moments"] + 8, 0x80000000)
        resign_moment(item, off["o2_record"], off["moments"], 2); resign_outer(item)
        self.assertEqual(self.invoke(item)[:2], (0, 0))
        item = data.copy(); p32(item, off["moments"] + 8, 0x80000001)
        resign_moment(item, off["o2_record"], off["moments"], 2); resign_outer(item)
        self.assertEqual(self.invoke(item)[:2], (2, 18))
        item = data.copy(); item[off["x1"]] ^= 1; resign_outer(item)
        self.assertEqual(self.invoke(item)[0], 2)
        item = data.copy(); device = item.index(b'"model.device":"cpu"', off["x1"])
        item[device + 16:device + 19] = b"gpu"
        resign_config_fingerprint(item, off["x1"], off["config_fp"]); resign_outer(item)
        self.assertEqual(self.invoke(item)[0], 8)
        item = data.copy(); provenance = item.index(b'"model.head-size":"derived"', off["x1"])
        item[provenance + 19:provenance + 26] = b"default"
        resign_config_fingerprint(item, off["x1"], off["config_fp"]); resign_outer(item)
        self.assertEqual(self.invoke(item)[0], 2)
        item = data.copy(); version = item.index(b'"format-version":[1,0]', off["x1"])
        item[version + 18] = ord("2")
        resign_config_fingerprint(item, off["x1"], off["config_fp"]); resign_outer(item)
        self.assertEqual(self.invoke(item)[0], 3)
        item = data.copy(); version = item.index(b'"format-version":[1,0]', off["x1"])
        item[version + 17:version + 22] = b'"2.0"'
        resign_config_fingerprint(item, off["x1"], off["config_fp"]); resign_outer(item)
        self.assertEqual(self.invoke(item)[0], 2)
        item = data.copy(); p64(item, 216, 1730); resign_outer(item)
        self.assertEqual(self.invoke(item)[0], 2)

    def test_random_mutations_are_rejected_or_safe(self) -> None:
        data, _ = make_fixture()
        randomizer = random.Random(0xC2)
        for _ in range(96):
            item = data.copy()
            for _ in range(randomizer.randrange(1, 5)):
                position = randomizer.randrange(len(item))
                item[position] ^= 1 << randomizer.randrange(8)
            status, _, _, _ = self.invoke(item)
            self.assertIn(status, range(0, 10))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("driver", nargs="?", type=Path)
    parser.add_argument("--emit", type=Path)
    parser.add_argument("--variant", choices=("base", "counter", "c1-header",
                                                "profile-exact", "profile-one",
                                                "profile-corrupt"),
                        default="base")
    args = parser.parse_args()
    if args.emit:
        if args.variant == "profile-exact":
            data, offsets = make_fixture(8 * 1024 * 1024, 3)
        elif args.variant in ("profile-one", "profile-corrupt"):
            data, offsets = make_fixture(8 * 1024 * 1024 + 4, 3)
        else:
            data, offsets = make_fixture()
        if args.variant == "counter":
            p64(data, 224, 1)
            resign_outer(data)
        elif args.variant == "c1-header":
            data[offsets["model"] + 100] = 1
            resign_c1(data, offsets["model"])
            resign_outer(data)
        elif args.variant == "profile-corrupt":
            p32(data, offsets["o2_group"] + 40, 1)
            resign_outer(data)
        args.emit.write_bytes(data)
        return 0
    if args.driver is None:
        parser.error("driver is required unless --emit is used")
    ParserTests.driver = args.driver.resolve()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ParserTests)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
