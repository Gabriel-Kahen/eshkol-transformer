#!/usr/bin/env python3
"""Bounded data-only C1 1.0/F32 GGUF v3 interoperability; not a model loader."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import struct
import sys

from safetensors_interop import (InteropError, Limits, State, Tensor, U64_MAX,
                                decode_c1, elements, encode_c1, make_manifest,
                                publish, read_file, require, state_from_manifest, text_bytes)

VERSION_KEY = "eshkol.io5.gguf_version"
MANIFEST_KEY = "eshkol.io5.manifest"
DIGEST_KEY = "eshkol.io5.c1_sha256"
RESERVED = {VERSION_KEY, MANIFEST_KEY, DIGEST_KEY}
KEY = re.compile(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*", re.ASCII)
ALIGNMENT = 32
MAX_METADATA_PAIRS = 128
STRING, UINT32 = 8, 4


def aligned(value: int) -> int:
    require(0 <= value <= U64_MAX - 31, "GGUF aligned size overflow")
    return (value + 31) & -32


def tensor_size(shape: tuple[int, ...], limits: Limits) -> int:
    require(1 <= len(shape) <= 4, "unsupported GGUF rank: require 1..4")
    require(all(type(extent) is int and 0 < extent <= (1 << 63) - 1 for extent in shape),
            "unsupported GGUF extents: require positive signed-i64")
    size = elements(shape) * 4
    require(size <= limits.tensor, "GGUF tensor byte limit")
    return size


def string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


class Reader:
    def __init__(self, raw: bytes, limits: Limits):
        self.raw, self.at = raw, 24
        self.end = min(len(raw), limits.metadata)

    def take(self, length: int) -> bytes:
        require(0 <= length <= self.end - self.at, "GGUF metadata/info span or limit")
        result = self.raw[self.at:self.at + length]
        self.at += length
        return result

    def number(self, code: str) -> int:
        return struct.unpack("<" + code, self.take(struct.calcsize("<" + code)))[0]

    def text(self, maximum: int, *, nonempty: bool = False) -> str:
        length = self.number("Q")
        require((not nonempty or length > 0) and length <= maximum, "GGUF string length")
        try:
            return self.take(length).decode("utf-8")
        except UnicodeError as error:
            raise InteropError("GGUF invalid UTF-8") from error


def encode_gguf_tensors(named: dict[str, Tensor], fields: dict[str, str], limits: Limits) -> bytes:
    infos, payloads, offset = [], [], 0
    for name, tensor in named.items():
        size = tensor_size(tensor.shape, limits)
        text_bytes(name, 63, "GGUF tensor name")
        require("\0" not in name, "GGUF tensor name")
        # GGML's first dimension is fastest varying; C1's last is fastest varying.
        infos.append(string(name) + struct.pack("<I", len(tensor.shape))
                     + struct.pack("<" + "Q" * len(tensor.shape), *reversed(tensor.shape))
                     + struct.pack("<IQ", 0, offset))
        padded = aligned(size)
        require(offset + padded <= limits.file, "GGUF payload/file limit")
        payloads.extend((tensor.data, bytes(padded - size)))
        offset += padded
    metadata = {key: (STRING, value) for key, value in fields.items()}
    metadata.update({"general.architecture": (STRING, "eshkol"),
                     "general.alignment": (UINT32, ALIGNMENT)})
    prefix = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(named), len(metadata)))
    for key, (kind, value) in sorted(metadata.items()):
        encoded = string(key) + struct.pack("<I", kind)
        encoded += string(value) if kind == STRING else struct.pack("<I", value)
        require(len(prefix) + len(encoded) <= limits.metadata, "GGUF metadata/info limit")
        prefix += encoded
    for info in infos:
        require(len(prefix) + len(info) <= limits.metadata, "GGUF metadata/info limit")
        prefix += info
    data_at = aligned(len(prefix))
    require(data_at <= limits.metadata and data_at + offset <= limits.file, "GGUF prefix/file limit")
    prefix += bytes(data_at - len(prefix))
    return bytes(prefix) + b"".join(payloads)


def encode_gguf(raw_c1: bytes, limits: Limits = Limits()) -> bytes:
    state = decode_c1(raw_c1, limits)
    names = tuple(f"t{index:06d}" for index in range(len(state.tensors)))
    manifest = make_manifest(state, names)
    require(len(manifest) <= limits.metadata, "GGUF manifest limit")
    fields = {VERSION_KEY: "1.0", MANIFEST_KEY: manifest,
              DIGEST_KEY: hashlib.sha256(raw_c1).hexdigest()}
    return encode_gguf_tensors(dict(zip(names, state.tensors)), fields, limits)


def decode_gguf_tensors(raw: bytes, limits: Limits) -> tuple[dict[str, Tensor], dict[str, str | int]]:
    require(24 <= len(raw) <= limits.file, "GGUF file size limit")
    magic, version, tensor_count, metadata_count = struct.unpack_from("<4sIQQ", raw)
    require(magic == b"GGUF", "GGUF magic")
    require(version == 3, "unsupported GGUF version/endian: require v3 little endian")
    require(tensor_count <= limits.entries and metadata_count <= MAX_METADATA_PAIRS, "GGUF count limit")
    reader = Reader(raw, limits)
    metadata = {}
    for _ in range(metadata_count):
        key = reader.text(128, nonempty=True)
        require(KEY.fullmatch(key) is not None and key not in metadata, "GGUF metadata key/duplicate")
        require(not key.startswith("split."), "unsupported GGUF shards")
        require(key != "general.quantization_version", "unsupported GGUF quantization metadata")
        kind = reader.number("I")
        if key == "general.alignment":
            require(kind == UINT32 and reader.number("I") == ALIGNMENT,
                    "unsupported GGUF alignment: require UINT32 32")
            metadata[key] = ALIGNMENT
        else:
            require(kind == STRING, "unsupported GGUF metadata type: require STRING")
            metadata[key] = reader.text(limits.metadata)
    architecture = metadata.get("general.architecture")
    require(type(architecture) is str and re.fullmatch(r"[a-z0-9]+", architecture, re.ASCII) is not None,
            "GGUF architecture string required")
    descriptors = {}
    for _ in range(tensor_count):
        name = reader.text(63, nonempty=True)
        require("\0" not in name and name not in descriptors, "GGUF tensor name/duplicate")
        rank = reader.number("I")
        require(1 <= rank <= 4, "unsupported GGUF rank: require 1..4")
        shape = tuple(reversed(tuple(reader.number("Q") for _ in range(rank))))
        size = tensor_size(shape, limits)
        require(reader.number("I") == 0, "unsupported GGUF tensor type: require F32")
        offset = reader.number("Q")
        require(offset % ALIGNMENT == 0, "GGUF tensor offset alignment")
        descriptors[name] = (shape, offset, size)
    data_at = aligned(reader.at)
    require(data_at <= len(raw) and data_at <= limits.metadata, "GGUF data start span/limit")
    require(not any(raw[reader.at:data_at]), "GGUF nonzero prefix padding")
    cursor = 0
    for shape, offset, size in descriptors.values():
        require(offset == cursor, "GGUF noncontiguous tensor offsets")
        require(offset <= len(raw) - data_at and aligned(size) <= len(raw) - data_at - offset,
                "GGUF tensor data/padding span")
        end = offset + size
        cursor = offset + aligned(size)
        require(not any(raw[data_at + end:data_at + cursor]), "GGUF nonzero tensor padding")
    require(data_at + cursor == len(raw), "GGUF exact EOF")
    named = {name: Tensor((name,), "parameter", shape, raw[data_at + offset:data_at + offset + size])
             for name, (shape, offset, size) in descriptors.items()}
    return named, metadata


def decode_gguf(raw: bytes, limits: Limits = Limits(), *, plain_provider: str | None = None) -> bytes:
    named, metadata = decode_gguf_tensors(raw, limits)
    reserved = {key for key in metadata if key.startswith("eshkol.io5.")}
    if reserved:
        require(plain_provider is None, "plain import cannot override IO5 metadata")
        require(reserved == RESERVED and metadata[VERSION_KEY] == "1.0", "unsupported IO5 GGUF metadata version/fields")
        require(metadata["general.architecture"] == "eshkol", "IO5 GGUF architecture profile")
        state, _ = state_from_manifest(metadata[MANIFEST_KEY], named, limits)
        result = encode_c1(state, limits)
        require(metadata[DIGEST_KEY] == hashlib.sha256(result).hexdigest(), "IO5 GGUF C1 checksum")
        return result
    require(plain_provider is not None, "plain GGUF requires explicit --plain-provider")
    text_bytes(plain_provider, 127, "provider")
    tensors = tuple(named[name] for name in sorted(named, key=lambda value: value.encode("utf-8")))
    return encode_c1(State(plain_provider, tensors), limits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=("c1-to-gguf", "gguf-to-c1"))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--plain-provider", help="explicit plain F32 import: one name segment, parameter, no aliases")
    parser.add_argument("--overwrite", action="store_true")
    for name, value in vars(Limits()).items():
        parser.add_argument("--max-" + name, type=int, default=value)
    arguments = parser.parse_args()
    try:
        limits = Limits(**{name: getattr(arguments, "max_" + name) for name in vars(Limits())})
        raw = read_file(arguments.source, limits)
        if arguments.direction == "c1-to-gguf":
            require(arguments.plain_provider is None, "--plain-provider is import-only")
            result = encode_gguf(raw, limits)
        else:
            result = decode_gguf(raw, limits, plain_provider=arguments.plain_provider)
        publish(arguments.destination, result, arguments.overwrite)
        return 0
    except (InteropError, OSError) as error:
        print(f"IO5 GGUF error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
