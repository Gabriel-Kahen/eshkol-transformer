#!/usr/bin/env python3
"""Bounded data-only C1 1.0/F32 safetensors interoperability; not a trainer API."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile

MAGIC = bytes.fromhex("894553484b4f4c434b50540d0a1a0a00")
WHOLE_DOMAIN = b"eshkol-checkpoint-container-v1\0"
TENSOR_DOMAIN = b"eshkol-checkpoint-tensor-v1\0"
U64_MAX = (1 << 64) - 1
VERSION_KEY = "eshkol.io5.version"
MANIFEST_KEY = "eshkol.io5.manifest"
DIGEST_KEY = "eshkol.io5.c1-sha256"
RESERVED = {VERSION_KEY, MANIFEST_KEY, DIGEST_KEY}


class InteropError(ValueError):
    """Untrusted format rejection; no tensor or executable deserialization."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InteropError(message)


@dataclass(frozen=True)
class Limits:
    file: int = 64 * 1024 * 1024
    metadata: int = 1024 * 1024
    tensor: int = 32 * 1024 * 1024
    entries: int = 4096

    def __post_init__(self) -> None:
        for value, ceiling in ((self.file, 64 * 1024 * 1024),
                               (self.metadata, 1024 * 1024),
                               (self.tensor, 32 * 1024 * 1024), (self.entries, 4096)):
            require(type(value) is int and 0 <= value <= ceiling, "invalid lowering policy")


@dataclass(frozen=True)
class Tensor:
    path: tuple[str, ...]
    kind: str
    shape: tuple[int, ...]
    data: bytes


@dataclass(frozen=True)
class State:
    provider: str  # Inert spelling, never used for provider lookup.
    tensors: tuple[Tensor, ...]
    aliases: tuple[tuple[int, ...], ...] = ()


def text_bytes(value: object, maximum: int, label: str) -> bytes:
    require(type(value) is str, label + " must be a string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as error:
        raise InteropError(label + " invalid UTF-8") from error
    require(1 <= len(encoded) <= maximum, label + " length")
    return encoded


def uint(value: object) -> int:
    require(type(value) is int and 0 <= value <= U64_MAX, "expected u64 integer")
    return value


def elements(shape: tuple[int, ...]) -> int:
    require(len(shape) <= 64, "tensor rank limit")
    product = 1
    for extent in shape:
        uint(extent)
        require(not extent or product <= U64_MAX // extent, "shape product overflow")
        product *= extent
    require(product <= U64_MAX // 4, "tensor byte size overflow")
    return product


def validate_state(state: State, limits: Limits) -> None:
    text_bytes(state.provider, 127, "provider")
    require(len(state.tensors) <= limits.entries, "tensor count limit")
    previous = None
    for tensor in state.tensors:
        require(1 <= len(tensor.path) <= 64, "path segment count")
        path = tuple(text_bytes(part, 65536, "path segment") for part in tensor.path)
        require(previous is None or previous < path, "duplicate or unordered path")
        previous = path
        require(tensor.kind in ("parameter", "buffer"), "entry kind")
        require(type(tensor.data) is bytes, "tensor data must be bytes")
        require(len(tensor.data) == elements(tensor.shape) * 4, "shape/payload mismatch")
        require(len(tensor.data) <= limits.tensor, "tensor byte limit")
    require(len(state.aliases) <= 2048, "alias group limit")
    seen = set()
    previous_first = -1
    members_total = 0
    for group in state.aliases:
        require(len(group) >= 2, "alias minimum")
        members_total += len(group)
        require(members_total <= 4096, "alias member limit")
        for index in group:
            uint(index)
            require(index < len(state.tensors), "alias index")
        require(tuple(sorted(set(group))) == group and group[0] > previous_first,
                "alias ordering")
        first = state.tensors[group[0]]
        for index in group:
            tensor = state.tensors[index]
            require(index not in seen and tensor.kind == "parameter", "alias membership")
            require(tensor.shape == first.shape and tensor.data == first.data,
                    "alias shape/value conflict")
            seen.add(index)
        previous_first = group[0]


def digest(*parts: bytes | bytearray) -> bytes:
    result = hashlib.sha256()
    for part in parts:
        result.update(part)
    return result.digest()


def encode_c1(state: State, limits: Limits = Limits()) -> bytes:
    validate_state(state, limits)
    metadata = bytearray(state.provider.encode("utf-8"))
    payload_size = 0
    for tensor in state.tensors:
        path = bytearray()
        for part in tensor.path:
            encoded = part.encode("utf-8")
            path += struct.pack("<I", len(encoded)) + encoded
        record = bytearray(80)
        record += struct.pack("<" + "Q" * len(tensor.shape), *tensor.shape) + path
        struct.pack_into("<QQQIHHBBBBIQ", record, 0, len(record), payload_size,
                         len(tensor.data), len(path), len(tensor.path), len(tensor.shape),
                         1 if tensor.kind == "parameter" else 2, 3, 1, 1, 0,
                         elements(tensor.shape))
        record[48:80] = digest(TENSOR_DOMAIN, record, tensor.data)
        require(len(metadata) + len(record) <= limits.metadata, "C1 metadata limit")
        metadata += record
        payload_size += len(tensor.data)
        require(160 + len(metadata) + payload_size <= limits.file, "C1 file limit")
    for group in state.aliases:
        require(len(metadata) + 8 + 4 * len(group) <= limits.metadata, "C1 metadata limit")
        metadata += struct.pack("<II", len(group), 0)
        metadata += struct.pack("<" + "I" * len(group), *group)
    require(len(metadata) <= limits.metadata, "C1 metadata limit")
    total = 160 + len(metadata) + payload_size
    require(total <= limits.file, "C1 file limit")
    header = bytearray(128)
    header[:16] = MAGIC
    struct.pack_into("<HHIIIQQQQQQIIIHHHH", header, 16,
                     1, 0, 128, 1, 1, 0, total, 128, len(metadata), 128 + len(metadata),
                     payload_size, len(state.tensors), len(state.aliases),
                     len(state.provider.encode("utf-8")), 1, 0, 2, 0)
    unsigned = bytes(header + metadata) + b"".join(tensor.data for tensor in state.tensors)
    return unsigned + digest(WHOLE_DOMAIN, unsigned)


def decode_c1(raw: bytes, limits: Limits = Limits()) -> State:
    require(160 <= len(raw) <= limits.file, "C1 file size limit")
    require(raw[:16] == MAGIC, "C1 magic")
    fields = struct.unpack_from("<HHIIIQQQQQQIIIHHHH", raw, 16)
    (major, minor, header, endian, checksum, features, total, metadata_at, metadata_len,
     payload_at, payload_len, count, groups, provider_len, sm, sn, pm, pn) = fields
    require((major, minor, sm, sn, pm, pn, features) == (1, 0, 1, 0, 2, 0, 0),
            "unsupported C1 version/features")
    require((header, endian, checksum) == (128, 1, 1) and not any(raw[100:128]),
            "C1 fixed header")
    require(total == len(raw) and metadata_at == 128 and metadata_len <= limits.metadata
            and payload_at == 128 + metadata_len
            and payload_at + payload_len == len(raw) - 32, "C1 section bounds")
    require(count <= limits.entries and groups <= 2048, "C1 count limit")
    require(1 <= provider_len <= 127 and 128 + provider_len <= payload_at, "C1 provider span")
    require(digest(WHOLE_DOMAIN, raw[:-32]) == raw[-32:], "C1 container checksum")
    try:
        provider = raw[128:128 + provider_len].decode("utf-8")
        tensors = []
        at, relative = 128 + provider_len, 0
        for _ in range(count):
            require(at + 80 <= payload_at, "C1 record prefix")
            (size, offset, length, path_len, segments, rank, kind, dtype, device, layout,
             reserved, product) = struct.unpack_from("<QQQIHHBBBBIQ", raw, at)
            require(1 <= segments <= 64 and rank <= 64, "C1 rank/path limit")
            require(size == 80 + rank * 8 + path_len and size <= payload_at - at,
                    "C1 record span")
            require(kind in (1, 2) and dtype in (1, 2, 3), "C1 kind/dtype")
            require(dtype == 3, "unsupported C1 dtype: only F32 interoperability")
            require((device, layout, reserved) == (1, 1, 0), "unsupported C1 device/layout")
            require(offset == relative and length <= limits.tensor
                    and length <= payload_len - relative, "C1 payload span")
            shape = struct.unpack_from("<" + "Q" * rank, raw, at + 80)
            require(product == elements(shape) and length == product * 4, "C1 shape size")
            cursor, end = at + 80 + 8 * rank, at + size
            path = []
            for _ in range(segments):
                require(cursor + 4 <= end, "C1 path prefix")
                part_len = struct.unpack_from("<I", raw, cursor)[0]
                cursor += 4
                require(1 <= part_len <= 65536 and part_len <= end - cursor, "C1 path span")
                path.append(raw[cursor:cursor + part_len].decode("utf-8"))
                cursor += part_len
            require(cursor == end, "C1 path end")
            data = raw[payload_at + relative:payload_at + relative + length]
            record = bytearray(raw[at:end])
            record[48:80] = bytes(32)
            require(digest(TENSOR_DOMAIN, record, data) == raw[at + 48:at + 80],
                    "C1 tensor checksum")
            tensors.append(Tensor(tuple(path), "parameter" if kind == 1 else "buffer", shape, data))
            at, relative = end, relative + length
        aliases = []
        members = 0
        for _ in range(groups):
            require(at + 8 <= payload_at, "C1 alias prefix")
            count, reserved = struct.unpack_from("<II", raw, at)
            members += count
            require(2 <= count and members <= 4096 and reserved == 0
                    and 8 + count * 4 <= payload_at - at, "C1 alias span")
            aliases.append(struct.unpack_from("<" + "I" * count, raw, at + 8))
            at += 8 + count * 4
        require(at == payload_at and relative == payload_len, "C1 exact section end")
    except UnicodeError as error:
        raise InteropError("C1 invalid UTF-8") from error
    state = State(provider, tuple(tensors), tuple(aliases))
    validate_state(state, limits)
    return state


def json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")


def parse_json(raw: bytes, maximum: int) -> object:
    require(len(raw) <= maximum, "JSON byte limit")
    # Limit recursive container depth before the standard decoder constructs objects.
    depth, quoted, escaped = 0, False, False
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            require(depth <= 16, "JSON nesting limit")
        elif byte in (93, 125):
            depth -= 1
            require(depth >= 0, "JSON nesting")

    def pairs(items: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def integer(value: str) -> int:
        require(not value.startswith("-") and len(value) <= 20, "JSON unsigned integer limit")
        return uint(int(value))

    def no_number(value: str) -> None:
        raise InteropError("JSON floating/nonfinite numbers unsupported")

    def strings(value: object) -> None:
        if type(value) is str:
            value.encode("utf-8")
        elif type(value) is list:
            for item in value:
                strings(item)
        elif type(value) is dict:
            for key, item in value.items():
                strings(key)
                strings(item)

    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                            parse_int=integer, parse_float=no_number, parse_constant=no_number)
        strings(result)
        return result
    except (UnicodeError, json.JSONDecodeError) as error:
        raise InteropError("invalid JSON") from error


def encode_safetensors(raw_c1: bytes, limits: Limits = Limits()) -> bytes:
    state = decode_c1(raw_c1, limits)
    manifest = {"provider": state.provider, "entries": [], "aliases": state.aliases}
    header = {}
    offset = 0
    for index, tensor in enumerate(state.tensors):
        name = f"t{index:06d}"
        manifest["entries"].append({"name": name, "path": tensor.path, "kind": tensor.kind})
        header[name] = {"dtype": "F32", "shape": tensor.shape,
                        "data_offsets": (offset, offset + len(tensor.data))}
        offset += len(tensor.data)
    header["__metadata__"] = {VERSION_KEY: "1.0", MANIFEST_KEY: json_bytes(manifest).decode("ascii"),
                              DIGEST_KEY: hashlib.sha256(raw_c1).hexdigest()}
    encoded = json_bytes(header)
    encoded += b" " * (-len(encoded) % 8)
    require(len(encoded) <= limits.metadata, "safetensors header limit")
    require(8 + len(encoded) + offset <= limits.file, "safetensors file limit")
    return struct.pack("<Q", len(encoded)) + encoded + b"".join(t.data for t in state.tensors)


def decode_safetensors(raw: bytes, limits: Limits = Limits(), *,
                       plain_provider: str | None = None) -> bytes:
    require(8 <= len(raw) <= limits.file, "safetensors file size limit")
    header_len = struct.unpack_from("<Q", raw)[0]
    require(2 <= header_len <= limits.metadata and header_len <= len(raw) - 8,
            "safetensors header span")
    require(raw[8:9] == b"{", "safetensors header must start with '{'")
    header = parse_json(raw[8:8 + header_len], limits.metadata)
    require(type(header) is dict, "safetensors header object")
    metadata = header.pop("__metadata__", {})
    require(type(metadata) is dict and all(type(v) is str for v in metadata.values()),
            "safetensors string metadata")
    require(len(header) <= limits.entries, "tensor count limit")
    start = 8 + header_len
    descriptors = {}
    spans = []
    for name, descriptor in header.items():
        text_bytes(name, 65536, "tensor name")
        require(type(descriptor) is dict and set(descriptor) == {"dtype", "shape", "data_offsets"},
                "tensor descriptor fields")
        require(descriptor["dtype"] == "F32", "unsupported safetensors dtype: only F32")
        shape, offsets = descriptor["shape"], descriptor["data_offsets"]
        require(type(shape) is list and type(offsets) is list and len(offsets) == 2,
                "tensor shape/offset arrays")
        shape = tuple(shape)
        size = elements(shape) * 4
        begin, end = map(uint, offsets)
        require(begin <= end and end <= len(raw) - start and end - begin == size
                and size <= limits.tensor, "tensor offset/shape span")
        descriptors[name] = (shape, begin, end)
        spans.append((begin, end))
    # Empty tensors at shared boundaries are valid; holes, overlap and hidden bytes are not.
    cursor = 0
    for begin, end in sorted(spans):
        require(begin == cursor, "noncontiguous tensor offsets")
        cursor = end
    require(cursor == len(raw) - start, "unindexed safetensors data")
    reserved = {key for key in metadata if key.startswith("eshkol.io5.")}
    if reserved:
        require(plain_provider is None, "plain import cannot override IO5 metadata")
        require(reserved == RESERVED and metadata[VERSION_KEY] == "1.0", "unsupported IO5 metadata version/fields")
        manifest = parse_json(text_bytes(metadata[MANIFEST_KEY], limits.metadata, "IO5 manifest"),
                              limits.metadata)
        require(type(manifest) is dict and set(manifest) == {"provider", "entries", "aliases"},
                "IO5 manifest fields")
        entries, groups = manifest["entries"], manifest["aliases"]
        require(type(entries) is list and len(entries) == len(header) and type(groups) is list,
                "IO5 manifest counts")
        tensors, names = [], set()
        for entry in entries:
            require(type(entry) is dict and set(entry) == {"name", "path", "kind"}, "IO5 entry fields")
            name, path = entry["name"], entry["path"]
            require(type(name) is str and name in descriptors and name not in names, "IO5 tensor name")
            require(type(path) is list, "IO5 path array")
            names.add(name)
            shape, begin, end = descriptors[name]
            tensors.append(Tensor(tuple(path), entry["kind"], shape, raw[start + begin:start + end]))
        require(all(type(group) is list for group in groups), "IO5 alias arrays")
        state = State(manifest["provider"], tuple(tensors), tuple(tuple(g) for g in groups))
        result = encode_c1(state, limits)
        require(metadata[DIGEST_KEY] == hashlib.sha256(result).hexdigest(), "IO5 C1 checksum")
        return result
    require(plain_provider is not None, "plain safetensors requires explicit --plain-provider")
    text_bytes(plain_provider, 127, "provider")
    tensors = []
    for name in sorted(descriptors, key=lambda value: value.encode("utf-8")):
        shape, begin, end = descriptors[name]
        tensors.append(Tensor((name,), "parameter", shape, raw[start + begin:start + end]))
    return encode_c1(State(plain_provider, tuple(tensors)), limits)


def read_file(path: Path, limits: Limits) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        info = os.fstat(source.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_size <= limits.file, "input regular-file/size policy")
        raw = source.read(limits.file + 1)
        require(len(raw) <= limits.file and len(raw) == info.st_size, "input changed size/size limit")
        return raw


def publish(path: Path, raw: bytes, overwrite: bool) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".io5-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            os.unlink(temporary)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=("c1-to-safetensors", "safetensors-to-c1"))
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
        if arguments.direction == "c1-to-safetensors":
            require(arguments.plain_provider is None, "--plain-provider is import-only")
            result = encode_safetensors(raw, limits)
        else:
            result = decode_safetensors(raw, limits, plain_provider=arguments.plain_provider)
        publish(arguments.destination, result, arguments.overwrite)
        return 0
    except (InteropError, OSError) as error:
        print(f"IO5 error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
