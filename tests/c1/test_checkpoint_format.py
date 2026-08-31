#!/usr/bin/env python3
"""Independent adversarial validator for the C1 checkpoint-v1 byte contract."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile


MAGIC = bytes.fromhex("894553484b4f4c434b50540d0a1a0a00")
CONTAINER_DOMAIN = b"eshkol-checkpoint-container-v1\0"
TENSOR_DOMAIN = b"eshkol-checkpoint-tensor-v1\0"
HEADER_BYTES = 128
CHECKSUM_BYTES = 32
RECORD_PREFIX_BYTES = 80
MAX_ENTRIES = 4096
MAX_ALIASES = 4096
MAX_ALIAS_MEMBERS = 4096
MAX_SEGMENTS = 64
MAX_SEGMENT_BYTES = 65536
MAX_RANK = 64
MAX_PROVIDER_BYTES = 127
U64_MAX = (1 << 64) - 1


class FormatError(AssertionError):
    pass


def u16(data: bytes | bytearray, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def u32(data: bytes | bytearray, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def u64(data: bytes | bytearray, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


def put16(data: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<H", data, offset, value)


def put32(data: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", data, offset, value)


def put64(data: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<Q", data, offset, value)


@dataclasses.dataclass(frozen=True)
class Record:
    index: int
    start: int
    end: int
    relative: int
    payload_bytes: int
    path_bytes: int
    segments: int
    rank: int
    kind: int
    dtype: int
    device: int
    layout: int
    elements: int
    extent_start: int
    path_start: int
    segment_spans: tuple[tuple[int, int, int], ...]
    path: tuple[bytes, ...]
    payload_start: int


@dataclasses.dataclass(frozen=True)
class Alias:
    start: int
    count: int
    members_start: int
    members: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class Container:
    raw: bytes
    metadata_offset: int
    metadata_bytes: int
    payload_offset: int
    payload_bytes: int
    provider_start: int
    provider_end: int
    records: tuple[Record, ...]
    aliases: tuple[Alias, ...]


def checked_product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        if value and result > U64_MAX // value:
            raise FormatError("shape product overflow")
        result *= value
    return result


def parse_golden(raw: bytes, expected_provider: bytes | None = b"fixture-v1") -> Container:
    """Parse and independently validate every canonical v1 structural rule."""
    if len(raw) < HEADER_BYTES + CHECKSUM_BYTES:
        raise FormatError("short file")
    if raw[:16] != MAGIC:
        raise FormatError("magic")
    if (u16(raw, 16), u16(raw, 18)) != (1, 0):
        raise FormatError("container version")
    if u32(raw, 20) != HEADER_BYTES or u32(raw, 24) != 1 or u32(raw, 28) != 1:
        raise FormatError("fixed header")
    if u64(raw, 32) != 0 or any(raw[100:128]):
        raise FormatError("features or reserved")
    if (u16(raw, 92), u16(raw, 94), u16(raw, 96), u16(raw, 98)) != (1, 0, 1, 0):
        raise FormatError("P1 versions")

    file_bytes = u64(raw, 40)
    metadata_offset = u64(raw, 48)
    metadata_bytes = u64(raw, 56)
    payload_offset = u64(raw, 64)
    payload_bytes = u64(raw, 72)
    entry_count = u32(raw, 80)
    alias_count = u32(raw, 84)
    provider_bytes = u32(raw, 88)
    unsigned_size = len(raw) - CHECKSUM_BYTES
    if file_bytes != len(raw):
        raise FormatError("file size")
    if metadata_offset != HEADER_BYTES:
        raise FormatError("metadata offset")
    if payload_offset != metadata_offset + metadata_bytes:
        raise FormatError("metadata size")
    if unsigned_size != payload_offset + payload_bytes:
        raise FormatError("payload size")
    if entry_count > MAX_ENTRIES or alias_count > MAX_ALIASES:
        raise FormatError("counts")
    if not (1 <= provider_bytes <= MAX_PROVIDER_BYTES) or metadata_offset + provider_bytes > payload_offset:
        raise FormatError("provider size")
    expected_whole = hashlib.sha256(CONTAINER_DOMAIN + raw[:unsigned_size]).digest()
    if raw[unsigned_size:] != expected_whole:
        raise FormatError("whole checksum")

    provider_start = metadata_offset
    provider_end = provider_start + provider_bytes
    provider = raw[provider_start:provider_end]
    if provider.decode("utf-8").encode("utf-8") != provider:
        raise FormatError("provider UTF-8")
    if expected_provider is not None and provider != expected_provider:
        raise FormatError("unexpected golden provider")

    records: list[Record] = []
    offset = provider_end
    expected_relative = 0
    previous_path: tuple[bytes, ...] | None = None
    widths = {1: 1, 2: 8, 3: 4}
    for index in range(entry_count):
        if offset + RECORD_PREFIX_BYTES > payload_offset:
            raise FormatError("record prefix")
        record_bytes = u64(raw, offset)
        relative = u64(raw, offset + 8)
        tensor_bytes = u64(raw, offset + 16)
        path_bytes = u32(raw, offset + 24)
        segments = u16(raw, offset + 28)
        rank = u16(raw, offset + 30)
        kind, dtype, device, layout = raw[offset + 32 : offset + 36]
        elements = u64(raw, offset + 40)
        end = offset + record_bytes
        if not (1 <= segments <= MAX_SEGMENTS) or rank > MAX_RANK:
            raise FormatError("rank/segments")
        if record_bytes != RECORD_PREFIX_BYTES + rank * 8 + path_bytes or end > payload_offset:
            raise FormatError("record size")
        if any(raw[offset + 36 : offset + 40]):
            raise FormatError("record reserved")
        if kind not in (1, 2) or dtype not in widths or (kind == 1 and dtype != 3):
            raise FormatError("kind/dtype")
        if device != 1 or layout != 1 or relative != expected_relative:
            raise FormatError("device/layout/relative")
        extents = tuple(u64(raw, offset + RECORD_PREFIX_BYTES + 8 * i) for i in range(rank))
        product = checked_product(extents)
        if elements != product or tensor_bytes != product * widths[dtype]:
            raise FormatError("shape/payload product")

        path_start = offset + RECORD_PREFIX_BYTES + rank * 8
        path_end = path_start + path_bytes
        at = path_start
        path: list[bytes] = []
        spans: list[tuple[int, int, int]] = []
        for _ in range(segments):
            if at + 4 > path_end:
                raise FormatError("path length")
            length = u32(raw, at)
            data_start = at + 4
            data_end = data_start + length
            if not (1 <= length <= MAX_SEGMENT_BYTES) or data_end > path_end:
                raise FormatError("segment bounds")
            segment = raw[data_start:data_end]
            if segment.decode("utf-8").encode("utf-8") != segment:
                raise FormatError("segment UTF-8")
            path.append(segment)
            spans.append((at, data_start, data_end))
            at = data_end
        path_tuple = tuple(path)
        if at != path_end or (previous_path is not None and not previous_path < path_tuple):
            raise FormatError("path ordering")

        payload_start = payload_offset + relative
        payload_end = payload_start + tensor_bytes
        if payload_end > unsigned_size:
            raise FormatError("payload span")
        record_zero = bytearray(raw[offset:end])
        record_zero[48:80] = bytes(32)
        payload = raw[payload_start:payload_end]
        digest = hashlib.sha256(TENSOR_DOMAIN + record_zero + payload).digest()
        if raw[offset + 48 : offset + 80] != digest:
            raise FormatError("tensor checksum")
        if dtype == 1 and any(value not in (0, 1) for value in payload):
            raise FormatError("bool payload")
        records.append(
            Record(index, offset, end, relative, tensor_bytes, path_bytes, segments,
                   rank, kind, dtype, device, layout, elements,
                   offset + RECORD_PREFIX_BYTES, path_start, tuple(spans),
                   path_tuple, payload_start)
        )
        previous_path = path_tuple
        expected_relative += tensor_bytes
        offset = end

    aliases: list[Alias] = []
    seen: set[int] = set()
    total_members = 0
    previous_first = -1
    for _ in range(alias_count):
        if offset + 8 > payload_offset:
            raise FormatError("alias prefix")
        count = u32(raw, offset)
        if count < 2 or u32(raw, offset + 4) != 0:
            raise FormatError("alias header")
        members_start = offset + 8
        end = members_start + count * 4
        if end > payload_offset:
            raise FormatError("alias bounds")
        members = tuple(u32(raw, members_start + 4 * i) for i in range(count))
        total_members += count
        if total_members > MAX_ALIAS_MEMBERS:
            raise FormatError("alias members")
        if any(index >= entry_count for index in members):
            raise FormatError("alias index")
        if tuple(sorted(set(members))) != members or members[0] <= previous_first:
            raise FormatError("alias ordering")
        if any(index in seen or records[index].kind != 1 for index in members):
            raise FormatError("alias membership")
        first_payload = raw[records[members[0]].payload_start:
                            records[members[0]].payload_start + records[members[0]].payload_bytes]
        if any(raw[records[index].payload_start:
                   records[index].payload_start + records[index].payload_bytes] != first_payload
               for index in members[1:]):
            raise FormatError("alias payload conflict")
        seen.update(members)
        aliases.append(Alias(offset, count, members_start, members))
        previous_first = members[0]
        offset = end

    if offset != payload_offset or expected_relative != payload_bytes:
        raise FormatError("metadata or payload end")
    return Container(raw, metadata_offset, metadata_bytes, payload_offset, payload_bytes,
                     provider_start, provider_end, tuple(records), tuple(aliases))


def repair_whole(data: bytearray) -> None:
    unsigned_size = len(data) - CHECKSUM_BYTES
    data[unsigned_size:] = hashlib.sha256(CONTAINER_DOMAIN + data[:unsigned_size]).digest()


def repair_tensor(data: bytearray, record: Record) -> None:
    record_zero = bytearray(data[record.start:record.end])
    record_zero[48:80] = bytes(32)
    payload = data[record.payload_start:record.payload_start + record.payload_bytes]
    data[record.start + 48:record.start + 80] = hashlib.sha256(
        TENSOR_DOMAIN + record_zero + payload
    ).digest()


def build_container(
    entries: tuple[tuple[tuple[bytes, ...], int, int, tuple[int, ...], bytes], ...],
    aliases: tuple[tuple[int, ...], ...] = (),
    provider: bytes = b"fixture-v1",
) -> bytes:
    """Build small, fully checksummed canonical or deliberately invalid fixtures."""
    records: list[bytes] = []
    payloads: list[bytes] = []
    relative = 0
    widths = {1: 1, 2: 8, 3: 4}
    for path, kind, dtype, extents, payload in entries:
        encoded_path = b"".join(struct.pack("<I", len(segment)) + segment
                                for segment in path)
        record = bytearray(RECORD_PREFIX_BYTES + 8 * len(extents) + len(encoded_path))
        put64(record, 0, len(record))
        put64(record, 8, relative)
        put64(record, 16, len(payload))
        put32(record, 24, len(encoded_path))
        put16(record, 28, len(path))
        put16(record, 30, len(extents))
        record[32:36] = bytes((kind, dtype, 1, 1))
        elements = 1
        for extent in extents:
            elements *= extent
        put64(record, 40, elements)
        for index, extent in enumerate(extents):
            put64(record, RECORD_PREFIX_BYTES + 8 * index, extent)
        path_start = RECORD_PREFIX_BYTES + 8 * len(extents)
        record[path_start:] = encoded_path
        record[48:80] = hashlib.sha256(TENSOR_DOMAIN + record + payload).digest()
        records.append(bytes(record))
        payloads.append(payload)
        relative += len(payload)
        if dtype in widths and len(payload) != elements * widths[dtype]:
            raise AssertionError("boundary fixture payload does not match its shape")

    alias_bytes = b"".join(
        struct.pack("<II", len(group), 0)
        + b"".join(struct.pack("<I", member) for member in group)
        for group in aliases
    )
    metadata = provider + b"".join(records) + alias_bytes
    payload = b"".join(payloads)
    header = bytearray(HEADER_BYTES)
    header[:16] = MAGIC
    put16(header, 16, 1)
    put32(header, 20, HEADER_BYTES)
    put32(header, 24, 1)
    put32(header, 28, 1)
    put64(header, 40, HEADER_BYTES + len(metadata) + len(payload) + CHECKSUM_BYTES)
    put64(header, 48, HEADER_BYTES)
    put64(header, 56, len(metadata))
    put64(header, 64, HEADER_BYTES + len(metadata))
    put64(header, 72, len(payload))
    put32(header, 80, len(entries))
    put32(header, 84, len(aliases))
    put32(header, 88, len(provider))
    put16(header, 92, 1)
    put16(header, 96, 1)
    unsigned = bytes(header) + metadata + payload
    return unsigned + hashlib.sha256(CONTAINER_DOMAIN + unsigned).digest()


class Runner:
    def __init__(self, validator: Path, timeout: float, temporary: Path) -> None:
        self.validator = validator
        self.timeout = timeout
        self.path = temporary / "mutation.etcp"
        self.checks = 0
        self.failures: list[str] = []

    def run(self, label: str, data: bytes | bytearray, category: str | None) -> None:
        self.path.write_bytes(data)
        try:
            result = subprocess.run(
                [str(self.validator), str(self.path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout,
                check=False,
                env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
            )
        except subprocess.TimeoutExpired as error:
            self.failures.append(f"{label}: validator timed out: {error}")
            return
        expected_code = 0 if category is None else 2
        expected_stdout = "VALID\n" if category is None else f"ERROR category={category}\n"
        if result.returncode != expected_code or result.stdout != expected_stdout or result.stderr:
            self.failures.append(
                f"{label}: got rc={result.returncode}, stdout={result.stdout!r}, "
                f"stderr={result.stderr!r}; expected rc={expected_code}, "
                f"stdout={expected_stdout!r}"
            )
        self.checks += 1


def mutate_header_byte(raw: bytes, offset: int) -> bytearray:
    data = bytearray(raw)
    data[offset] ^= 0x01
    return data


def early_header_category(offset: int) -> str:
    if 16 <= offset < 20 or 32 <= offset < 40 or 92 <= offset < 100:
        return "version-mismatch"
    return "corrupt-data"


def mutated_category(data: bytes | bytearray) -> str:
    """Return the category implied by the parser's documented validation order."""
    if bytes(data[:16]) != MAGIC:
        return "corrupt-data"
    if (u16(data, 16), u16(data, 18)) != (1, 0):
        return "version-mismatch"
    if u64(data, 32) != 0:
        return "version-mismatch"
    if (u32(data, 20), u32(data, 24), u32(data, 28)) != (128, 1, 1):
        return "corrupt-data"
    if any(data[100:128]):
        return "corrupt-data"
    if (u16(data, 92), u16(data, 94), u16(data, 96), u16(data, 98)) != (1, 0, 1, 0):
        return "version-mismatch"
    return "corrupt-data"


def deep_mutations(runner: Runner, parsed: Container) -> None:
    raw = parsed.raw
    first = parsed.records[0]
    boolean = next(record for record in parsed.records if record.dtype == 1)
    buffer_record = next(record for record in parsed.records if record.kind == 2)
    alias = parsed.aliases[0]

    def case(label: str, data: bytearray, *, tensors: tuple[Record, ...] = (),
             category: str = "corrupt-data") -> None:
        for record in tensors:
            repair_tensor(data, record)
        repair_whole(data)
        runner.run(label, data, category)

    data = bytearray(raw); data[first.start + 48] ^= 1
    repair_whole(data); runner.run("per-tensor-checksum", data, "corrupt-data")
    data = bytearray(raw); data[-1] ^= 1
    runner.run("whole-checksum", data, "corrupt-data")
    data = bytearray(raw); data[first.payload_start] ^= 1
    repair_whole(data); runner.run("payload-with-stale-tensor-checksum", data, "corrupt-data")

    data = bytearray(raw); data[parsed.provider_start:parsed.provider_end] = b"unknown-v1"
    case("trusted-provider-mismatch", data, category="unsupported")
    data = bytearray(raw); data[parsed.provider_start] = 0xFF
    case("provider-invalid-utf8", data)
    for label, prefix in (
        ("overlong", b"\xc0\x80"),
        ("surrogate", b"\xed\xa0\x80"),
        ("above-unicode", b"\xf4\x90\x80\x80"),
        ("truncated-continuation", b"\xe2\x82"),
    ):
        data = bytearray(raw)
        provider_length = parsed.provider_end - parsed.provider_start
        data[parsed.provider_start:parsed.provider_end] = (
            prefix + b"x" * (provider_length - len(prefix))
        )
        case(f"provider-utf8-{label}", data)

    # Same-size path edits reach UTF-8, duplicate, and global ordering checks.
    _, content_start, _ = first.segment_spans[0]
    data = bytearray(raw); data[content_start] = 0xFF
    case("path-invalid-utf8", data, tensors=(first,))
    segment_length = first.segment_spans[0][2] - content_start
    for label, prefix in (
        ("overlong", b"\xc0\x80"),
        ("surrogate", b"\xed\xa0\x80"),
        ("above-unicode", b"\xf4\x90\x80\x80"),
        ("truncated-continuation", b"\xe2\x82"),
    ):
        data = bytearray(raw)
        data[content_start:content_start + segment_length] = (
            prefix + b"x" * (segment_length - len(prefix))
        )
        case(f"path-utf8-{label}", data, tensors=(first,))
    equal_path_pair = next(
        (left, right) for left in parsed.records for right in parsed.records
        if left.index < right.index and left.path_bytes == right.path_bytes
    )
    left, right = equal_path_pair
    data = bytearray(raw)
    data[left.path_start:left.path_start + left.path_bytes] = raw[
        right.path_start:right.path_start + right.path_bytes]
    case("duplicate-path", data, tensors=(left,))
    data = bytearray(raw)
    left_path = bytes(data[left.path_start:left.path_start + left.path_bytes])
    right_path = bytes(data[right.path_start:right.path_start + right.path_bytes])
    data[left.path_start:left.path_start + left.path_bytes] = right_path
    data[right.path_start:right.path_start + right.path_bytes] = left_path
    case("noncanonical-path-order", data, tensors=(left, right))
    data = bytearray(raw); put32(data, first.segment_spans[0][0], 0)
    case("zero-path-segment", data)
    data = bytearray(raw); put32(data, first.segment_spans[0][0], MAX_SEGMENT_BYTES + 1)
    case("oversize-path-segment", data)

    record_changes: tuple[tuple[str, int, int, int], ...] = (
        ("record-size", first.start, 8, 0),
        ("payload-relative", first.start + 8, 8, first.relative + 1),
        ("payload-length", first.start + 16, 8, first.payload_bytes + 1),
        ("path-byte-length", first.start + 24, 4, first.path_bytes + 1),
        ("segment-count-zero", first.start + 28, 2, 0),
        ("rank-over-limit", first.start + 30, 2, MAX_RANK + 1),
        ("element-count", first.start + 40, 8, first.elements + 1),
    )
    for label, offset, width, value in record_changes:
        data = bytearray(raw)
        {2: put16, 4: put32, 8: put64}[width](data, offset, value)
        case(label, data)
    for label, relative, value in (
        ("kind", 32, 0), ("dtype", 33, 2), ("device", 34, 0), ("layout", 35, 0),
        ("record-reserved", 36, 1),
    ):
        data = bytearray(raw); data[first.start + relative] = value
        case(label, data)

    data = bytearray(raw); put64(data, first.extent_start, first.elements + 1)
    case("shape-extent-product", data)
    # Increase rank without increasing record size, consuming eight path bytes as
    # a second U64 extent. The parser reaches checked-product overflow first.
    if first.path_bytes >= 9 and first.elements >= 2:
        data = bytearray(raw)
        put16(data, first.start + 30, first.rank + 1)
        put32(data, first.start + 24, first.path_bytes - 8)
        put64(data, first.path_start, U64_MAX)
        case("shape-product-overflow", data)

    data = bytearray(raw); data[boolean.payload_start] = 2
    case("bool-noncanonical-byte", data, tensors=(boolean,))
    data = bytearray(raw); data[buffer_record.start + 33] = 0
    case("buffer-invalid-dtype", data)

    data = bytearray(raw); put32(data, alias.start + 4, 1)
    case("alias-reserved", data)
    data = bytearray(raw); put32(data, alias.start, 1)
    case("alias-too-small", data)
    data = bytearray(raw)
    put32(data, alias.members_start, alias.members[1]); put32(data, alias.members_start + 4, alias.members[0])
    case("alias-unsorted", data)
    data = bytearray(raw); put32(data, alias.members_start + 4, alias.members[0])
    case("alias-duplicate", data)
    data = bytearray(raw); put32(data, alias.members_start + 4, len(parsed.records))
    case("alias-out-of-range", data)
    data = bytearray(raw); put32(data, alias.members_start + 4, buffer_record.index)
    case("alias-buffer-member", data)
    tied = parsed.records[alias.members[1]]
    data = bytearray(raw); data[tied.payload_start] ^= 1
    case("alias-conflicting-payload", data, tensors=(tied,))

    # Policy and aggregate bounds are rejected before allocation.
    for label, offset, width, value in (
        ("policy-file-bytes", 40, 8, 1048577),
        ("policy-metadata-bytes", 56, 8, 524289),
        ("policy-entry-count", 80, 4, 65),
        ("provider-length-zero", 88, 4, 0),
        ("provider-length", 88, 4, MAX_PROVIDER_BYTES + 1),
    ):
        data = bytearray(raw)
        {4: put32, 8: put64}[width](data, offset, value)
        case(label, data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("validator", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--fuzz-cases", type=int, default=192)
    arguments = parser.parse_args()
    if arguments.timeout <= 0 or arguments.fuzz_cases < 1 or arguments.fuzz_cases > 4096:
        parser.error("timeout must be positive and fuzz-cases must be in 1..4096")
    if not arguments.validator.is_file() or not os.access(arguments.validator, os.X_OK):
        parser.error("validator must be an executable file")

    raw = arguments.checkpoint.read_bytes()
    parsed = parse_golden(raw)
    with tempfile.TemporaryDirectory(prefix="c1-format-adversarial-") as directory:
        runner = Runner(arguments.validator.resolve(), arguments.timeout, Path(directory))
        runner.run("golden", raw, None)

        # Every fixed-header byte is independently perturbed. Version/feature
        # domains have their precise category; all other bytes are corruption.
        for offset in range(HEADER_BYTES):
            runner.run(f"header-byte-{offset}", mutate_header_byte(raw, offset),
                       early_header_category(offset))

        # Explicit field-boundary probes supplement byte perturbation.
        header_fields = (
            ("major-zero", 16, 2, 0, "version-mismatch"),
            ("minor-one", 18, 2, 1, "version-mismatch"),
            ("header-size-zero", 20, 4, 0, "corrupt-data"),
            ("endianness-zero", 24, 4, 0, "corrupt-data"),
            ("checksum-zero", 28, 4, 0, "corrupt-data"),
            ("feature-max", 32, 8, U64_MAX, "version-mismatch"),
            ("file-size-max", 40, 8, U64_MAX, "corrupt-data"),
            ("metadata-offset-zero", 48, 8, 0, "corrupt-data"),
            ("metadata-size-max", 56, 8, U64_MAX, "corrupt-data"),
            ("payload-offset-max", 64, 8, U64_MAX, "corrupt-data"),
            ("payload-size-max", 72, 8, U64_MAX, "corrupt-data"),
            ("entry-count-max", 80, 4, (1 << 32) - 1, "corrupt-data"),
            ("alias-count-max", 84, 4, (1 << 32) - 1, "corrupt-data"),
            ("provider-size-max", 88, 4, (1 << 32) - 1, "corrupt-data"),
            ("state-major-zero", 92, 2, 0, "version-mismatch"),
            ("state-minor-one", 94, 2, 1, "version-mismatch"),
            ("provider-major-zero", 96, 2, 0, "version-mismatch"),
            ("provider-minor-one", 98, 2, 1, "version-mismatch"),
        )
        for label, offset, width, value, category in header_fields:
            data = bytearray(raw)
            {2: put16, 4: put32, 8: put64}[width](data, offset, value)
            runner.run(label, data, category)

        for length in range(len(raw)):
            runner.run(f"truncate-{length}", raw[:length], "corrupt-data")
        for suffix in (b"\0", b"\xff", b"TRAILING"):
            runner.run(f"trailing-{suffix.hex()}", raw + suffix, "corrupt-data")

        deep_mutations(runner, parsed)

        # Deterministic exact/one-over v1 boundaries use independently built,
        # fully checksummed containers rather than stale-checksum header edits.
        scalar = ((b"scalar",), 2, 3, (), bytes(4))
        zero = ((b"zero",), 2, 3, (0,), b"")
        runner.run("boundary-empty-state", build_container(()), None)
        runner.run("boundary-rank-zero", build_container((scalar,)), None)
        runner.run("boundary-zero-extent-payload", build_container((zero,)), None)
        runner.run("boundary-provider-127",
                   build_container((scalar,), provider=b"p" * MAX_PROVIDER_BYTES),
                   "unsupported")
        runner.run("boundary-provider-128",
                   build_container((scalar,), provider=b"p" * (MAX_PROVIDER_BYTES + 1)),
                   "corrupt-data")
        runner.run("boundary-rank-64",
                   build_container((((b"rank",), 2, 3, (1,) * MAX_RANK, bytes(4)),)),
                   None)
        runner.run("boundary-rank-65",
                   build_container((((b"rank",), 2, 3, (1,) * (MAX_RANK + 1), bytes(4)),)),
                   "corrupt-data")
        runner.run("boundary-path-depth-64",
                   build_container(((tuple(b"x" for _ in range(MAX_SEGMENTS)),
                                     2, 3, (1,), bytes(4)),)), None)
        runner.run("boundary-path-depth-65",
                   build_container(((tuple(b"x" for _ in range(MAX_SEGMENTS + 1)),
                                     2, 3, (1,), bytes(4)),)), "corrupt-data")
        runner.run("boundary-segment-65536",
                   build_container((((b"x" * MAX_SEGMENT_BYTES,),
                                     2, 3, (1,), bytes(4)),)), None)
        runner.run("boundary-segment-65537",
                   build_container((((b"x" * (MAX_SEGMENT_BYTES + 1),),
                                     2, 3, (1,), bytes(4)),)), "corrupt-data")

        lower_count_entries = tuple(
            ((f"e{index:02d}".encode(),), 2, 3, (0,), b"")
            for index in range(65)
        )
        runner.run("boundary-lowered-entry-count-64",
                   build_container(lower_count_entries[:64]), None)
        runner.run("boundary-lowered-entry-count-65",
                   build_container(lower_count_entries), "corrupt-data")

        tied_entries = tuple(
            ((f"t{index}".encode(),), 1, 3, (0,), b"") for index in range(4)
        )
        runner.run("boundary-two-disjoint-alias-groups",
                   build_container(tied_entries, ((0, 1), (2, 3))), None)
        runner.run("boundary-noncanonical-alias-group-order",
                   build_container(tied_entries, ((2, 3), (0, 1))), "corrupt-data")
        runner.run("boundary-overlapping-alias-groups",
                   build_container(tied_entries, ((0, 1), (1, 2))), "corrupt-data")

        def parser_case(label: str, data: bytes, valid: bool) -> None:
            try:
                parse_golden(data, expected_provider=None)
                accepted = True
            except (FormatError, UnicodeError):
                accepted = False
            if accepted != valid:
                runner.failures.append(
                    f"{label}: independent parser acceptance={accepted}, expected={valid}"
                )
            runner.checks += 1

        hard_entries = tuple(
            ((f"h{index:04d}".encode(),), 1, 3, (0,), b"")
            for index in range(MAX_ENTRIES + 1)
        )
        canonical_pairs = tuple((index, index + 1)
                                for index in range(0, MAX_ALIAS_MEMBERS, 2))
        parser_case(
            "boundary-hard-4096-entries-2048-effective-groups-4096-members",
            build_container(hard_entries[:MAX_ENTRIES], canonical_pairs), True,
        )
        parser_case("boundary-hard-entry-count-4097",
                    build_container(hard_entries), False)
        parser_case("boundary-hard-alias-header-count-4097",
                    build_container(tied_entries, ((0, 1),) * (MAX_ALIASES + 1)), False)
        parser_case("boundary-hard-alias-members-4097",
                    build_container(hard_entries[:MAX_ENTRIES],
                                    (tuple(range(MAX_ALIAS_MEMBERS))
                                     + (MAX_ALIAS_MEMBERS - 1,),)), False)

        randomizer = random.Random(0xC1C0FFEE)
        for index in range(arguments.fuzz_cases):
            data = bytearray(raw)
            flips = randomizer.randint(1, 4)
            offsets: list[int] = []
            for _ in range(flips):
                offset = randomizer.randrange(len(data))
                data[offset] ^= 1 << randomizer.randrange(8)
                offsets.append(offset)
            runner.run(f"seeded-fuzz-{index}", data, mutated_category(data))

        if runner.failures:
            for failure in runner.failures[:32]:
                print(f"C1 format adversarial: FAIL: {failure}", file=sys.stderr)
            if len(runner.failures) > 32:
                print(f"C1 format adversarial: FAIL: ... {len(runner.failures) - 32} more",
                      file=sys.stderr)
            print(f"C1 format adversarial: FAIL ({len(runner.failures)} of "
                  f"{runner.checks} format checks)", file=sys.stderr)
            return 1
        print(f"C1 format adversarial: PASS ({runner.checks} format checks)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
