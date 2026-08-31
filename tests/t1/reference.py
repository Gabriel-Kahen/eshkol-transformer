"""Independent, development-only reference for the T1 v1 tokenizer format.

This module is intentionally Python-only test/oracle code.  Production tokenization,
format handling, checksums, and persistence remain Eshkol-authored.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import re
from typing import NoReturn

MAX_FILE = 1_048_576
MAX_SPECIALS = 4_096
MAX_INSERTIONS = 4_096
MAX_HEADER_ITEMS = 4_096
MAX_LINE = 256
MAX_I64 = (1 << 63) - 1
NAME_RE = re.compile(rb"[a-z][a-z0-9._-]{0,63}\Z")
DIGEST_RE = re.compile(rb"[0-9a-f]{64}\Z")
HEX_RE = re.compile(rb"(?:[0-9a-f]{2}){1,64}\Z")
CHECKSUM_DOMAIN = b"eshkol-byte-tokenizer-checksum-v1\n"
IDENTITY_DOMAIN = b"sha256:eshkol-byte-tokenizer-v1\n"
V1_HEADER = (
    b"format\teshkol-byte-tokenizer\n"
    b"version\t1\t0\n"
    b"required-features\t0\n"
    b"optional-fields\t0\n"
    b"limit-file-bytes\t1048576\n"
    b"limit-specials\t4096\n"
    b"limit-name-bytes\t64\n"
    b"limit-header-items\t4096\n"
    b"limit-optional-value-bytes\t64\n"
    b"limit-prefix\t4096\n"
    b"limit-suffix\t4096\n"
)


class FormatError(ValueError):
    def __init__(self, category: str, message: str) -> None:
        self.category = category
        super().__init__(f"{category}: {message}")


def fail(message: str, category: str = "corrupt-data") -> NoReturn:
    raise FormatError(category, message)


@dataclass(frozen=True, slots=True)
class Special:
    token_id: int
    name: str
    decode: str


@dataclass(frozen=True, slots=True)
class Identity:
    utf8_policy: str = "raw"
    specials: tuple[Special, ...] = ()
    prefix: tuple[int, ...] = ()
    suffix: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Parsed:
    identity: Identity
    fingerprint: str
    minor: int
    optional_fields: tuple[tuple[str, bytes], ...]


def _u64(raw: bytes, where: str, maximum: int = MAX_I64) -> int:
    if not raw or not raw.isdigit() or (len(raw) > 1 and raw.startswith(b"0")):
        fail(f"{where}: noncanonical unsigned decimal")
    value = int(raw)
    if value > maximum:
        fail(f"{where}: overflow")
    return value


def _name(value: str, where: str) -> bytes:
    if not isinstance(value, str):
        fail(f"{where}: expected string")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        fail(f"{where}: non-ASCII name")
    if not NAME_RE.fullmatch(encoded):
        fail(f"{where}: invalid name")
    return encoded


def validate(identity: Identity) -> Identity:
    if not isinstance(identity, Identity):
        fail("identity: wrong type")
    if identity.utf8_policy not in {"raw", "strict"}:
        fail("identity: invalid UTF-8 policy")
    if len(identity.specials) > MAX_SPECIALS:
        fail("identity: too many specials")
    if len(identity.prefix) > MAX_INSERTIONS or len(identity.suffix) > MAX_INSERTIONS:
        fail("identity: too many insertions")
    specials = sorted(identity.specials, key=lambda item: item.token_id)
    names: set[str] = set()
    for index, special in enumerate(specials):
        if not isinstance(special, Special):
            fail("identity: malformed special")
        if isinstance(special.token_id, bool) or special.token_id != 256 + index:
            fail("identity: special IDs must be contiguous from 256")
        _name(special.name, "identity.special.name")
        if special.name in names:
            fail("identity: duplicate special name")
        names.add(special.name)
        if special.decode not in {"omit", "error"}:
            fail("identity: invalid decode policy")
    by_id = {item.token_id: item for item in specials}
    for label, values in (("prefix", identity.prefix), ("suffix", identity.suffix)):
        for token_id in values:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                fail(f"identity.{label}: invalid ID type")
            if token_id not in by_id:
                fail(f"identity.{label}: unknown special ID")
            if by_id[token_id].decode != "omit":
                fail(f"identity.{label}: inserted special must decode as omit")
    return Identity(identity.utf8_policy, tuple(specials), identity.prefix, identity.suffix)


def payload(identity: Identity) -> bytes:
    identity = validate(identity)
    lines = [
        b"kind\tbyte\n",
        b"normalization\tnone\n",
        f"utf8-policy\t{identity.utf8_policy}\n".encode(),
        b"byte-ids\t0\t255\n",
        f"special-count\t{len(identity.specials)}\n".encode(),
    ]
    for special in identity.specials:
        lines.append(
            b"special\t"
            + str(special.token_id).encode()
            + b"\t"
            + _name(special.name, "special.name")
            + b"\t"
            + special.decode.encode()
            + b"\n"
        )
    lines.append(f"prefix-count\t{len(identity.prefix)}\n".encode())
    lines.extend(
        f"prefix\t{index}\t{token_id}\n".encode()
        for index, token_id in enumerate(identity.prefix)
    )
    lines.append(f"suffix-count\t{len(identity.suffix)}\n".encode())
    lines.extend(
        f"suffix\t{index}\t{token_id}\n".encode()
        for index, token_id in enumerate(identity.suffix)
    )
    return b"".join(lines)


def _header(minor: int, required: tuple[str, ...], optional: tuple[tuple[str, bytes], ...]) -> bytes:
    required_bytes = tuple(_name(item, "required feature") for item in required)
    if tuple(sorted(required_bytes)) != required_bytes or len(set(required_bytes)) != len(required_bytes):
        fail("required features must be unique and sorted")
    optional_bytes = tuple((_name(name, "optional field"), value) for name, value in optional)
    names = tuple(name for name, _ in optional_bytes)
    if tuple(sorted(names)) != names or len(set(names)) != len(names):
        fail("optional fields must be unique and sorted")
    if any(not isinstance(value, bytes) or not HEX_RE.fullmatch(value) for _, value in optional_bytes):
        fail("optional value must be canonical lowercase hex")
    lines = [
        b"format\teshkol-byte-tokenizer\n",
        f"version\t1\t{minor}\n".encode(),
        f"required-features\t{len(required_bytes)}\n".encode(),
    ]
    lines.extend(b"required-feature\t" + item + b"\n" for item in required_bytes)
    lines.append(f"optional-fields\t{len(optional_bytes)}\n".encode())
    lines.extend(b"optional-field\t" + name + b"\t" + value + b"\n" for name, value in optional_bytes)
    lines.extend(V1_HEADER.splitlines(keepends=True)[4:])
    return b"".join(lines)


def encode_artifact(
    identity: Identity,
    *,
    minor: int = 0,
    required: tuple[str, ...] = (),
    optional: tuple[tuple[str, bytes], ...] = (),
) -> bytes:
    if minor < 0 or minor > MAX_I64:
        fail("minor: invalid")
    header = _header(minor, required, optional)
    body_payload = payload(identity)
    body = header + f"payload-bytes\t{len(body_payload)}\n".encode() + body_payload + b"checksum-algorithm\tsha256\n"
    digest = hashlib.sha256(CHECKSUM_DOMAIN + body).hexdigest().encode()
    result = body + b"checksum\t" + digest + b"\n"
    if len(result) > MAX_FILE:
        fail("artifact exceeds hard file limit")
    return result


def fingerprint(identity: Identity) -> str:
    digest = hashlib.sha256(IDENTITY_DOMAIN + V1_HEADER + payload(identity)).hexdigest()
    return f"sha256:eshkol-byte-tokenizer-v1:{digest}"


def encode_bytes(identity: Identity, source: bytes) -> tuple[int, ...]:
    identity = validate(identity)
    if not isinstance(source, bytes):
        fail("encode input must be bytes", "invalid-argument")
    if identity.utf8_policy == "strict":
        try:
            source.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            fail("encode input is malformed UTF-8", "invalid-argument")
    return identity.prefix + tuple(source) + identity.suffix


def decode_ids(identity: Identity, ids: tuple[int, ...]) -> bytes:
    identity = validate(identity)
    if not isinstance(ids, tuple):
        fail("decode IDs must be a tuple", "invalid-argument")
    by_id = {item.token_id: item for item in identity.specials}
    output = bytearray()
    for token_id in ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            fail("decode ID must be exact integer", "invalid-argument")
        if 0 <= token_id <= 255:
            output.append(token_id)
        elif token_id in by_id:
            if by_id[token_id].decode == "error":
                fail("special ID decode is forbidden", "invalid-argument")
        else:
            fail("decode ID out of vocabulary", "invalid-argument")
    result = bytes(output)
    if identity.utf8_policy == "strict":
        try:
            result.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            fail("decoded bytes are malformed UTF-8", "invalid-argument")
    return result


class _Reader:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.offset = 0

    def line(self, where: str) -> bytes:
        end = self.raw.find(b"\n", self.offset)
        if end < 0 or end - self.offset > MAX_LINE:
            fail(f"{where}: missing LF or overlong line")
        line = self.raw[self.offset:end]
        self.offset = end + 1
        return line


def _fields(reader: _Reader, name: bytes, count: int) -> tuple[bytes, ...]:
    fields = tuple(reader.line(name.decode()).split(b"\t"))
    if len(fields) != count or fields[0] != name:
        fail(f"{name.decode()}: malformed record")
    return fields


def _parse_payload(raw: bytes) -> Identity:
    reader = _Reader(raw)
    if _fields(reader, b"kind", 2)[1] != b"byte": fail("kind: unsupported")
    if _fields(reader, b"normalization", 2)[1] != b"none": fail("normalization: unsupported")
    utf8 = _fields(reader, b"utf8-policy", 2)[1]
    if utf8 not in {b"raw", b"strict"}: fail("utf8-policy: unsupported")
    if _fields(reader, b"byte-ids", 3)[1:] != (b"0", b"255"): fail("byte IDs: noncanonical")
    count = _u64(_fields(reader, b"special-count", 2)[1], "special-count", MAX_SPECIALS)
    specials: list[Special] = []
    for index in range(count):
        fields = _fields(reader, b"special", 4)
        token_id = _u64(fields[1], "special.id", 255 + MAX_SPECIALS)
        if token_id != 256 + index or not NAME_RE.fullmatch(fields[2]) or fields[3] not in {b"omit", b"error"}:
            fail("special: noncanonical")
        specials.append(Special(token_id, fields[2].decode(), fields[3].decode()))
    by_id = {item.token_id: item for item in specials}

    def insertion(label: bytes) -> tuple[int, ...]:
        count = _u64(_fields(reader, label + b"-count", 2)[1], "insertion count", MAX_INSERTIONS)
        result: list[int] = []
        for index in range(count):
            fields = _fields(reader, label, 3)
            if _u64(fields[1], "insertion index", MAX_INSERTIONS) != index:
                fail("insertion index: noncanonical")
            token_id = _u64(fields[2], "insertion ID", 255 + MAX_SPECIALS)
            if token_id not in by_id or by_id[token_id].decode != "omit":
                fail("insertion: invalid special reference")
            result.append(token_id)
        return tuple(result)

    prefix = insertion(b"prefix")
    suffix = insertion(b"suffix")
    if reader.offset != len(raw): fail("payload: trailing record")
    return validate(Identity(utf8.decode(), tuple(specials), prefix, suffix))


def decode_artifact(source: bytes, *, max_file: int = MAX_FILE, max_metadata: int = MAX_FILE) -> Parsed:
    if not isinstance(source, bytes): fail("artifact source must be bytes")
    if max_file < 1 or max_metadata < 1: fail("policy limit must be positive", "invalid-argument")
    if len(source) > min(max_file, MAX_FILE): fail("artifact exceeds effective file policy")
    try: source.decode("ascii")
    except UnicodeDecodeError: fail("artifact is not ASCII")
    if b"\r" in source or not source.endswith(b"\n"): fail("artifact newline is noncanonical")
    reader = _Reader(source)
    if _fields(reader, b"format", 2)[1] != b"eshkol-byte-tokenizer": fail("format: bad identity")
    version = _fields(reader, b"version", 3)
    major = _u64(version[1], "version major")
    minor = _u64(version[2], "version minor")
    if major != 1: fail("unsupported major", "version-mismatch")
    required_count = _u64(_fields(reader, b"required-features", 2)[1], "required count", MAX_HEADER_ITEMS)
    required: list[bytes] = []
    for _ in range(required_count):
        value = _fields(reader, b"required-feature", 2)[1]
        if not NAME_RE.fullmatch(value): fail("required feature: bad name")
        required.append(value)
    if required != sorted(set(required)): fail("required features: noncanonical")
    optional_count = _u64(_fields(reader, b"optional-fields", 2)[1], "optional count", MAX_HEADER_ITEMS)
    optional: list[tuple[bytes, bytes]] = []
    for _ in range(optional_count):
        fields = _fields(reader, b"optional-field", 3)
        if not NAME_RE.fullmatch(fields[1]) or not HEX_RE.fullmatch(fields[2]): fail("optional field: malformed")
        optional.append((fields[1], fields[2]))
    if [x[0] for x in optional] != sorted({x[0] for x in optional}): fail("optional fields: noncanonical")
    if minor == 0 and optional:
        fail("version 1.0 does not admit optional fields")
    if required: fail("required feature is unsupported", "unsupported")
    for expected in V1_HEADER.splitlines()[4:]:
        if reader.line("limit") != expected: fail("limit record: noncanonical")
    header = source[:reader.offset]
    payload_size = _u64(_fields(reader, b"payload-bytes", 2)[1], "payload bytes", MAX_FILE)
    if payload_size > min(max_metadata, MAX_FILE): fail("payload exceeds effective metadata policy")
    payload_end = reader.offset + payload_size
    if payload_end > len(source): fail("payload: truncated")
    raw_payload = source[reader.offset:payload_end]
    reader.offset = payload_end
    if _fields(reader, b"checksum-algorithm", 2)[1] != b"sha256": fail("checksum algorithm: unsupported", "unsupported")
    checksum_input = source[:reader.offset]
    checksum = _fields(reader, b"checksum", 2)[1]
    if not DIGEST_RE.fullmatch(checksum): fail("checksum: malformed")
    if reader.offset != len(source): fail("artifact: trailing bytes")
    actual = hashlib.sha256(CHECKSUM_DOMAIN + checksum_input).hexdigest().encode()
    if not hmac.compare_digest(checksum, actual): fail("checksum mismatch")
    identity = _parse_payload(raw_payload)
    canonical = encode_artifact(
        identity,
        minor=minor,
        optional=tuple((name.decode(), value) for name, value in optional),
    )
    if canonical != source: fail("artifact is not canonical")
    digest = hashlib.sha256(IDENTITY_DOMAIN + header + raw_payload).hexdigest()
    return Parsed(identity, f"sha256:eshkol-byte-tokenizer-v1:{digest}", minor,
                  tuple((name.decode(), value) for name, value in optional))
