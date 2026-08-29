"""Strict development-only codec for the proposed byte-tokenizer TSV v1 format.

This module is test evidence, not the Eshkol public tokenizer runtime. It neither
implements ``tokenizer-encode``/``tokenizer-decode`` nor supplies a fallback for
their currently unverified contiguous-i64 result contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import re
from typing import NoReturn, Sequence

FORMAT = b"eshkol-byte-tokenizer"
VERSION_MAJOR = 1
VERSION_MINOR = 0
MAX_ARTIFACT_BYTES = 1024 * 1024
MAX_SPECIALS = 4096
MAX_INSERTIONS = 4096
MAX_HEADER_ITEMS = 4096
MAX_OPTIONAL_VALUE_BYTES = 64
MAX_LINE_BYTES = 256

_NAME = re.compile(rb"[a-z][a-z0-9_.-]{0,63}\Z")
_DIGEST = re.compile(rb"[0-9a-f]{64}\Z")
_OPTIONAL_VALUE = re.compile(rb"[0-9a-f]*\Z")
_IDENTITY_DOMAIN = b"sha256:eshkol-byte-tokenizer-v1\n"
_CHECKSUM_DOMAIN = b"eshkol-byte-tokenizer-checksum-v1\n"
_V1_IDENTITY_HEADER = (
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


class TokenizerFormatError(ValueError):
    """Malformed, corrupt, noncanonical, or unsupported development artifact."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(f"{category}: {message}")
        self.category = category


def _fail(message: str, *, category: str = "corrupt-data") -> NoReturn:
    raise TokenizerFormatError(category, message)


@dataclass(frozen=True)
class SpecialToken:
    token_id: int
    name: str
    decode: str


@dataclass(frozen=True)
class TokenizerIdentity:
    utf8_policy: str
    specials: tuple[SpecialToken, ...] = ()
    prefix: tuple[int, ...] = ()
    suffix: tuple[int, ...] = ()


@dataclass(frozen=True)
class ParsedArtifact:
    identity: TokenizerIdentity
    payload: bytes
    fingerprint: str
    version_minor: int
    optional_fields: tuple[tuple[str, bytes], ...]


def _unsigned_decimal(raw: bytes, where: str, *, maximum: int) -> int:
    if not raw or (raw != b"0" and (raw[:1] == b"0" or not raw.isdigit())):
        _fail(f"{where}: expected canonical unsigned decimal")
    if raw == b"0":
        return 0
    if not raw.isdigit():
        _fail(f"{where}: expected canonical unsigned decimal")
    value = int(raw)
    if value > maximum:
        _fail(f"{where}: exceeds {maximum}")
    return value


def _ascii_name(name: object, where: str) -> bytes:
    if not isinstance(name, str):
        _fail(f"{where}: expected string")
    try:
        encoded = name.encode("ascii", errors="strict")
    except UnicodeEncodeError as error:
        raise TokenizerFormatError("corrupt-data", f"{where}: expected ASCII") from error
    if _NAME.fullmatch(encoded) is None:
        _fail(f"{where}: invalid special-token name")
    return encoded


def _validate_identity(identity: object) -> TokenizerIdentity:
    if not isinstance(identity, TokenizerIdentity):
        _fail("identity: expected TokenizerIdentity")
    if identity.utf8_policy not in {"raw", "strict"}:
        _fail("identity.utf8_policy: expected 'raw' or 'strict'")
    if len(identity.specials) > MAX_SPECIALS:
        _fail(f"identity.specials: exceeds {MAX_SPECIALS}")
    if len(identity.prefix) > MAX_INSERTIONS:
        _fail(f"identity.prefix: exceeds {MAX_INSERTIONS}")
    if len(identity.suffix) > MAX_INSERTIONS:
        _fail(f"identity.suffix: exceeds {MAX_INSERTIONS}")

    specials: list[SpecialToken] = []
    for index, special in enumerate(identity.specials):
        if not isinstance(special, SpecialToken):
            _fail(f"identity.specials[{index}]: expected SpecialToken")
        if isinstance(special.token_id, bool) or not isinstance(special.token_id, int):
            _fail(f"identity.specials[{index}].token_id: expected integer")
        _ascii_name(special.name, f"identity.specials[{index}].name")
        if special.decode not in {"omit", "error"}:
            _fail(f"identity.specials[{index}].decode: expected 'omit' or 'error'")
        specials.append(special)
    specials.sort(key=lambda special: special.token_id)
    for index, special in enumerate(specials):
        expected = 256 + index
        if special.token_id != expected:
            _fail(
                f"identity.specials[{index}].token_id: expected contiguous ID "
                f"{expected}"
            )
    names = [special.name for special in specials]
    if len(names) != len(set(names)):
        _fail("identity.specials: duplicate special-token name")
    by_id = {special.token_id: special for special in specials}

    def validate_insertions(values: Sequence[int], where: str) -> tuple[int, ...]:
        checked: list[int] = []
        for index, token_id in enumerate(values):
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                _fail(f"{where}[{index}]: expected integer token ID")
            special = by_id.get(token_id)
            if special is None:
                _fail(f"{where}[{index}]: unknown special-token ID {token_id}")
            if special.decode != "omit":
                _fail(f"{where}[{index}]: inserted special token must decode as omit")
            checked.append(token_id)
        return tuple(checked)

    return TokenizerIdentity(
        utf8_policy=identity.utf8_policy,
        specials=tuple(specials),
        prefix=validate_insertions(identity.prefix, "identity.prefix"),
        suffix=validate_insertions(identity.suffix, "identity.suffix"),
    )


def _payload(identity: TokenizerIdentity) -> bytes:
    checked = _validate_identity(identity)
    lines = [
        b"kind\tbyte\n",
        b"normalization\tnone\n",
        b"utf8-policy\t" + checked.utf8_policy.encode("ascii") + b"\n",
        b"byte-ids\t0\t255\n",
        f"special-count\t{len(checked.specials)}\n".encode("ascii"),
    ]
    for special in checked.specials:
        lines.append(
            b"special\t"
            + str(special.token_id).encode("ascii")
            + b"\t"
            + _ascii_name(special.name, "special.name")
            + b"\t"
            + special.decode.encode("ascii")
            + b"\n"
        )
    lines.append(f"prefix-count\t{len(checked.prefix)}\n".encode("ascii"))
    for position, token_id in enumerate(checked.prefix):
        lines.append(f"prefix\t{position}\t{token_id}\n".encode("ascii"))
    lines.append(f"suffix-count\t{len(checked.suffix)}\n".encode("ascii"))
    for position, token_id in enumerate(checked.suffix):
        lines.append(f"suffix\t{position}\t{token_id}\n".encode("ascii"))
    return b"".join(lines)


def fingerprint(identity: TokenizerIdentity) -> str:
    """Return the public lexical fingerprint for a validated identity."""
    digest = hashlib.sha256(
        _IDENTITY_DOMAIN + _V1_IDENTITY_HEADER + _payload(identity)
    ).hexdigest()
    return f"sha256:eshkol-byte-tokenizer-v1:{digest}"


def encode_artifact(identity: TokenizerIdentity) -> bytes:
    """Encode one byte-identical canonical development artifact."""
    checked = _validate_identity(identity)
    payload = _payload(checked)
    body = (
        _V1_IDENTITY_HEADER
        + f"payload-bytes\t{len(payload)}\n".encode("ascii")
        + payload
        + b"checksum-algorithm\tsha256\n"
    )
    digest = hashlib.sha256(_CHECKSUM_DOMAIN + body).hexdigest().encode("ascii")
    artifact = body + b"checksum\t" + digest + b"\n"
    if len(artifact) > MAX_ARTIFACT_BYTES:
        _fail(f"artifact exceeds {MAX_ARTIFACT_BYTES} bytes")
    return artifact


def _line(raw: bytes, offset: int, where: str) -> tuple[bytes, int]:
    newline = raw.find(b"\n", offset)
    if newline < 0:
        _fail(f"{where}: missing final LF")
    if newline - offset > MAX_LINE_BYTES:
        _fail(f"{where}: line exceeds {MAX_LINE_BYTES} bytes")
    return raw[offset:newline], newline + 1


def _exact_fields(line: bytes, expected: tuple[bytes, ...], where: str) -> None:
    if tuple(line.split(b"\t")) != expected:
        _fail(f"{where}: unexpected fields")


def _parse_payload(payload: bytes) -> TokenizerIdentity:
    if not payload.endswith(b"\n"):
        _fail("payload: missing final LF")
    start = 0
    while start < len(payload):
        newline = payload.find(b"\n", start)
        if newline < 0 or newline - start > MAX_LINE_BYTES:
            _fail(f"payload: line exceeds {MAX_LINE_BYTES} bytes")
        start = newline + 1
    lines = payload[:-1].split(b"\n")
    cursor = 0

    def take(where: str) -> list[bytes]:
        nonlocal cursor
        if cursor >= len(lines):
            _fail(f"{where}: missing line")
        fields = lines[cursor].split(b"\t")
        cursor += 1
        return fields

    if take("payload.kind") != [b"kind", b"byte"]:
        _fail("payload.kind: expected kind byte")
    if take("payload.normalization") != [b"normalization", b"none"]:
        _fail("payload.normalization: expected none")
    fields = take("payload.utf8-policy")
    if len(fields) != 2 or fields[0] != b"utf8-policy" or fields[1] not in {
        b"raw",
        b"strict",
    }:
        _fail("payload.utf8-policy: expected raw or strict")
    utf8_policy = fields[1].decode("ascii")
    if take("payload.byte-ids") != [b"byte-ids", b"0", b"255"]:
        _fail("payload.byte-ids: expected exact range 0..255")

    fields = take("payload.special-count")
    if len(fields) != 2 or fields[0] != b"special-count":
        _fail("payload.special-count: unexpected fields")
    special_count = _unsigned_decimal(
        fields[1], "payload.special-count", maximum=MAX_SPECIALS
    )
    specials: list[SpecialToken] = []
    names: set[str] = set()
    for index in range(special_count):
        fields = take(f"payload.special[{index}]")
        if len(fields) != 4 or fields[0] != b"special":
            _fail(f"payload.special[{index}]: unexpected fields")
        token_id = _unsigned_decimal(
            fields[1], f"payload.special[{index}].id", maximum=256 + MAX_SPECIALS - 1
        )
        if token_id != 256 + index:
            _fail(f"payload.special[{index}].id: expected {256 + index}")
        if _NAME.fullmatch(fields[2]) is None:
            _fail(f"payload.special[{index}].name: invalid special-token name")
        name = fields[2].decode("ascii")
        if name in names:
            _fail("payload.specials: duplicate special-token name")
        names.add(name)
        if fields[3] not in {b"omit", b"error"}:
            _fail(f"payload.special[{index}].decode: expected omit or error")
        specials.append(SpecialToken(token_id, name, fields[3].decode("ascii")))
    by_id = {special.token_id: special for special in specials}

    def parse_insertions(label: bytes) -> tuple[int, ...]:
        count_fields = take(f"payload.{label.decode()}-count")
        if len(count_fields) != 2 or count_fields[0] != label + b"-count":
            _fail(f"payload.{label.decode()}-count: unexpected fields")
        count = _unsigned_decimal(
            count_fields[1],
            f"payload.{label.decode()}-count",
            maximum=MAX_INSERTIONS,
        )
        values: list[int] = []
        for position in range(count):
            fields = take(f"payload.{label.decode()}[{position}]")
            if len(fields) != 3 or fields[0] != label:
                _fail(f"payload.{label.decode()}[{position}]: unexpected fields")
            actual_position = _unsigned_decimal(
                fields[1],
                f"payload.{label.decode()}[{position}].position",
                maximum=MAX_INSERTIONS - 1,
            )
            if actual_position != position:
                _fail(
                    f"payload.{label.decode()}[{position}].position: expected {position}"
                )
            token_id = _unsigned_decimal(
                fields[2],
                f"payload.{label.decode()}[{position}].id",
                maximum=256 + MAX_SPECIALS - 1,
            )
            special = by_id.get(token_id)
            if special is None:
                _fail(f"payload.{label.decode()}[{position}].id: unknown special")
            if special.decode != "omit":
                _fail(
                    f"payload.{label.decode()}[{position}].id: inserted special "
                    "must decode as omit"
                )
            values.append(token_id)
        return tuple(values)

    prefix = parse_insertions(b"prefix")
    suffix = parse_insertions(b"suffix")
    if cursor != len(lines):
        _fail("payload: trailing or unknown line")
    return TokenizerIdentity(utf8_policy, tuple(specials), prefix, suffix)


def decode_artifact(source: bytes | bytearray | memoryview) -> ParsedArtifact:
    """Strictly validate canonical bytes and return inert identity data."""
    try:
        source_size = len(source)
    except TypeError as error:
        raise TokenizerFormatError(
            "corrupt-data", "artifact source must have a bounded byte length"
        ) from error
    if source_size > MAX_ARTIFACT_BYTES:
        _fail(f"artifact exceeds {MAX_ARTIFACT_BYTES} bytes")
    raw = bytes(source)
    if len(raw) != source_size:
        _fail("artifact source length changed during conversion")
    try:
        raw.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise TokenizerFormatError("corrupt-data", "artifact must be ASCII") from error
    if b"\r" in raw:
        _fail("artifact must use LF, not CRLF")

    offset = 0
    line, offset = _line(raw, offset, "format")
    _exact_fields(line, (b"format", FORMAT), "format")
    line, offset = _line(raw, offset, "version")
    fields = tuple(line.split(b"\t"))
    if len(fields) != 3 or fields[0] != b"version":
        _fail("version: unexpected fields")
    major = _unsigned_decimal(fields[1], "version.major", maximum=2**31 - 1)
    minor = _unsigned_decimal(fields[2], "version.minor", maximum=2**31 - 1)
    if major != VERSION_MAJOR:
        _fail(f"unsupported version {major}.{minor}", category="version-mismatch")
    line, offset = _line(raw, offset, "required-features")
    fields = tuple(line.split(b"\t"))
    if len(fields) != 2 or fields[0] != b"required-features":
        _fail("required-features: unexpected fields")
    feature_count = _unsigned_decimal(
        fields[1], "required-features", maximum=MAX_HEADER_ITEMS
    )
    required_features: list[bytes] = []
    for index in range(feature_count):
        line, offset = _line(raw, offset, f"required-feature[{index}]")
        fields = tuple(line.split(b"\t"))
        if (
            len(fields) != 2
            or fields[0] != b"required-feature"
            or _NAME.fullmatch(fields[1]) is None
        ):
            _fail(f"required-feature[{index}]: unexpected fields or invalid name")
        required_features.append(fields[1])
    if required_features != sorted(required_features):
        _fail("required-feature records must be sorted")
    if len(required_features) != len(set(required_features)):
        _fail("required-feature records must be unique")

    line, offset = _line(raw, offset, "optional-fields")
    fields = tuple(line.split(b"\t"))
    if len(fields) != 2 or fields[0] != b"optional-fields":
        _fail("optional-fields: unexpected fields")
    optional_count = _unsigned_decimal(
        fields[1], "optional-fields", maximum=MAX_HEADER_ITEMS
    )
    optional_fields: list[tuple[bytes, bytes]] = []
    for index in range(optional_count):
        line, offset = _line(raw, offset, f"optional-field[{index}]")
        fields = tuple(line.split(b"\t"))
        if (
            len(fields) != 3
            or fields[0] != b"optional-field"
            or _NAME.fullmatch(fields[1]) is None
        ):
            _fail(f"optional-field[{index}]: unexpected fields or invalid name")
        value = fields[2]
        if (
            not value
            or len(value) > 2 * MAX_OPTIONAL_VALUE_BYTES
            or len(value) % 2 != 0
            or _OPTIONAL_VALUE.fullmatch(value) is None
        ):
            _fail(
                f"optional-field[{index}]: value must be at most 128 even lowercase "
                "hexadecimal characters and must not be empty"
            )
        optional_fields.append((fields[1], value))
    optional_names = [name for name, _value in optional_fields]
    if optional_names != sorted(optional_names):
        _fail("optional-field records must be sorted")
    if len(optional_names) != len(set(optional_names)):
        _fail("optional-field records must be unique")

    if feature_count != 0:
        _fail(
            f"required features are unsupported in version 1.{minor}",
            category="unsupported",
        )

    limits = (
        (b"limit-file-bytes", b"1048576"),
        (b"limit-specials", b"4096"),
        (b"limit-name-bytes", b"64"),
        (b"limit-header-items", b"4096"),
        (b"limit-optional-value-bytes", b"64"),
        (b"limit-prefix", b"4096"),
        (b"limit-suffix", b"4096"),
    )
    for name, value in limits:
        line, offset = _line(raw, offset, name.decode("ascii"))
        _exact_fields(line, (name, value), name.decode("ascii"))
    identity_header = raw[:offset]
    line, offset = _line(raw, offset, "payload-bytes")
    fields = tuple(line.split(b"\t"))
    if len(fields) != 2 or fields[0] != b"payload-bytes":
        _fail("payload-bytes: unexpected fields")
    payload_size = _unsigned_decimal(
        fields[1], "payload-bytes", maximum=MAX_ARTIFACT_BYTES
    )
    payload_end = offset + payload_size
    if payload_end > len(raw):
        _fail("payload: truncated")
    payload = raw[offset:payload_end]
    offset = payload_end
    line, algorithm_end = _line(raw, offset, "checksum-algorithm")
    _exact_fields(
        line, (b"checksum-algorithm", b"sha256"), "checksum-algorithm"
    )
    checksum_input = raw[:algorithm_end]
    offset = algorithm_end
    line, offset = _line(raw, offset, "checksum")
    fields = tuple(line.split(b"\t"))
    if len(fields) != 2 or fields[0] != b"checksum" or _DIGEST.fullmatch(fields[1]) is None:
        _fail("checksum: expected 64 lowercase hexadecimal digits")
    if offset != len(raw):
        _fail("artifact: trailing bytes")
    actual = hashlib.sha256(_CHECKSUM_DOMAIN + checksum_input).hexdigest().encode("ascii")
    if not hmac.compare_digest(fields[1], actual):
        _fail("checksum mismatch")

    identity = _parse_payload(payload)
    if minor == VERSION_MINOR:
        if encode_artifact(identity) != raw:
            _fail("version 1.0 artifact bytes are not canonical")
    else:
        canonical_body = (
            identity_header
            + f"payload-bytes\t{len(payload)}\n".encode("ascii")
            + _payload(identity)
            + b"checksum-algorithm\tsha256\n"
        )
        canonical_digest = hashlib.sha256(
            _CHECKSUM_DOMAIN + canonical_body
        ).hexdigest().encode("ascii")
        canonical = canonical_body + b"checksum\t" + canonical_digest + b"\n"
        if canonical != raw:
            _fail("higher-minor artifact bytes are not canonical")
    identity_digest = hashlib.sha256(
        _IDENTITY_DOMAIN + identity_header + payload
    ).hexdigest()
    return ParsedArtifact(
        identity=identity,
        payload=payload,
        fingerprint=f"sha256:eshkol-byte-tokenizer-v1:{identity_digest}",
        version_minor=minor,
        optional_fields=tuple(
            (name.decode("ascii"), value) for name, value in optional_fields
        ),
    )
