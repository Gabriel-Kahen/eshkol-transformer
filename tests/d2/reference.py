"""Carrier-neutral, development-only oracle for accepted D2 semantics.

This module intentionally does not import or name a production D2 implementation.
It materializes rows and permutations for small tests, so it is an oracle rather
than a memory-bounded loader. Its cursor encoder independently implements the
accepted canonical ESHKDCU1 version-1.0 bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Iterable, NoReturn, Sequence


MAX_I64 = (1 << 63) - 1
MAX_U64 = (1 << 64) - 1
BATCH_ELEMENT_BYTES = 8 + 8 + 1
ROW_ORDINAL_BYTES = 8

SHUFFLE_DOMAIN = b"eshkol-d2-window-shuffle-v1\n"
CURSOR_DOMAIN = b"eshkol-token-dataset-cursor-checksum-v1\n"
CURSOR_MAGIC = b"ESHKDCU1"
CURSOR_FIXED_BYTES = 208


class D2ReferenceError(ValueError):
    """An oracle rejection with the intended structured-error category."""

    def __init__(self, category: str, message: str) -> None:
        self.category = category
        super().__init__(f"{category}: {message}")


def _fail(category: str, message: str) -> NoReturn:
    raise D2ReferenceError(category, message)


def exact_i64(value: object, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("invalid-argument", f"{where}: expected exact signed-i64")
    if not minimum <= value <= MAX_I64:
        _fail("invalid-argument", f"{where}: out of range")
    return value


def checked_add(left: int, right: int, where: str, maximum: int = MAX_I64) -> int:
    if left < 0 or right < 0 or left > maximum - right:
        _fail("invalid-argument", f"{where}: addition overflow")
    return left + right


def checked_mul(left: int, right: int, where: str, maximum: int = MAX_I64) -> int:
    if left < 0 or right < 0 or (right and left > maximum // right):
        _fail("invalid-argument", f"{where}: multiplication overflow")
    return left * right


def packed_row_count(token_count: object, sequence_length: object) -> int:
    """Return 0 for L<2, else 1+floor((L-2)/T), without L+T arithmetic."""

    length = exact_i64(token_count, "token-count")
    width = exact_i64(sequence_length, "sequence-length", 1)
    return 0 if length < 2 else 1 + (length - 2) // width


def unpacked_row_count(shard_lengths: Iterable[object], sequence_length: object) -> int:
    width = exact_i64(sequence_length, "sequence-length", 1)
    total = 0
    for index, raw_length in enumerate(shard_lengths):
        rows = packed_row_count(raw_length, width)
        total = checked_add(total, rows, f"row-count[{index}]")
    return total


def batch_payload_bytes(
    batch_size: object,
    sequence_length: object,
    maximum_batch_bytes: object,
    *,
    size_max: int = MAX_U64,
) -> int:
    """Exact storage for two i64[N,T] arrays and one uint8 bool[N,T] array."""

    n = exact_i64(batch_size, "batch-size", 1)
    t = exact_i64(sequence_length, "sequence-length", 1)
    limit = exact_i64(maximum_batch_bytes, "maximum-batch-bytes", 1)
    if size_max < 0:
        _fail("invalid-argument", "SIZE_MAX: out of range")
    elements = checked_mul(n, t, "batch elements", min(MAX_I64, size_max))
    payload = checked_mul(
        elements, BATCH_ELEMENT_BYTES, "batch payload", min(MAX_I64, size_max)
    )
    if payload > limit:
        _fail("invalid-argument", "batch payload exceeds configured limit")
    return payload


def working_payload_bytes(
    maximum_manifest_bytes: object,
    maximum_shard_bytes: object,
    batch_size: object,
    sequence_length: object,
    shuffle_window_rows: object,
    total_rows: object,
    *,
    maximum_batch_bytes: object,
    size_max: int = MAX_U64,
) -> int:
    """Carrier-neutral data payload ceiling, excluding allocator/control overhead.

    The conservative simultaneous formula is one bounded manifest window, one
    bounded serialized shard, the three batch arrays, and one uint64 permutation
    ordinal per admitted row in the current shuffle window.  It is not a new
    public configuration limit.
    """

    manifest = exact_i64(maximum_manifest_bytes, "maximum-manifest-bytes", 1)
    shard = exact_i64(maximum_shard_bytes, "maximum-shard-bytes", 1)
    window = exact_i64(shuffle_window_rows, "shuffle-window-rows", 1)
    rows = exact_i64(total_rows, "total-rows")
    batch = batch_payload_bytes(
        batch_size, sequence_length, maximum_batch_bytes, size_max=size_max
    )
    ordinal_bytes = checked_mul(
        min(window, rows), ROW_ORDINAL_BYTES, "shuffle ordinal bytes", min(MAX_I64, size_max)
    )
    result = checked_add(manifest, shard, "working payload", min(MAX_I64, size_max))
    result = checked_add(result, batch, "working payload", min(MAX_I64, size_max))
    result = checked_add(result, ordinal_bytes, "working payload", min(MAX_I64, size_max))
    return result


@dataclass(frozen=True, slots=True)
class Row:
    inputs: tuple[int, ...]
    targets: tuple[int, ...]
    loss_mask: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class Batch:
    inputs: tuple[tuple[int, ...], ...]
    targets: tuple[tuple[int, ...], ...]
    loss_mask: tuple[tuple[bool, ...], ...]


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    batch_size: int
    sequence_length: int
    shuffle_seed: int | None = None
    shuffle_window_rows: int = 1
    packing: bool = True
    directory: str = "corpus"
    maximum_manifest_bytes: int = 1_048_576
    maximum_shard_bytes: int = 1_048_576
    maximum_total_tokens: int = 65_536
    maximum_batch_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        exact_i64(self.batch_size, "batch-size", 1)
        exact_i64(self.sequence_length, "sequence-length", 1)
        if self.shuffle_seed is not None:
            exact_i64(self.shuffle_seed, "shuffle-seed")
        exact_i64(self.shuffle_window_rows, "shuffle-window-rows", 1)
        if not isinstance(self.packing, bool):
            _fail("invalid-argument", "packing: expected boolean")
        if not isinstance(self.directory, str) or not self.directory or "\x00" in self.directory:
            _fail("invalid-argument", "directory: expected nonempty UTF-8 without NUL")
        exact_i64(self.maximum_manifest_bytes, "maximum-manifest-bytes", 1)
        exact_i64(self.maximum_shard_bytes, "maximum-shard-bytes", 1)
        exact_i64(self.maximum_total_tokens, "maximum-total-tokens")
        exact_i64(self.maximum_batch_bytes, "maximum-batch-bytes", 1)
        batch_payload_bytes(
            self.batch_size, self.sequence_length, self.maximum_batch_bytes
        )


def _tokens(values: Sequence[object], where: str) -> tuple[int, ...]:
    result = []
    for index, value in enumerate(values):
        result.append(exact_i64(value, f"{where}[{index}]"))
    return tuple(result)


def _rows_for_stream(values: tuple[int, ...], width: int) -> list[Row]:
    rows = []
    for ordinal in range(packed_row_count(len(values), width)):
        start = ordinal * width
        true_count = min(width, len(values) - 1 - start)
        inputs = values[start : start + true_count] + (0,) * (width - true_count)
        targets = values[start + 1 : start + 1 + true_count] + (0,) * (width - true_count)
        mask = (True,) * true_count + (False,) * (width - true_count)
        rows.append(Row(inputs, targets, mask))
    return rows


def logical_rows(
    shards: Sequence[Sequence[object]], sequence_length: object, packing: bool
) -> tuple[Row, ...]:
    width = exact_i64(sequence_length, "sequence-length", 1)
    if not isinstance(packing, bool):
        _fail("invalid-argument", "packing: expected boolean")
    normalized = tuple(_tokens(shard, f"shards[{index}]") for index, shard in enumerate(shards))
    if packing:
        return tuple(_rows_for_stream(tuple(token for shard in normalized for token in shard), width))
    return tuple(row for shard in normalized for row in _rows_for_stream(shard, width))


def _draw(seed: int, window_index: int, counter: int, modulus: int) -> tuple[int, int]:
    """Modulo-bias-free reference draw; counter is returned after consumption."""

    threshold = ((1 << 64) // modulus) * modulus
    while True:
        if counter > MAX_U64:
            _fail("unsupported", "shuffle draw counter exhausted unsigned-u64")
        material = struct.pack("<QQQ", seed, window_index, counter)
        word = int.from_bytes(
            hashlib.sha256(SHUFFLE_DOMAIN + material).digest()[:8], "little"
        )
        if word < threshold:
            return word % modulus, counter + 1
        if counter == MAX_U64:
            _fail("unsupported", "shuffle draw counter exhausted unsigned-u64")
        counter += 1


def rejection_sample_words(
    modulus: object, words: Iterable[object], *, start_counter: object = 0
) -> tuple[int, int]:
    """Expose rejection arithmetic independently of SHA-256 for boundary tests."""

    bound = exact_i64(modulus, "shuffle modulus", 1)
    if (
        isinstance(start_counter, bool)
        or not isinstance(start_counter, int)
        or not 0 <= start_counter <= MAX_U64
    ):
        _fail("unsupported", "shuffle draw counter exhausted unsigned-u64")
    threshold = ((1 << 64) // bound) * bound
    counter = start_counter
    for candidate in words:
        if (
            isinstance(candidate, bool)
            or not isinstance(candidate, int)
            or not 0 <= candidate <= MAX_U64
        ):
            _fail("invalid-argument", "shuffle candidate: expected unsigned-u64")
        if candidate < threshold:
            return candidate % bound, counter + 1
        if counter == MAX_U64:
            _fail("unsupported", "shuffle draw counter exhausted unsigned-u64")
        counter += 1
    _fail("unsupported", "shuffle candidate source exhausted")


def window_permutation(total_rows: object, seed: object, window_rows: object) -> tuple[int, ...]:
    total = exact_i64(total_rows, "total-rows")
    window = exact_i64(window_rows, "shuffle-window-rows", 1)
    if seed is None:
        return tuple(range(total))
    normalized_seed = exact_i64(seed, "shuffle-seed")
    output: list[int] = []
    window_index = 0
    start = 0
    while start < total:
        stop = start + min(window, total - start)
        values = list(range(start, stop))
        counter = 0
        for index in range(len(values) - 1, 0, -1):
            selected, counter = _draw(
                normalized_seed, window_index, counter, index + 1
            )
            values[index], values[selected] = values[selected], values[index]
        output.extend(values)
        start = stop
        window_index += 1
    return tuple(output)


def _cursor_config_identity(config: ReferenceConfig) -> tuple[object, ...]:
    return (
        config.batch_size,
        config.sequence_length,
        config.maximum_manifest_bytes,
        config.maximum_shard_bytes,
        config.maximum_total_tokens,
        config.maximum_batch_bytes,
        config.shuffle_seed,
        config.shuffle_window_rows,
        config.packing,
    )


@dataclass(frozen=True, slots=True)
class CursorFields:
    manifest_digest: bytes
    tokenizer_fingerprint: str
    vocab_size: int
    config: ReferenceConfig
    total_rows: int
    next_ordinal: int


def encode_reference_cursor(
    manifest_digest: bytes,
    tokenizer_fingerprint: str,
    vocab_size: int,
    config: ReferenceConfig,
    total_rows: int,
    next_ordinal: int,
) -> bytes:
    if not isinstance(manifest_digest, bytes) or len(manifest_digest) != 32:
        _fail("invalid-argument", "manifest digest: expected 32 bytes")
    if not isinstance(tokenizer_fingerprint, str):
        _fail("invalid-argument", "tokenizer fingerprint: expected UTF-8 string")
    fingerprint = tokenizer_fingerprint.encode("utf-8")
    if not 1 <= len(fingerprint) <= 192:
        _fail("invalid-argument", "tokenizer fingerprint: expected 1..192 UTF-8 bytes")
    vocab = exact_i64(vocab_size, "vocab-size", 1)
    total = exact_i64(total_rows, "cursor total-rows")
    ordinal = exact_i64(next_ordinal, "cursor next-ordinal")
    if ordinal > total:
        _fail("invalid-argument", "cursor next-ordinal exceeds total rows")
    if not isinstance(config, ReferenceConfig):
        _fail("invalid-argument", "cursor config: expected ReferenceConfig")

    header_bytes = 176 + len(fingerprint)
    cursor_bytes = CURSOR_FIXED_BYTES + len(fingerprint)
    seed_present = int(config.shuffle_seed is not None)
    raw = bytearray(cursor_bytes)
    raw[:8] = CURSOR_MAGIC
    struct.pack_into("<HH", raw, 8, 1, 0)
    struct.pack_into("<IIIIIII", raw, 12, header_bytes, 0, 1, seed_present,
                     int(config.packing), seed_present, len(fingerprint))
    struct.pack_into(
        "<QQQQQQQQQQQ",
        raw,
        40,
        vocab,
        config.batch_size,
        config.sequence_length,
        config.maximum_manifest_bytes,
        config.maximum_shard_bytes,
        config.maximum_total_tokens,
        config.maximum_batch_bytes,
        0 if config.shuffle_seed is None else config.shuffle_seed,
        config.shuffle_window_rows,
        total,
        ordinal,
    )
    raw[128:160] = manifest_digest
    struct.pack_into("<QQ", raw, 160, cursor_bytes, 0)
    raw[176:header_bytes] = fingerprint
    raw[header_bytes:] = hashlib.sha256(CURSOR_DOMAIN + raw[:header_bytes]).digest()
    return bytes(raw)


def decode_reference_cursor(raw: object) -> CursorFields:
    if not isinstance(raw, bytes):
        _fail("invalid-argument", "cursor: expected bytevector bytes")
    if not 209 <= len(raw) <= 400:
        _fail("corrupt-data", "cursor: noncanonical physical byte length")
    if raw[:8] != CURSOR_MAGIC:
        _fail("corrupt-data", "cursor: bad magic")
    header_bytes, features, checksum_id, shuffle_id, packing, seed_present, fp_bytes = (
        struct.unpack_from("<IIIIIII", raw, 12)
    )
    declared_bytes, reserved = struct.unpack_from("<QQ", raw, 160)
    if (
        not 1 <= fp_bytes <= 192
        or header_bytes != 176 + fp_bytes
        or len(raw) != CURSOR_FIXED_BYTES + fp_bytes
        or declared_bytes != len(raw)
    ):
        _fail("corrupt-data", "cursor: noncanonical size arithmetic")
    if struct.unpack_from("<HH", raw, 8) != (1, 0):
        _fail("version-mismatch", "cursor: unsupported version")
    if features != 0:
        _fail("unsupported", "cursor: unknown required feature")
    if checksum_id != 1:
        _fail("unsupported", "cursor: unknown checksum algorithm")
    if shuffle_id not in (0, 1):
        _fail("unsupported", "cursor: unknown shuffle algorithm")

    values = struct.unpack_from("<QQQQQQQQQQQ", raw, 40)
    (
        vocab,
        batch_size,
        sequence_length,
        maximum_manifest_bytes,
        maximum_shard_bytes,
        maximum_total_tokens,
        maximum_batch_bytes,
        seed_word,
        shuffle_window_rows,
        total_rows,
        next_ordinal,
    ) = values
    try:
        fingerprint = raw[176:header_bytes].decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise D2ReferenceError("corrupt-data", "cursor: noncanonical UTF-8") from error
    if (
        packing not in (0, 1)
        or seed_present not in (0, 1)
        or (shuffle_id, seed_present) not in ((0, 0), (1, 1))
        or (seed_present == 0 and seed_word != 0)
        or reserved != 0
        or fingerprint.encode("utf-8") != raw[176:header_bytes]
        or not 1 <= vocab <= MAX_I64
        or not 1 <= batch_size <= MAX_I64
        or not 1 <= sequence_length <= MAX_I64
        or not 1 <= maximum_manifest_bytes <= MAX_I64
        or not 1 <= maximum_shard_bytes <= MAX_I64
        or not 0 <= maximum_total_tokens <= MAX_I64
        or not 1 <= maximum_batch_bytes <= MAX_I64
        or not 0 <= seed_word <= MAX_I64
        or not 1 <= shuffle_window_rows <= MAX_I64
        or not 0 <= total_rows <= MAX_I64
        or not 0 <= next_ordinal <= MAX_I64
    ):
        _fail("corrupt-data", "cursor: noncanonical semantic fields")
    try:
        config = ReferenceConfig(
            batch_size,
            sequence_length,
            None if seed_present == 0 else seed_word,
            shuffle_window_rows,
            bool(packing),
            maximum_manifest_bytes=maximum_manifest_bytes,
            maximum_shard_bytes=maximum_shard_bytes,
            maximum_total_tokens=maximum_total_tokens,
            maximum_batch_bytes=maximum_batch_bytes,
        )
        working_payload_bytes(
            maximum_manifest_bytes,
            maximum_shard_bytes,
            batch_size,
            sequence_length,
            shuffle_window_rows,
            total_rows,
            maximum_batch_bytes=maximum_batch_bytes,
        )
    except D2ReferenceError as error:
        raise D2ReferenceError("corrupt-data", "cursor: noncanonical resource fields") from error
    if hashlib.sha256(CURSOR_DOMAIN + raw[:header_bytes]).digest() != raw[header_bytes:]:
        _fail("corrupt-data", "cursor: checksum mismatch")
    return CursorFields(
        raw[128:160], fingerprint, vocab, config, total_rows, next_ordinal
    )


def resign_reference_cursor(raw: bytes) -> bytes:
    if not isinstance(raw, bytes) or not 209 <= len(raw) <= 400:
        _fail("invalid-argument", "cursor: cannot resign noncanonical length")
    header_bytes = struct.unpack_from("<I", raw, 12)[0]
    if header_bytes + 32 != len(raw):
        _fail("invalid-argument", "cursor: cannot resign noncanonical header")
    return raw[:header_bytes] + hashlib.sha256(CURSOR_DOMAIN + raw[:header_bytes]).digest()


class ReferenceDataset:
    """Small materializing state machine used only to establish D2 semantics."""

    def __init__(
        self,
        shards: Sequence[Sequence[object]],
        config: ReferenceConfig,
        *,
        manifest_digest: bytes,
        tokenizer_fingerprint: str,
        vocab_size: int,
    ) -> None:
        if not isinstance(config, ReferenceConfig):
            _fail("invalid-argument", "config: expected ReferenceConfig")
        self.config = config
        self.rows = logical_rows(shards, config.sequence_length, config.packing)
        self.order = window_permutation(
            len(self.rows), config.shuffle_seed, config.shuffle_window_rows
        )
        if not isinstance(manifest_digest, bytes) or len(manifest_digest) != 32:
            _fail("invalid-argument", "manifest digest: expected 32 bytes")
        if not isinstance(tokenizer_fingerprint, str):
            _fail("invalid-argument", "tokenizer fingerprint: expected UTF-8 string")
        encoded_fingerprint = tokenizer_fingerprint.encode("utf-8")
        if not 1 <= len(encoded_fingerprint) <= 192:
            _fail("invalid-argument", "tokenizer fingerprint: expected 1..192 UTF-8 bytes")
        self.manifest_digest = manifest_digest
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.vocab_size = exact_i64(vocab_size, "vocab-size", 1)
        self.next_ordinal = 0

    def snapshot(self) -> bytes:
        return encode_reference_cursor(
            self.manifest_digest,
            self.tokenizer_fingerprint,
            self.vocab_size,
            self.config,
            len(self.rows),
            self.next_ordinal,
        )

    def seek(self, cursor: bytes) -> None:
        fields = decode_reference_cursor(cursor)
        if (
            fields.manifest_digest != self.manifest_digest
            or fields.tokenizer_fingerprint != self.tokenizer_fingerprint
            or fields.vocab_size != self.vocab_size
            or _cursor_config_identity(fields.config)
            != _cursor_config_identity(self.config)
            or fields.total_rows != len(self.rows)
        ):
            _fail("invalid-argument", "cursor identity/config/corpus mismatch")
        if fields.next_ordinal > len(self.rows):
            _fail("corrupt-data", "cursor next ordinal exceeds row count")
        self.next_ordinal = fields.next_ordinal

    def next_batch(self) -> Batch | None:
        if self.next_ordinal == len(self.rows):
            return None
        stop = min(len(self.rows), self.next_ordinal + self.config.batch_size)
        selected = [self.rows[self.order[index]] for index in range(self.next_ordinal, stop)]
        self.next_ordinal = stop
        width = self.config.sequence_length
        empty = Row((0,) * width, (0,) * width, (False,) * width)
        selected.extend(empty for _ in range(self.config.batch_size - len(selected)))
        return Batch(
            tuple(row.inputs for row in selected),
            tuple(row.targets for row in selected),
            tuple(row.loss_mask for row in selected),
        )
