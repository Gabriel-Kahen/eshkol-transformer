"""Independent development-only oracle for deterministic T2 byte BPE.

Production training, tokenization, streaming, and artifact handling remain
Eshkol-authored.  This module deliberately uses only the Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import re
import struct
from typing import NoReturn, Sequence


MAX_FILE = 1_048_576
MAX_MERGES = 256
MAX_TOKEN_BYTES = 256
MAX_SPECIALS = 4_096
MAX_NAME_BYTES = 64
MAX_INSERTIONS = 4_096
MAX_TRAIN_BYTES = 65_536
MAX_DOCUMENTS = 4_096
MAX_CHUNKS = 4_096
MAX_STREAM_BYTES = 65_536
MAX_STREAM_TOKEN_IDS = MAX_STREAM_BYTES + 2 * MAX_INSERTIONS
MAX_LINE = 256
MAX_I64 = (1 << 63) - 1
NAME_RE = re.compile(rb"[a-z][a-z0-9._-]{0,63}\Z")
DIGEST_RE = re.compile(rb"[0-9a-f]{64}\Z")
CHECKSUM_DOMAIN = b"eshkol-bpe-tokenizer-checksum-v1\n"
IDENTITY_DOMAIN = b"sha256:eshkol-bpe-tokenizer-v1\n"
V1_HEADER = (
    b"format\teshkol-bpe-tokenizer\n"
    b"version\t1\t0\n"
    b"required-features\t0\n"
    b"limit-file-bytes\t1048576\n"
    b"limit-merges\t256\n"
    b"limit-token-bytes\t256\n"
    b"limit-specials\t4096\n"
    b"limit-name-bytes\t64\n"
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
class Merge:
    rank: int
    result_id: int
    left_id: int
    right_id: int


@dataclass(frozen=True, slots=True)
class Special:
    token_id: int
    name: str
    decode: str


@dataclass(frozen=True, slots=True)
class Core:
    utf8_policy: str = "raw"
    merges: tuple[Merge, ...] = ()
    specials: tuple[Special, ...] = ()
    prefix: tuple[int, ...] = ()
    suffix: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Parsed:
    core: Core
    fingerprint: str


def _argument(condition: bool, message: str) -> None:
    if not condition:
        fail(message, "invalid-argument")


def _exact_int(value: object, where: str, low: int, high: int) -> int:
    _argument(not isinstance(value, bool) and isinstance(value, int), f"{where}: expected exact integer")
    result = int(value)
    _argument(low <= result <= high, f"{where}: out of range")
    return result


def _u64(raw: bytes, where: str, maximum: int = MAX_I64) -> int:
    if not raw or not raw.isdigit() or (len(raw) > 1 and raw.startswith(b"0")):
        fail(f"{where}: noncanonical unsigned decimal")
    value = int(raw)
    if value > maximum:
        fail(f"{where}: overflow")
    return value


def _name(value: str, where: str, category: str = "corrupt-data") -> bytes:
    if not isinstance(value, str):
        fail(f"{where}: expected string", category)
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        fail(f"{where}: non-ASCII name", category)
    if not NAME_RE.fullmatch(encoded):
        fail(f"{where}: invalid name", category)
    return encoded


def validate(core: Core) -> Core:
    if not isinstance(core, Core):
        fail("core: wrong type")
    if core.utf8_policy not in {"raw", "strict"}:
        fail("core: invalid UTF-8 policy")
    if len(core.merges) > MAX_MERGES:
        fail("core: too many merges")
    if len(core.specials) > MAX_SPECIALS:
        fail("core: too many specials")
    if len(core.prefix) > MAX_INSERTIONS or len(core.suffix) > MAX_INSERTIONS:
        fail("core: too many insertions")

    spellings: list[bytes] = [bytes((value,)) for value in range(256)]
    pairs: set[tuple[int, int]] = set()
    merges: list[Merge] = []
    for rank, merge in enumerate(core.merges):
        if not isinstance(merge, Merge):
            fail("core: malformed merge")
        result_id = 256 + rank
        if merge.rank != rank or merge.result_id != result_id:
            fail("core: merge rank/result gap")
        if (isinstance(merge.left_id, bool) or isinstance(merge.right_id, bool)
                or not isinstance(merge.left_id, int) or not isinstance(merge.right_id, int)
                or not 0 <= merge.left_id < result_id or not 0 <= merge.right_id < result_id):
            fail("core: merge operand is not an earlier token")
        pair = (merge.left_id, merge.right_id)
        if pair in pairs:
            fail("core: duplicate merge pair")
        pairs.add(pair)
        spelling = spellings[merge.left_id] + spellings[merge.right_id]
        if len(spelling) > MAX_TOKEN_BYTES:
            fail("core: learned token exceeds byte limit")
        spellings.append(spelling)
        merges.append(merge)

    special_base = 256 + len(merges)
    names: set[str] = set()
    specials: list[Special] = []
    for index, special in enumerate(core.specials):
        if not isinstance(special, Special) or special.token_id != special_base + index:
            fail("core: special IDs must be contiguous after merges")
        _name(special.name, "core.special.name")
        if special.name in names:
            fail("core: duplicate special name")
        names.add(special.name)
        if special.decode not in {"omit", "error"}:
            fail("core: invalid special decode policy")
        specials.append(special)
    by_id = {item.token_id: item for item in specials}
    for label, values in (("prefix", core.prefix), ("suffix", core.suffix)):
        for token_id in values:
            if (isinstance(token_id, bool) or not isinstance(token_id, int)
                    or token_id not in by_id or by_id[token_id].decode != "omit"):
                fail(f"core: {label} must reference an omit special")
    return Core(core.utf8_policy, tuple(merges), tuple(specials), core.prefix, core.suffix)


def _chunk_bytes(chunk: object, where: str) -> bytes:
    if isinstance(chunk, bytes):
        return chunk
    if isinstance(chunk, str):
        try:
            return chunk.encode("utf-8")
        except UnicodeEncodeError:
            fail(f"{where}: string is not encodable as UTF-8", "invalid-argument")
    fail(f"{where}: chunk must be string or bytes", "invalid-argument")


def _validate_utf8(raw: bytes, where: str) -> None:
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        fail(f"{where}: malformed UTF-8", "invalid-argument")


def _replace_pair(tokens: Sequence[int], left: int, right: int, result: int) -> list[int]:
    output: list[int] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and tokens[index] == left and tokens[index + 1] == right:
            output.append(result)
            index += 2
        else:
            output.append(tokens[index])
            index += 1
    return output


def train(
    documents: object,
    maximum_merges: object,
    minimum_frequency: object,
    utf8_policy: object = "raw",
    special_specs: object = (),
    prefix_names: object = (),
    suffix_names: object = (),
) -> Core:
    """Train with document-order and intra-document chunk-partition invariance."""
    maximum = _exact_int(maximum_merges, "maximum-merges", 0, MAX_MERGES)
    minimum = _exact_int(minimum_frequency, "minimum-frequency", 1, MAX_I64)
    _argument(utf8_policy in {"raw", "strict"}, "utf8-policy: expected raw or strict")
    _argument(isinstance(documents, (list, tuple)), "documents: expected proper sequence")
    _argument(len(documents) <= MAX_DOCUMENTS, "documents: count limit exceeded")
    encoded_documents: list[list[int]] = []
    aggregate_bytes = 0
    chunk_count = 0
    for document_index, document in enumerate(documents):
        _argument(isinstance(document, (list, tuple)), f"document {document_index}: expected proper sequence")
        chunk_count += len(document)
        _argument(chunk_count <= MAX_CHUNKS, "chunks: count limit exceeded")
        document_bytes = bytearray()
        for chunk in document:
            encoded = _chunk_bytes(chunk, f"document {document_index}")
            _argument(len(encoded) <= MAX_TRAIN_BYTES - aggregate_bytes,
                      "training bytes: limit exceeded")
            aggregate_bytes += len(encoded)
            document_bytes.extend(encoded)
        raw = bytes(document_bytes)
        if utf8_policy == "strict":
            _validate_utf8(raw, f"document {document_index}")
        encoded_documents.append(list(raw))

    _argument(isinstance(special_specs, (list, tuple)), "special-specs: expected proper sequence")
    _argument(len(special_specs) <= MAX_SPECIALS, "special-specs: count limit exceeded")
    specs: list[tuple[str, str]] = []
    for item in special_specs:
        _argument(isinstance(item, (list, tuple)) and len(item) == 2, "special-spec: malformed")
        name, policy = item
        _name(name, "special-spec.name", "invalid-argument")
        _argument(policy in {"omit", "error"}, "special-spec: invalid decode policy")
        specs.append((name, policy))
    _argument([name for name, _ in specs] == sorted({name for name, _ in specs}),
              "special-specs: names must be unique and sorted")
    for label, values in (("prefix", prefix_names), ("suffix", suffix_names)):
        _argument(isinstance(values, (list, tuple)), f"{label}: expected proper sequence")
        _argument(len(values) <= MAX_INSERTIONS, f"{label}: count limit exceeded")

    spellings: list[bytes] = [bytes((value,)) for value in range(256)]
    merges: list[Merge] = []
    for rank in range(maximum):
        counts: dict[tuple[int, int], int] = {}
        for document in encoded_documents:
            for pair in zip(document, document[1:]):
                if len(spellings[pair[0]]) + len(spellings[pair[1]]) > MAX_TOKEN_BYTES:
                    continue
                prior = counts.get(pair, 0)
                if prior == MAX_I64:
                    fail("pair count overflow", "internal")
                counts[pair] = prior + 1
        eligible = [(pair, count) for pair, count in counts.items() if count >= minimum]
        if not eligible:
            break
        pair, _ = min(eligible, key=lambda item: (-item[1], item[0][0], item[0][1]))
        result = 256 + rank
        merge = Merge(rank, result, pair[0], pair[1])
        merges.append(merge)
        spellings.append(spellings[pair[0]] + spellings[pair[1]])
        encoded_documents = [
            _replace_pair(document, pair[0], pair[1], result)
            for document in encoded_documents
        ]

    special_base = 256 + len(merges)
    specials = tuple(
        Special(special_base + index, name, policy)
        for index, (name, policy) in enumerate(specs)
    )
    by_name = {item.name: item for item in specials}

    def insertion(label: str, values: object) -> tuple[int, ...]:
        result: list[int] = []
        for name in values:  # type: ignore[union-attr]
            _argument(isinstance(name, str) and name in by_name, f"{label}: unknown special name")
            special = by_name[name]
            _argument(special.decode == "omit", f"{label}: special must use omit policy")
            result.append(special.token_id)
        return tuple(result)

    return validate(Core(str(utf8_policy), tuple(merges), specials,
                         insertion("prefix", prefix_names), insertion("suffix", suffix_names)))


def spellings(core: Core) -> tuple[bytes, ...]:
    core = validate(core)
    result = [bytes((value,)) for value in range(256)]
    for merge in core.merges:
        result.append(result[merge.left_id] + result[merge.right_id])
    return tuple(result)


def payload(core: Core) -> bytes:
    core = validate(core)
    lines = [
        b"kind\tbyte-bpe\n",
        b"normalization\tnone\n",
        f"utf8-policy\t{core.utf8_policy}\n".encode(),
        b"byte-ids\t0\t255\n",
        f"merge-count\t{len(core.merges)}\n".encode(),
    ]
    lines.extend(
        f"merge\t{x.rank}\t{x.result_id}\t{x.left_id}\t{x.right_id}\n".encode()
        for x in core.merges
    )
    lines.append(f"special-count\t{len(core.specials)}\n".encode())
    lines.extend(
        b"special\t" + str(x.token_id).encode() + b"\t"
        + _name(x.name, "special.name") + b"\t" + x.decode.encode() + b"\n"
        for x in core.specials
    )
    lines.append(f"prefix-count\t{len(core.prefix)}\n".encode())
    lines.extend(f"prefix\t{i}\t{x}\n".encode() for i, x in enumerate(core.prefix))
    lines.append(f"suffix-count\t{len(core.suffix)}\n".encode())
    lines.extend(f"suffix\t{i}\t{x}\n".encode() for i, x in enumerate(core.suffix))
    return b"".join(lines)


def encode_artifact(core: Core) -> bytes:
    body_payload = payload(core)
    if len(body_payload) > MAX_FILE:
        fail("artifact payload exceeds hard limit")
    body = V1_HEADER + f"payload-bytes\t{len(body_payload)}\n".encode() + body_payload
    checksum_input = body + b"checksum-algorithm\tsha256\n"
    digest = hashlib.sha256(CHECKSUM_DOMAIN + checksum_input).hexdigest().encode()
    result = checksum_input + b"checksum\t" + digest + b"\n"
    if len(result) > MAX_FILE:
        fail("artifact exceeds hard file limit")
    return result


def fingerprint(core: Core) -> str:
    digest = hashlib.sha256(IDENTITY_DOMAIN + V1_HEADER + payload(core)).hexdigest()
    return f"sha256:eshkol-bpe-tokenizer-v1:{digest}"


def encode_bytes(core: Core, source: object) -> tuple[int, ...]:
    core = validate(core)
    raw = _chunk_bytes(source, "encode input")
    _argument(len(raw) <= MAX_STREAM_BYTES, "encode input: byte limit exceeded")
    if core.utf8_policy == "strict":
        _validate_utf8(raw, "encode input")
    tokens = list(raw)
    for merge in core.merges:
        tokens = _replace_pair(tokens, merge.left_id, merge.right_id, merge.result_id)
    return core.prefix + tuple(tokens) + core.suffix


def decode_ids(core: Core, ids: object) -> bytes:
    core = validate(core)
    _argument(isinstance(ids, (list, tuple)), "decode IDs: expected proper sequence")
    _argument(len(ids) <= MAX_STREAM_TOKEN_IDS, "decode IDs: token limit exceeded")
    words = spellings(core)
    specials = {item.token_id: item for item in core.specials}
    output = bytearray()
    for token_id in ids:
        _argument(not isinstance(token_id, bool) and isinstance(token_id, int), "decode ID: expected exact integer")
        if 0 <= token_id < len(words):
            output.extend(words[token_id])
        elif token_id in specials:
            if specials[token_id].decode == "error":
                fail("special ID decode is forbidden", "invalid-argument")
        else:
            fail("decode ID out of vocabulary", "invalid-argument")
        _argument(len(output) <= MAX_STREAM_BYTES,
                  "decoded bytes: byte limit exceeded")
    result = bytes(output)
    if core.utf8_policy == "strict":
        _validate_utf8(result, "decoded bytes")
    return result


class _Reader:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.offset = 0

    def line(self, where: str) -> bytes:
        end = self.raw.find(b"\n", self.offset)
        if end < 0 or end - self.offset > MAX_LINE:
            fail(f"{where}: missing LF or overlong line")
        result = self.raw[self.offset:end]
        self.offset = end + 1
        return result


def _fields(reader: _Reader, name: bytes, count: int) -> tuple[bytes, ...]:
    fields = tuple(reader.line(name.decode()).split(b"\t"))
    if len(fields) != count or fields[0] != name:
        fail(f"{name.decode()}: malformed record")
    return fields


def _parse_payload(raw: bytes) -> Core:
    reader = _Reader(raw)
    if _fields(reader, b"kind", 2)[1] != b"byte-bpe":
        fail("kind: noncanonical")
    if _fields(reader, b"normalization", 2)[1] != b"none":
        fail("normalization: noncanonical")
    utf8 = _fields(reader, b"utf8-policy", 2)[1]
    if utf8 not in {b"raw", b"strict"}:
        fail("utf8-policy: noncanonical")
    if _fields(reader, b"byte-ids", 3)[1:] != (b"0", b"255"):
        fail("byte IDs: noncanonical")
    merge_count = _u64(_fields(reader, b"merge-count", 2)[1], "merge-count", MAX_MERGES)
    merges: list[Merge] = []
    for rank in range(merge_count):
        fields = _fields(reader, b"merge", 5)
        merges.append(Merge(
            _u64(fields[1], "merge.rank", MAX_MERGES),
            _u64(fields[2], "merge.result", 255 + MAX_MERGES),
            _u64(fields[3], "merge.left", 255 + MAX_MERGES),
            _u64(fields[4], "merge.right", 255 + MAX_MERGES),
        ))
    special_count = _u64(_fields(reader, b"special-count", 2)[1], "special-count", MAX_SPECIALS)
    specials: list[Special] = []
    for _ in range(special_count):
        fields = _fields(reader, b"special", 4)
        if not NAME_RE.fullmatch(fields[2]) or fields[3] not in {b"omit", b"error"}:
            fail("special: noncanonical")
        specials.append(Special(_u64(fields[1], "special.id", MAX_I64),
                                fields[2].decode(), fields[3].decode()))
    by_id = {item.token_id: item for item in specials}

    def insertion(label: bytes) -> tuple[int, ...]:
        count = _u64(_fields(reader, label + b"-count", 2)[1], "insertion count", MAX_INSERTIONS)
        result: list[int] = []
        for index in range(count):
            fields = _fields(reader, label, 3)
            if _u64(fields[1], "insertion index", MAX_INSERTIONS) != index:
                fail("insertion index: noncanonical")
            token_id = _u64(fields[2], "insertion ID", MAX_I64)
            if token_id not in by_id or by_id[token_id].decode != "omit":
                fail("insertion: invalid special reference")
            result.append(token_id)
        return tuple(result)

    prefix = insertion(b"prefix")
    suffix = insertion(b"suffix")
    if reader.offset != len(raw):
        fail("payload: trailing record")
    return validate(Core(utf8.decode(), tuple(merges), tuple(specials), prefix, suffix))


def decode_artifact(source: object, *, max_file: int = MAX_FILE, max_metadata: int = MAX_FILE) -> Parsed:
    if not isinstance(source, bytes):
        fail("artifact source must be bytes")
    _argument(not isinstance(max_file, bool) and isinstance(max_file, int) and max_file >= 1,
              "max-file: expected positive integer")
    _argument(not isinstance(max_metadata, bool) and isinstance(max_metadata, int) and max_metadata >= 1,
              "max-metadata: expected positive integer")
    if len(source) > min(max_file, MAX_FILE):
        fail("artifact exceeds effective file policy")
    try:
        source.decode("ascii")
    except UnicodeDecodeError:
        fail("artifact is not ASCII")
    if b"\r" in source or not source.endswith(b"\n"):
        fail("artifact newline is noncanonical")
    reader = _Reader(source)
    if _fields(reader, b"format", 2)[1] != b"eshkol-bpe-tokenizer":
        fail("format: bad identity")
    version = _fields(reader, b"version", 3)
    major = _u64(version[1], "version major")
    minor = _u64(version[2], "version minor")
    if major != 1 or minor != 0:
        fail("unsupported version", "version-mismatch")
    if _u64(_fields(reader, b"required-features", 2)[1], "required count", MAX_SPECIALS) != 0:
        fail("required feature is unsupported", "unsupported")
    for expected in V1_HEADER.splitlines()[3:]:
        if reader.line("limit") != expected:
            fail("limit record: noncanonical")
    header = source[:reader.offset]
    payload_size = _u64(_fields(reader, b"payload-bytes", 2)[1], "payload bytes", MAX_FILE)
    if payload_size > min(max_metadata, MAX_FILE):
        fail("payload exceeds effective metadata policy")
    payload_end = reader.offset + payload_size
    if payload_end > len(source):
        fail("payload: truncated")
    raw_payload = source[reader.offset:payload_end]
    reader.offset = payload_end
    if _fields(reader, b"checksum-algorithm", 2)[1] != b"sha256":
        fail("checksum algorithm: unsupported", "unsupported")
    checksum_input = source[:reader.offset]
    checksum = _fields(reader, b"checksum", 2)[1]
    if not DIGEST_RE.fullmatch(checksum):
        fail("checksum: malformed")
    if reader.offset != len(source):
        fail("artifact: trailing bytes")
    actual = hashlib.sha256(CHECKSUM_DOMAIN + checksum_input).hexdigest().encode()
    if not hmac.compare_digest(checksum, actual):
        fail("checksum mismatch")
    core = _parse_payload(raw_payload)
    if encode_artifact(core) != source:
        fail("artifact is not canonical")
    digest = hashlib.sha256(IDENTITY_DOMAIN + header + raw_payload).hexdigest()
    return Parsed(core, f"sha256:eshkol-bpe-tokenizer-v1:{digest}")


@dataclass(slots=True)
class _Stage:
    merge: Merge
    pending: int | None = None


class StreamEncoder:
    def __init__(self, core: Core) -> None:
        self.core = validate(core)
        self.stages = [_Stage(merge) for merge in self.core.merges]
        self.total_bytes = 0
        self.started = False
        self.finished = False
        self.failed = False
        self.utf8_pending = b""

    def _guard(self) -> None:
        if self.failed or self.finished:
            fail("encoder state is failed or finished", "invalid-argument")

    def _send(self, stage_index: int, token_id: int, output: list[int]) -> None:
        if stage_index == len(self.stages):
            output.append(token_id)
            return
        stage = self.stages[stage_index]
        if stage.pending is None:
            stage.pending = token_id
        elif stage.pending == stage.merge.left_id and token_id == stage.merge.right_id:
            stage.pending = None
            self._send(stage_index + 1, stage.merge.result_id, output)
        else:
            pending = stage.pending
            stage.pending = token_id
            self._send(stage_index + 1, pending, output)

    def push(self, chunk: object) -> tuple[int, ...]:
        self._guard()
        try:
            raw = _chunk_bytes(chunk, "stream encoder chunk")
            if self.total_bytes + len(raw) > MAX_STREAM_BYTES:
                fail("stream encoder byte limit exceeded", "invalid-argument")
            if self.core.utf8_policy == "strict":
                complete, self.utf8_pending = _strict_prefix(
                    self.utf8_pending + raw, final=False
                )
            else:
                complete = raw
            self.total_bytes += len(raw)
            output = list(self.core.prefix) if not self.started else []
            self.started = True
            for token_id in complete:
                self._send(0, token_id, output)
            return tuple(output)
        except Exception:
            self.failed = True
            raise

    def finish(self) -> tuple[int, ...]:
        self._guard()
        try:
            if self.core.utf8_policy == "strict":
                _strict_prefix(self.utf8_pending, final=True)
            output = list(self.core.prefix) if not self.started else []
            self.started = True
            for index, stage in enumerate(self.stages):
                if stage.pending is not None:
                    pending = stage.pending
                    stage.pending = None
                    self._send(index + 1, pending, output)
            output.extend(self.core.suffix)
            self.finished = True
            return tuple(output)
        except Exception:
            self.failed = True
            raise


def _strict_prefix(data: bytes, *, final: bool) -> tuple[bytes, bytes]:
    try:
        data.decode("utf-8", errors="strict")
        return data, b""
    except UnicodeDecodeError as error:
        incomplete = error.reason == "unexpected end of data" and error.end == len(data)
        if incomplete and not final:
            prefix, pending = data[:error.start], data[error.start:]
            prefix.decode("utf-8", errors="strict")
            return prefix, pending
        fail("stream bytes are malformed UTF-8", "invalid-argument")


class StreamDecoder:
    def __init__(self, core: Core) -> None:
        self.core = validate(core)
        self.words = spellings(self.core)
        self.specials = {item.token_id: item for item in self.core.specials}
        self.total_bytes = 0
        self.total_tokens = 0
        self.finished = False
        self.failed = False
        self.utf8_pending = b""

    def _guard(self) -> None:
        if self.failed or self.finished:
            fail("decoder state is failed or finished", "invalid-argument")

    def push(self, staging: object) -> bytes:
        self._guard()
        try:
            if not isinstance(staging, bytes) or len(staging) % 8:
                fail("decoder staging must be complete i64-le bytes", "invalid-argument")
            count = len(staging) // 8
            if self.total_tokens + count > MAX_STREAM_TOKEN_IDS:
                fail("stream decoder token limit exceeded", "invalid-argument")
            output = bytearray()
            for offset in range(0, len(staging), 8):
                token_id = struct.unpack_from("<q", staging, offset)[0]
                if 0 <= token_id < len(self.words):
                    spelling = self.words[token_id]
                    if self.total_bytes + len(output) + len(spelling) > MAX_STREAM_BYTES:
                        fail("stream decoder byte limit exceeded", "invalid-argument")
                    output.extend(spelling)
                elif token_id in self.specials:
                    if self.specials[token_id].decode == "error":
                        fail("special ID decode is forbidden", "invalid-argument")
                else:
                    fail("decode ID out of vocabulary", "invalid-argument")
            self.total_bytes += len(output)
            self.total_tokens += count
            raw = bytes(output)
            if self.core.utf8_policy == "strict":
                complete, self.utf8_pending = _strict_prefix(self.utf8_pending + raw, final=False)
                return complete
            return raw
        except Exception:
            self.failed = True
            raise

    def finish(self) -> bytes:
        self._guard()
        try:
            if self.core.utf8_policy == "strict":
                complete, pending = _strict_prefix(self.utf8_pending, final=True)
                if pending:
                    fail("stream decoder retained bytes", "internal")
            else:
                complete = b""
            self.finished = True
            return complete
        except Exception:
            self.failed = True
            raise


def pack_i64(ids: Sequence[int]) -> bytes:
    output = bytearray()
    for token_id in ids:
        _argument(not isinstance(token_id, bool) and isinstance(token_id, int)
                  and 0 <= token_id <= MAX_I64, "staging ID: out of signed-i64 range")
        output.extend(struct.pack("<q", token_id))
    return bytes(output)
