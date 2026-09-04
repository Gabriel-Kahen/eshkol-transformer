#!/usr/bin/env python3
"""Generate bounded T2 artifacts for delivered-runtime negative checks."""

import argparse
import hashlib
from pathlib import Path

from tests.t2.generate_fixture import DOCUMENTS, SPECIALS, fixture_bytes
from tests.t2.reference import CHECKSUM_DOMAIN, V1_HEADER, encode_artifact, train


def replace_once(source: bytes, old: bytes, new: bytes) -> bytes:
    if source.count(old) != 1:
        raise AssertionError(f"expected exactly one occurrence of {old!r}")
    return source.replace(old, new, 1)


def resign(source: bytes) -> bytes:
    start = source.rfind(b"checksum\t")
    body = source[:start]
    digest = hashlib.sha256(CHECKSUM_DOMAIN + body).hexdigest().encode()
    return body + b"checksum\t" + digest + b"\n"


def artifact_with_payload(payload: bytes) -> bytes:
    body = V1_HEADER + f"payload-bytes\t{len(payload)}\n".encode() + payload
    return resign(body + b"checksum-algorithm\tsha256\nchecksum\t" + b"0" * 64 + b"\n")


def payload_from(source: bytes) -> bytes:
    marker = b"payload-bytes\t"
    line_start = source.index(marker)
    payload_start = source.index(b"\n", line_start) + 1
    payload_end = source.index(b"checksum-algorithm\t", payload_start)
    return source[payload_start:payload_end]


def mutate_field(
    source: bytes, label: bytes, field_index: int, value: bytes, occurrence: int = 0
) -> bytes:
    lines = source.splitlines(keepends=True)
    seen = 0
    for index, line in enumerate(lines):
        body = line[:-1] if line.endswith(b"\n") else line
        fields = body.split(b"\t")
        if fields[0] != label:
            continue
        if seen == occurrence:
            if field_index >= len(fields):
                raise AssertionError(f"field {field_index} missing from {label!r}")
            fields[field_index] = value
            lines[index] = b"\t".join(fields) + (b"\n" if line.endswith(b"\n") else b"")
            return b"".join(lines)
        seen += 1
    raise AssertionError(f"record {label!r}[{occurrence}] not found")


def record_index(lines: list[bytes], label: bytes, occurrence: int = 0) -> int:
    seen = 0
    for index, line in enumerate(lines):
        if line.split(b"\t", 1)[0] == label:
            if seen == occurrence:
                return index
            seen += 1
    raise AssertionError(f"record {label!r}[{occurrence}] not found")


def remove_record(source: bytes, label: bytes, occurrence: int = 0) -> bytes:
    lines = source.splitlines(keepends=True)
    del lines[record_index(lines, label, occurrence)]
    return b"".join(lines)


def duplicate_record(source: bytes, label: bytes, occurrence: int = 0) -> bytes:
    lines = source.splitlines(keepends=True)
    index = record_index(lines, label, occurrence)
    lines.insert(index + 1, lines[index])
    return b"".join(lines)


def wrong_arity_record(source: bytes, label: bytes, occurrence: int = 0) -> bytes:
    lines = source.splitlines(keepends=True)
    index = record_index(lines, label, occurrence)
    line = lines[index]
    lines[index] = line[:-1] + b"\textra\n"
    return b"".join(lines)


def short_arity_record(source: bytes, label: bytes, occurrence: int = 0) -> bytes:
    lines = source.splitlines(keepends=True)
    index = record_index(lines, label, occurrence)
    fields = lines[index][:-1].split(b"\t")
    if len(fields) < 2:
        raise AssertionError(f"record {label!r} has no removable field")
    lines[index] = b"\t".join(fields[:-1]) + b"\n"
    return b"".join(lines)


def unknown_label_record(source: bytes, label: bytes, occurrence: int = 0) -> bytes:
    return mutate_field(
        source, label, 0, b"unknown-" + label, occurrence=occurrence
    )


def swap_records(
    source: bytes,
    first_label: bytes,
    second_label: bytes,
    first_occurrence: int = 0,
    second_occurrence: int = 0,
) -> bytes:
    lines = source.splitlines(keepends=True)
    first = record_index(lines, first_label, first_occurrence)
    second = record_index(lines, second_label, second_occurrence)
    lines[first], lines[second] = lines[second], lines[first]
    return b"".join(lines)


def payload_artifact(source: bytes) -> bytes:
    return artifact_with_payload(source)


def model_payload(
    merges: list[tuple[int, int, int, int]],
    specials: list[tuple[int, bytes, bytes]],
    prefix: list[int],
    suffix: list[int],
) -> bytes:
    records = [
        b"kind\tbyte-bpe\n",
        b"normalization\tnone\n",
        b"utf8-policy\traw\n",
        b"byte-ids\t0\t255\n",
        f"merge-count\t{len(merges)}\n".encode(),
    ]
    records.extend(
        f"merge\t{rank}\t{result}\t{left}\t{right}\n".encode()
        for rank, result, left, right in merges
    )
    records.append(f"special-count\t{len(specials)}\n".encode())
    records.extend(
        b"special\t"
        + str(token_id).encode()
        + b"\t"
        + name
        + b"\t"
        + policy
        + b"\n"
        for token_id, name, policy in specials
    )
    records.append(f"prefix-count\t{len(prefix)}\n".encode())
    records.extend(
        f"prefix\t{index}\t{token_id}\n".encode()
        for index, token_id in enumerate(prefix)
    )
    records.append(f"suffix-count\t{len(suffix)}\n".encode())
    records.extend(
        f"suffix\t{index}\t{token_id}\n".encode()
        for index, token_id in enumerate(suffix)
    )
    return b"".join(records)


def overflow_payload() -> bytes:
    """Return a syntactically valid payload whose final token is 257 bytes."""
    records = [
        b"kind\tbyte-bpe\n",
        b"normalization\tnone\n",
        b"utf8-policy\traw\n",
        b"byte-ids\t0\t255\n",
        b"merge-count\t256\n",
        b"merge\t0\t256\t0\t1\n",
    ]
    for rank in range(1, 256):
        records.append(
            f"merge\t{rank}\t{256 + rank}\t{255 + rank}\t0\n".encode()
        )
    records.extend(
        (
            b"special-count\t0\n",
            b"prefix-count\t0\n",
            b"suffix-count\t0\n",
        )
    )
    return b"".join(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    valid = fixture_bytes()
    base_payload = payload_from(valid)
    fixtures = {
        "truncated.tsv": valid[:-1],
        "bad-checksum.tsv": replace_once(
            valid, b"merge\t0\t256", b"merge\t0\t257"
        ),
        "algorithm.tsv": resign(
            replace_once(
                valid,
                b"checksum-algorithm\tsha256",
                b"checksum-algorithm\tsha512",
            )
        ),
        "oversized.tsv": b"x" * 1_048_577,
        "forward-reference.tsv": resign(
            replace_once(
                replace_once(valid, b"payload-bytes\t295", b"payload-bytes\t296"),
                b"merge\t0\t256\t97\t110",
                b"merge\t0\t256\t97\t256",
            )
        ),
        "rank-gap.tsv": resign(
            replace_once(valid, b"merge\t0\t256", b"merge\t1\t256")
        ),
        "merge-count-mismatch.tsv": resign(
            replace_once(valid, b"merge-count\t4", b"merge-count\t5")
        ),
        "special-id-collision.tsv": resign(
            replace_once(valid, b"special\t260\tbos", b"special\t259\tbos")
        ),
        "noncanonical-version.tsv": resign(
            replace_once(valid, b"version\t1\t0", b"version\t01\t0")
        ),
        "alternate-same-vocab.tsv": encode_artifact(
            train(DOCUMENTS, 8, 2, "strict", SPECIALS, ("bos",), ("eos",))
        ),
        # Compiled-parser matrix. Mutations after the checksum boundary are
        # intentionally not resigned; semantic/header mutations are resigned.
        "trailing-byte.tsv": valid + b"x",
        "extra-lf.tsv": valid + b"\n",
        "duplicate-merge-pair.tsv": resign(
            replace_once(
                valid, b"merge\t1\t257\t98\t256", b"merge\t1\t257\t97\t110"
            )
        ),
        "out-of-order-special.tsv": resign(
            replace_once(
                valid,
                b"special\t260\tbos\tomit\nspecial\t261\teos\tomit\n",
                b"special\t261\teos\tomit\nspecial\t260\tbos\tomit\n",
            )
        ),
        "out-of-order-prefix.tsv": resign(
            replace_once(valid, b"prefix\t0\t260", b"prefix\t1\t260")
        ),
        "out-of-order-suffix.tsv": resign(
            replace_once(valid, b"suffix\t0\t261", b"suffix\t1\t261")
        ),
        "required-feature.tsv": resign(
            replace_once(valid, b"required-features\t0", b"required-features\t1")
        ),
        "unsupported-major.tsv": resign(
            replace_once(valid, b"version\t1\t0", b"version\t2\t0")
        ),
        "unsupported-minor.tsv": resign(
            replace_once(valid, b"version\t1\t0", b"version\t1\t1")
        ),
        "altered-limit-file-bytes.tsv": resign(
            replace_once(
                valid, b"limit-file-bytes\t1048576", b"limit-file-bytes\t1048575"
            )
        ),
        "altered-limit-merges.tsv": resign(
            replace_once(valid, b"limit-merges\t256", b"limit-merges\t255")
        ),
        "altered-limit-token-bytes.tsv": resign(
            replace_once(
                valid, b"limit-token-bytes\t256", b"limit-token-bytes\t255"
            )
        ),
        "altered-limit-specials.tsv": resign(
            replace_once(
                valid, b"limit-specials\t4096", b"limit-specials\t4095"
            )
        ),
        "altered-limit-name-bytes.tsv": resign(
            replace_once(
                valid, b"limit-name-bytes\t64", b"limit-name-bytes\t63"
            )
        ),
        "altered-limit-prefix.tsv": resign(
            replace_once(valid, b"limit-prefix\t4096", b"limit-prefix\t4095")
        ),
        "altered-limit-suffix.tsv": resign(
            replace_once(valid, b"limit-suffix\t4096", b"limit-suffix\t4095")
        ),
        "altered-payload-size-short.tsv": resign(
            replace_once(valid, b"payload-bytes\t295", b"payload-bytes\t294")
        ),
        "altered-payload-size-long.tsv": resign(
            replace_once(valid, b"payload-bytes\t295", b"payload-bytes\t296")
        ),
        "malformed-name.tsv": resign(
            replace_once(
                valid, b"special\t260\tbos\tomit", b"special\t260\tBos\tomit"
            )
        ),
        "malformed-special-policy.tsv": resign(
            replace_once(
                valid, b"special\t260\tbos\tomit", b"special\t260\tbos\toops"
            )
        ),
        "malformed-utf8-policy.tsv": resign(
            replace_once(valid, b"utf8-policy\traw", b"utf8-policy\tbad")
        ),
        "learned-token-overflow.tsv": artifact_with_payload(overflow_payload()),
    }

    # First-record routing and canonical line grammar. A malformed first format
    # record intentionally falls through to the accepted T1 parser, which must
    # still reject it through the public tokenizer-load error surface.
    format_line = b"format\teshkol-bpe-tokenizer\n"
    fixtures.update(
        {
            "missing-format.tsv": b"",
            "wrong-format.tsv": resign(
                mutate_field(valid, b"format", 1, b"eshkol-bpe-tokenizer-x")
            ),
            "empty-format-value.tsv": resign(
                mutate_field(valid, b"format", 1, b"")
            ),
            "wrong-arity-format.tsv": resign(
                replace_once(
                    valid,
                    format_line,
                    b"format\teshkol-bpe-tokenizer\textra\textra\n",
                )
            ),
            "bom-before-format.tsv": b"\xef\xbb\xbf" + valid,
            "cr-header.tsv": resign(
                replace_once(valid, b"version\t1\t0\n", b"version\t1\t0\r\n")
            ),
            "non-ascii-header.tsv": resign(
                replace_once(valid, b"version", b"versi\xffn")
            ),
            "trailing-space-header.tsv": resign(
                replace_once(valid, b"version\t1\t0\n", b"version\t1\t0 \n")
            ),
            "blank-header-record.tsv": resign(
                replace_once(valid, format_line, format_line + b"\n")
            ),
            "overlong-header-record.tsv": resign(
                replace_once(
                    valid,
                    b"version\t1\t0\n",
                    b"version\t" + b"1" * 250 + b"\n",
                )
            ),
            "bom-payload.tsv": payload_artifact(b"\xef\xbb\xbf" + base_payload),
            "cr-payload.tsv": payload_artifact(
                replace_once(base_payload, b"kind\tbyte-bpe\n", b"kind\tbyte-bpe\r\n")
            ),
            "non-ascii-payload.tsv": payload_artifact(
                replace_once(base_payload, b"normalization", b"normalizati\xffn")
            ),
            "trailing-space-payload.tsv": payload_artifact(
                replace_once(base_payload, b"kind\tbyte-bpe\n", b"kind\tbyte-bpe \n")
            ),
            "blank-payload-record.tsv": payload_artifact(b"\n" + base_payload),
            "overlong-payload-record.tsv": payload_artifact(
                replace_once(
                    base_payload,
                    b"kind\tbyte-bpe\n",
                    b"kind\t" + b"b" * 252 + b"\n",
                )
            ),
        }
    )

    # Envelope records: every frozen record has missing, duplicate, and
    # wrong-arity coverage. Reorder and unknown-record representatives cover
    # both the prefix discriminator and the exact BPE envelope reader.
    envelope_labels = (
        b"format",
        b"version",
        b"required-features",
        b"limit-file-bytes",
        b"limit-merges",
        b"limit-token-bytes",
        b"limit-specials",
        b"limit-name-bytes",
        b"limit-prefix",
        b"limit-suffix",
        b"payload-bytes",
        b"checksum-algorithm",
    )
    for label in envelope_labels:
        stem = label.decode("ascii")
        fixtures[f"missing-envelope-{stem}.tsv"] = resign(
            remove_record(valid, label)
        )
        fixtures[f"duplicate-envelope-{stem}.tsv"] = resign(
            duplicate_record(valid, label)
        )
        fixtures[f"wrong-arity-envelope-{stem}.tsv"] = resign(
            wrong_arity_record(valid, label)
        )
        fixtures[f"short-arity-envelope-{stem}.tsv"] = resign(
            short_arity_record(valid, label)
        )
        fixtures[f"unknown-label-envelope-{stem}.tsv"] = resign(
            unknown_label_record(valid, label)
        )
    checksum_start = valid.rfind(b"checksum\t")
    checksum_line = valid[checksum_start:]
    fixtures["missing-envelope-checksum.tsv"] = valid[:checksum_start]
    fixtures["duplicate-envelope-checksum.tsv"] = valid + checksum_line
    fixtures["wrong-arity-envelope-checksum.tsv"] = (
        valid[:checksum_start] + checksum_line[:-1] + b"\textra\n"
    )
    fixtures["short-arity-envelope-checksum.tsv"] = (
        valid[:checksum_start] + b"checksum\n"
    )
    fixtures["unknown-label-envelope-checksum.tsv"] = (
        valid[:checksum_start]
        + replace_once(checksum_line, b"checksum\t", b"unknown-checksum\t")
    )
    fixtures["unknown-envelope-record.tsv"] = resign(
        replace_once(
            valid,
            format_line,
            format_line + b"unknown-envelope\tvalue\n",
        )
    )
    fixtures["reordered-envelope-format-version.tsv"] = resign(
        swap_records(valid, b"format", b"required-features")
    )
    fixtures["reordered-envelope-version-features.tsv"] = resign(
        swap_records(valid, b"version", b"limit-file-bytes")
    )
    fixtures["reordered-envelope-limits.tsv"] = resign(
        swap_records(valid, b"limit-merges", b"limit-specials")
    )
    fixtures["reordered-envelope-payload-size.tsv"] = resign(
        swap_records(valid, b"limit-prefix", b"payload-bytes")
    )
    fixtures["reordered-envelope-checksum.tsv"] = swap_records(
        valid, b"checksum-algorithm", b"checksum"
    )
    for first, second in zip(envelope_labels, envelope_labels[1:]):
        fixtures[
            f"reordered-envelope-{first.decode('ascii')}-before-"
            f"{second.decode('ascii')}.tsv"
        ] = resign(swap_records(valid, first, second))

    # Payload records receive the same structural matrix. Every artifact is
    # rebuilt with its actual payload length and a valid checksum so the
    # production parser reaches the intended payload invariant.
    payload_labels = (
        b"kind",
        b"normalization",
        b"utf8-policy",
        b"byte-ids",
        b"merge-count",
        b"merge",
        b"special-count",
        b"special",
        b"prefix-count",
        b"prefix",
        b"suffix-count",
        b"suffix",
    )
    for label in payload_labels:
        stem = label.decode("ascii")
        fixtures[f"missing-payload-{stem}.tsv"] = payload_artifact(
            remove_record(base_payload, label)
        )
        fixtures[f"duplicate-payload-{stem}.tsv"] = payload_artifact(
            duplicate_record(base_payload, label)
        )
        fixtures[f"wrong-arity-payload-{stem}.tsv"] = payload_artifact(
            wrong_arity_record(base_payload, label)
        )
        fixtures[f"short-arity-payload-{stem}.tsv"] = payload_artifact(
            short_arity_record(base_payload, label)
        )
        fixtures[f"unknown-label-payload-{stem}.tsv"] = payload_artifact(
            unknown_label_record(base_payload, label)
        )
    fixtures["unknown-payload-record.tsv"] = payload_artifact(
        b"unknown-payload\tvalue\n" + base_payload
    )
    fixtures["trailing-unknown-payload-record.tsv"] = payload_artifact(
        base_payload + b"unknown-payload\tvalue\n"
    )
    fixtures["reordered-payload-kind-normalization.tsv"] = payload_artifact(
        swap_records(base_payload, b"kind", b"utf8-policy")
    )
    fixtures["reordered-payload-merge-section.tsv"] = payload_artifact(
        swap_records(base_payload, b"merge-count", b"special-count")
    )
    fixtures["reordered-payload-special-section.tsv"] = payload_artifact(
        swap_records(base_payload, b"special-count", b"prefix-count")
    )
    fixtures["reordered-payload-prefix-section.tsv"] = payload_artifact(
        swap_records(base_payload, b"prefix-count", b"suffix-count")
    )
    fixtures["reordered-payload-suffix-section.tsv"] = payload_artifact(
        swap_records(base_payload, b"suffix-count", b"special-count")
    )
    for first, second in zip(payload_labels, payload_labels[1:]):
        fixtures[
            f"reordered-payload-{first.decode('ascii')}-before-"
            f"{second.decode('ascii')}.tsv"
        ] = payload_artifact(swap_records(base_payload, first, second))

    # Canonical unsigned integer grammar is pinned independently for every
    # envelope/payload numeric field family, outside the prior version-only
    # representative. Counts also receive exact ceiling and one-over models.
    integer_variants = {
        "empty": b"",
        "negative": b"-1",
        "noncanonical": b"00",
        "leading-plus": b"+1",
        "trailing-space": b"1 ",
        "greater-than-i64": b"9223372036854775808",
    }
    envelope_integer_fields = (
        ("version-major", b"version", 1),
        ("version-minor", b"version", 2),
        ("required-features", b"required-features", 1),
        ("payload-bytes", b"payload-bytes", 1),
    )
    for stem, label, field_index in envelope_integer_fields:
        for variant, value in integer_variants.items():
            fixtures[f"integer-{stem}-{variant}.tsv"] = resign(
                mutate_field(valid, label, field_index, value)
            )
    payload_integer_fields = (
        ("merge-count", b"merge-count", 1),
        ("merge-rank", b"merge", 1),
        ("merge-result", b"merge", 2),
        ("merge-left", b"merge", 3),
        ("merge-right", b"merge", 4),
        ("special-count", b"special-count", 1),
        ("special-id", b"special", 1),
        ("prefix-count", b"prefix-count", 1),
        ("prefix-index", b"prefix", 1),
        ("prefix-target", b"prefix", 2),
        ("suffix-count", b"suffix-count", 1),
        ("suffix-index", b"suffix", 1),
        ("suffix-target", b"suffix", 2),
    )
    for stem, label, field_index in payload_integer_fields:
        for variant, value in integer_variants.items():
            fixtures[f"integer-{stem}-{variant}.tsv"] = payload_artifact(
                mutate_field(base_payload, label, field_index, value)
            )
    for stem, label, value in (
        ("merge", b"merge-count", b"257"),
        ("special", b"special-count", b"4097"),
        ("prefix", b"prefix-count", b"4097"),
        ("suffix", b"suffix-count", b"4097"),
    ):
        fixtures[f"count-{stem}-one-over.tsv"] = payload_artifact(
            mutate_field(base_payload, label, 1, value)
        )
    fixtures["payload-size-one-over.tsv"] = resign(
        mutate_field(valid, b"payload-bytes", 1, b"1048577")
    )
    for stem, label, value in (
        ("special", b"special-count", b"4"),
        ("prefix", b"prefix-count", b"2"),
        ("suffix", b"suffix-count", b"2"),
    ):
        fixtures[f"count-{stem}-records-missing.tsv"] = payload_artifact(
            mutate_field(base_payload, label, 1, value)
        )

    # Checksum grammar, semantic identifier continuity, special-name grammar,
    # and insertion target/index invariants.
    checksum_value = checksum_line[len(b"checksum\t") : -1]
    fixtures["checksum-uppercase.tsv"] = (
        valid[:checksum_start] + b"checksum\t" + checksum_value.upper() + b"\n"
    )
    fixtures["checksum-short.tsv"] = (
        valid[:checksum_start] + b"checksum\t" + checksum_value[:-1] + b"\n"
    )
    fixtures["checksum-long.tsv"] = (
        valid[:checksum_start] + b"checksum\t" + checksum_value + b"0\n"
    )
    fixtures["checksum-nonhex.tsv"] = (
        valid[:checksum_start] + b"checksum\tg" + checksum_value[1:] + b"\n"
    )
    fixtures["checksum-empty.tsv"] = valid[:checksum_start] + b"checksum\t\n"
    fixtures["merge-result-gap.tsv"] = payload_artifact(
        mutate_field(base_payload, b"merge", 2, b"258", occurrence=1)
    )
    fixtures["merge-left-self-reference.tsv"] = payload_artifact(
        mutate_field(base_payload, b"merge", 3, b"256")
    )
    fixtures["merge-right-self-reference.tsv"] = payload_artifact(
        mutate_field(base_payload, b"merge", 4, b"256")
    )
    fixtures["merge-left-forward-reference.tsv"] = payload_artifact(
        mutate_field(base_payload, b"merge", 3, b"259", occurrence=1)
    )
    fixtures["merge-right-forward-reference.tsv"] = payload_artifact(
        mutate_field(base_payload, b"merge", 4, b"259", occurrence=1)
    )
    fixtures["duplicate-special-name.tsv"] = payload_artifact(
        mutate_field(base_payload, b"special", 2, b"bos", occurrence=1)
    )
    for stem, name in (
        ("empty", b""),
        ("65-byte", b"a" * 65),
        ("starts-digit", b"1name"),
        ("starts-hyphen", b"-name"),
        ("contains-slash", b"bad/name"),
        ("contains-space", b"bad name"),
        ("contains-nul", b"bad\x00name"),
        ("non-ascii", b"bad\xffname"),
    ):
        fixtures[f"invalid-special-name-{stem}.tsv"] = payload_artifact(
            mutate_field(base_payload, b"special", 2, name)
        )
    fixtures["duplicate-prefix-index.tsv"] = payload_artifact(
        replace_once(
            mutate_field(base_payload, b"prefix-count", 1, b"2"),
            b"prefix\t0\t260\n",
            b"prefix\t0\t260\nprefix\t0\t260\n",
        )
    )
    fixtures["duplicate-suffix-index.tsv"] = payload_artifact(
        replace_once(
            mutate_field(base_payload, b"suffix-count", 1, b"2"),
            b"suffix\t0\t261\n",
            b"suffix\t0\t261\nsuffix\t0\t261\n",
        )
    )
    fixtures["gapped-prefix-index.tsv"] = payload_artifact(
        replace_once(
            mutate_field(base_payload, b"prefix-count", 1, b"2"),
            b"prefix\t0\t260\n",
            b"prefix\t0\t260\nprefix\t2\t260\n",
        )
    )
    fixtures["gapped-suffix-index.tsv"] = payload_artifact(
        replace_once(
            mutate_field(base_payload, b"suffix-count", 1, b"2"),
            b"suffix\t0\t261\n",
            b"suffix\t0\t261\nsuffix\t2\t261\n",
        )
    )
    fixtures["unknown-prefix-target.tsv"] = payload_artifact(
        mutate_field(base_payload, b"prefix", 2, b"263")
    )
    fixtures["error-prefix-target.tsv"] = payload_artifact(
        mutate_field(base_payload, b"prefix", 2, b"262")
    )
    fixtures["unknown-suffix-target.tsv"] = payload_artifact(
        mutate_field(base_payload, b"suffix", 2, b"263")
    )
    fixtures["error-suffix-target.tsv"] = payload_artifact(
        mutate_field(base_payload, b"suffix", 2, b"262")
    )

    valid_max_merges = [
        (rank, 256 + rank, rank, (rank + 1) % 256) for rank in range(256)
    ]
    valid_max_specials = [
        (256 + index, f"s{index}".encode(), b"omit") for index in range(4096)
    ]
    fixtures["valid-max-merges.tsv"] = artifact_with_payload(
        model_payload(valid_max_merges, [], [], [])
    )
    fixtures["valid-max-specials.tsv"] = artifact_with_payload(
        model_payload([], valid_max_specials, [], [])
    )
    fixtures["valid-max-prefix.tsv"] = artifact_with_payload(
        model_payload([], [(256, b"omit", b"omit")], [256] * 4096, [])
    )
    fixtures["valid-max-suffix.tsv"] = artifact_with_payload(
        model_payload([], [(256, b"omit", b"omit")], [], [256] * 4096)
    )
    for name, payload in fixtures.items():
        (arguments.output_directory / name).write_bytes(payload)
    print(f"T2 ADVERSARIAL FIXTURES PASS: {len(fixtures)} deterministic artifacts")


if __name__ == "__main__":
    main()
