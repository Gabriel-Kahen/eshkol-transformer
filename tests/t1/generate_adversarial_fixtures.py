"""Generate inert T1 artifacts for delivered-runtime adversarial tests.

Python is used only to prepare deterministic bytes.  The authoritative parser,
validation, error-category, and tokenizer checks run through public Eshkol AOT.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from tests.t1.reference import (
    CHECKSUM_DOMAIN,
    Identity,
    Special,
    encode_artifact,
)


RAW = Identity(
    "raw",
    (
        Special(256, "bos", "omit"),
        Special(257, "eos", "omit"),
        Special(258, "invalid", "error"),
    ),
    (256,),
    (257,),
)


def resign(raw: bytes) -> bytes:
    marker = b"checksum\t"
    start = raw.rfind(marker)
    if start < 0:
        raise ValueError("mutation lost checksum record")
    body = raw[:start]
    digest = hashlib.sha256(CHECKSUM_DOMAIN + body).hexdigest().encode("ascii")
    return body + marker + digest + b"\n"


def repair_payload_size(raw: bytes) -> bytes:
    payload = raw.index(b"kind\tbyte\n")
    checksum_algorithm = raw.index(b"checksum-algorithm\t")
    start = raw.index(b"payload-bytes\t")
    end = raw.index(b"\n", start)
    value = str(checksum_algorithm - payload).encode("ascii")
    return raw[:start] + b"payload-bytes\t" + value + raw[end:]


def semantic_mutation(raw: bytes, old: bytes, new: bytes) -> bytes:
    if raw.count(old) != 1:
        raise ValueError(f"mutation source is not unique: {old!r}")
    return resign(repair_payload_size(raw.replace(old, new, 1)))


def header_mutation(raw: bytes, old: bytes, new: bytes) -> bytes:
    if raw.count(old) != 1:
        raise ValueError(f"mutation source is not unique: {old!r}")
    return resign(raw.replace(old, new, 1))


def fixtures() -> dict[str, bytes]:
    raw = encode_artifact(RAW)
    strict = encode_artifact(Identity("strict", RAW.specials, RAW.prefix, RAW.suffix))
    overlap = encode_artifact(
        Identity(
            "raw",
            (Special(256, "boundary", "omit"), Special(257, "forbidden", "error")),
            (256, 256),
            (256, 256),
        )
    )
    optional = encode_artifact(RAW, minor=1, optional=(("opaque", b"00ff"),))
    duplicate_optional = optional.replace(b"optional-fields\t1", b"optional-fields\t2", 1)
    duplicate_optional = duplicate_optional.replace(
        b"optional-field\topaque\t00ff\n",
        b"optional-field\topaque\t00ff\noptional-field\topaque\t00ff\n",
        1,
    )
    result = {
        "valid-strict.tsv": strict,
        "valid-overlap.tsv": overlap,
        "valid-optional.tsv": optional,
        "corrupt-body.tsv": raw.replace(b"kind\tbyte", b"kind\tbytf", 1),
        "truncated.tsv": raw[:-1],
        "trailing.tsv": raw + b"x",
        "bad-checksum.tsv": raw[:-2] + (b"0" if raw[-2:-1] != b"0" else b"1") + b"\n",
        "major.tsv": header_mutation(raw, b"version\t1\t0", b"version\t2\t0"),
        "version.tsv": header_mutation(raw, b"version\t1\t0", b"version\t1\t00"),
        "required-feature.tsv": encode_artifact(
            RAW, minor=1, required=("new-semantics",)
        ),
        "optional-v1.tsv": encode_artifact(
            RAW, minor=0, optional=(("opaque", b"00"),)
        ),
        "reserved-limit.tsv": header_mutation(
            raw, b"limit-prefix\t4096", b"limit-prefix\t4095"
        ),
        "algorithm.tsv": header_mutation(
            raw, b"checksum-algorithm\tsha256", b"checksum-algorithm\tsha512"
        ),
        "payload-overflow.tsv": header_mutation(
            raw, b"payload-bytes\t200", b"payload-bytes\t9223372036854775808"
        ),
        "overlong-line.tsv": semantic_mutation(
            raw, b"normalization\tnone", b"normalization\t" + b"x" * 257
        ),
        "non-ascii.tsv": raw.replace(b"kind\tbyte", b"kind\tbyt\xff", 1),
        "bad-special-name.tsv": semantic_mutation(
            raw, b"special\t256\tbos\tomit", b"special\t256\tBos\tomit"
        ),
        "special-collision.tsv": semantic_mutation(
            raw, b"special\t257\teos\tomit", b"special\t257\tbos\tomit"
        ),
        "special-order.tsv": semantic_mutation(
            raw, b"special\t256\tbos\tomit", b"special\t257\tbos\tomit"
        ),
        "unknown-insertion.tsv": semantic_mutation(
            raw, b"prefix\t0\t256", b"prefix\t0\t259"
        ),
        "error-insertion.tsv": semantic_mutation(
            raw, b"prefix\t0\t256", b"prefix\t0\t258"
        ),
        "decode-policy.tsv": semantic_mutation(
            raw, b"special\t258\tinvalid\terror", b"special\t258\tinvalid\treplace"
        ),
        "normalization.tsv": semantic_mutation(
            raw, b"normalization\tnone", b"normalization\tnfc"
        ),
        "payload-size-mismatch.tsv": header_mutation(
            raw, b"payload-bytes\t200", b"payload-bytes\t199"
        ),
        "insertion-order.tsv": semantic_mutation(
            raw, b"prefix\t0\t256", b"prefix\t1\t256"
        ),
        "unknown-suffix.tsv": semantic_mutation(
            raw, b"suffix\t0\t257", b"suffix\t0\t259"
        ),
        "duplicate-optional.tsv": resign(duplicate_optional),
        "special-count-limit.tsv": semantic_mutation(
            raw, b"special-count\t3", b"special-count\t4097"
        ),
        "prefix-count-limit.tsv": semantic_mutation(
            raw, b"prefix-count\t1", b"prefix-count\t4097"
        ),
        "name-limit.tsv": semantic_mutation(
            raw, b"special\t256\tbos\tomit",
            b"special\t256\t" + b"a" * 65 + b"\tomit",
        ),
        "optional-value-limit.tsv": resign(
            optional.replace(b"optional-field\topaque\t00ff",
                             b"optional-field\topaque\t" + b"ab" * 65, 1)
        ),
        "file-limit.tsv": raw + b"x" * (1_048_577 - len(raw)),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    generated = fixtures()
    for name in sorted(generated):
        (args.output_directory / name).write_bytes(generated[name])
    print(f"T1 ADVERSARIAL FIXTURES PASS: {len(generated)} deterministic artifacts")


if __name__ == "__main__":
    main()
