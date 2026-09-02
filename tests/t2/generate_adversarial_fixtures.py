#!/usr/bin/env python3
"""Generate bounded T2 artifacts for delivered-runtime negative checks."""

import argparse
import hashlib
from pathlib import Path

from tests.t2.generate_fixture import DOCUMENTS, SPECIALS, fixture_bytes
from tests.t2.reference import CHECKSUM_DOMAIN, V1_HEADER, encode_artifact, train


def resign(source: bytes) -> bytes:
    start = source.rfind(b"checksum\t")
    body = source[:start]
    digest = hashlib.sha256(CHECKSUM_DOMAIN + body).hexdigest().encode()
    return body + b"checksum\t" + digest + b"\n"


def artifact_with_payload(payload: bytes) -> bytes:
    body = V1_HEADER + f"payload-bytes\t{len(payload)}\n".encode() + payload
    return resign(body + b"checksum-algorithm\tsha256\nchecksum\t" + b"0" * 64 + b"\n")


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
    fixtures = {
        "truncated.tsv": valid[:-1],
        "bad-checksum.tsv": valid.replace(b"merge\t0\t256", b"merge\t0\t257", 1),
        "algorithm.tsv": resign(
            valid.replace(b"checksum-algorithm\tsha256", b"checksum-algorithm\tsha512", 1)
        ),
        "oversized.tsv": b"x" * 1_048_577,
        "forward-reference.tsv": resign(
            valid.replace(b"payload-bytes\t295", b"payload-bytes\t296", 1)
            .replace(b"merge\t0\t256\t97\t110", b"merge\t0\t256\t97\t256", 1)
        ),
        "rank-gap.tsv": resign(
            valid.replace(b"merge\t0\t256", b"merge\t1\t256", 1)
        ),
        "merge-count-mismatch.tsv": resign(
            valid.replace(b"merge-count\t4", b"merge-count\t5", 1)
        ),
        "special-id-collision.tsv": resign(
            valid.replace(b"special\t260\tbos", b"special\t259\tbos", 1)
        ),
        "noncanonical-version.tsv": resign(
            valid.replace(b"version\t1\t0", b"version\t01\t0", 1)
        ),
        "alternate-same-vocab.tsv": encode_artifact(
            train(DOCUMENTS, 8, 2, "strict", SPECIALS, ("bos",), ("eos",))
        ),
        # Compiled-parser matrix. Mutations after the checksum boundary are
        # intentionally not resigned; semantic/header mutations are resigned.
        "trailing-byte.tsv": valid + b"x",
        "extra-lf.tsv": valid + b"\n",
        "duplicate-merge-pair.tsv": resign(
            valid.replace(b"merge\t1\t257\t98\t256", b"merge\t1\t257\t97\t110", 1)
        ),
        "out-of-order-special.tsv": resign(
            valid.replace(
                b"special\t260\tbos\tomit\nspecial\t261\teos\tomit\n",
                b"special\t261\teos\tomit\nspecial\t260\tbos\tomit\n",
                1,
            )
        ),
        "out-of-order-prefix.tsv": resign(
            valid.replace(b"prefix\t0\t260", b"prefix\t1\t260", 1)
        ),
        "out-of-order-suffix.tsv": resign(
            valid.replace(b"suffix\t0\t261", b"suffix\t1\t261", 1)
        ),
        "required-feature.tsv": resign(
            valid.replace(b"required-features\t0", b"required-features\t1", 1)
        ),
        "unsupported-major.tsv": resign(
            valid.replace(b"version\t1\t0", b"version\t2\t0", 1)
        ),
        "unsupported-minor.tsv": resign(
            valid.replace(b"version\t1\t0", b"version\t1\t1", 1)
        ),
        "altered-limit-file-bytes.tsv": resign(
            valid.replace(b"limit-file-bytes\t1048576", b"limit-file-bytes\t1048575", 1)
        ),
        "altered-limit-merges.tsv": resign(
            valid.replace(b"limit-merges\t256", b"limit-merges\t255", 1)
        ),
        "altered-limit-token-bytes.tsv": resign(
            valid.replace(b"limit-token-bytes\t256", b"limit-token-bytes\t255", 1)
        ),
        "altered-limit-specials.tsv": resign(
            valid.replace(b"limit-specials\t4096", b"limit-specials\t4095", 1)
        ),
        "altered-limit-name-bytes.tsv": resign(
            valid.replace(b"limit-name-bytes\t64", b"limit-name-bytes\t63", 1)
        ),
        "altered-limit-prefix.tsv": resign(
            valid.replace(b"limit-prefix\t4096", b"limit-prefix\t4095", 1)
        ),
        "altered-limit-suffix.tsv": resign(
            valid.replace(b"limit-suffix\t4096", b"limit-suffix\t4095", 1)
        ),
        "altered-payload-size-short.tsv": resign(
            valid.replace(b"payload-bytes\t295", b"payload-bytes\t294", 1)
        ),
        "altered-payload-size-long.tsv": resign(
            valid.replace(b"payload-bytes\t295", b"payload-bytes\t296", 1)
        ),
        "malformed-name.tsv": resign(
            valid.replace(b"special\t260\tbos\tomit", b"special\t260\tBos\tomit", 1)
        ),
        "malformed-special-policy.tsv": resign(
            valid.replace(b"special\t260\tbos\tomit", b"special\t260\tbos\toops", 1)
        ),
        "malformed-utf8-policy.tsv": resign(
            valid.replace(b"utf8-policy\traw", b"utf8-policy\tbad", 1)
        ),
        "learned-token-overflow.tsv": artifact_with_payload(overflow_payload()),
    }
    for name, payload in fixtures.items():
        (arguments.output_directory / name).write_bytes(payload)
    print(f"T2 ADVERSARIAL FIXTURES PASS: {len(fixtures)} deterministic artifacts")


if __name__ == "__main__":
    main()
