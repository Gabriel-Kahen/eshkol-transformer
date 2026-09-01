#!/usr/bin/env python3
"""Generate bounded T2 artifacts for delivered-runtime negative checks."""

import argparse
import hashlib
from pathlib import Path

from tests.t2.generate_fixture import DOCUMENTS, SPECIALS, fixture_bytes
from tests.t2.reference import CHECKSUM_DOMAIN, encode_artifact, train


def resign(source: bytes) -> bytes:
    start = source.rfind(b"checksum\t")
    body = source[:start]
    digest = hashlib.sha256(CHECKSUM_DOMAIN + body).hexdigest().encode()
    return body + b"checksum\t" + digest + b"\n"


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
    }
    for name, payload in fixtures.items():
        (arguments.output_directory / name).write_bytes(payload)
    print("T2 ADVERSARIAL FIXTURES PASS: 10 deterministic artifacts")


if __name__ == "__main__":
    main()
