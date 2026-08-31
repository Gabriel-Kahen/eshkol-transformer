"""Generate deterministic T1 exact-limit and one-over test artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from tests.t1.reference import CHECKSUM_DOMAIN, Identity, Special, V1_HEADER, encode_artifact


def signed_payload(payload: bytes) -> bytes:
    unsigned = (
        V1_HEADER
        + f"payload-bytes\t{len(payload)}\n".encode("ascii")
        + payload
        + b"checksum-algorithm\tsha256\n"
    )
    digest = hashlib.sha256(CHECKSUM_DOMAIN + unsigned).hexdigest().encode("ascii")
    return unsigned + b"checksum\t" + digest + b"\n"


def base_payload(special_count: int) -> list[bytes]:
    return [
        b"kind\tbyte\n",
        b"normalization\tnone\n",
        b"utf8-policy\tstrict\n",
        b"byte-ids\t0\t255\n",
        f"special-count\t{special_count}\n".encode("ascii"),
    ]


def one_over_specials() -> bytes:
    return signed_payload(b"".join(base_payload(4097)))


def one_over_prefix() -> bytes:
    lines = base_payload(1)
    lines.extend((b"special\t256\ta\tomit\n", b"prefix-count\t4097\n"))
    return signed_payload(b"".join(lines))


def one_over_suffix() -> bytes:
    lines = base_payload(1)
    lines.extend(
        (
            b"special\t256\ta\tomit\n",
            b"prefix-count\t0\n",
            b"suffix-count\t4097\n",
        )
    )
    return signed_payload(b"".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)

    specials = tuple(
        Special(256 + index, "a" + format(index, "063d"), "omit")
        for index in range(4096)
    )
    exact = encode_artifact(
        Identity("strict", specials, (256,) * 4096, (4351,) * 4096)
    )
    if len(exact) != 472_645:
        raise SystemExit(f"unexpected exact-max artifact size: {len(exact)}")

    fixtures = {
        "exact-max.tsv": exact,
        "specials-one-over.tsv": one_over_specials(),
        "prefix-one-over.tsv": one_over_prefix(),
        "suffix-one-over.tsv": one_over_suffix(),
    }
    for name, contents in fixtures.items():
        (args.output_directory / name).write_bytes(contents)

    print("T1 LIMIT FIXTURES PASS: 4 deterministic artifacts; exact-max 472645 bytes")


if __name__ == "__main__":
    main()
