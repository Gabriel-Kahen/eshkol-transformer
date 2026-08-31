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


def resign_artifact(raw: bytes) -> bytes:
    marker = b"checksum\t"
    start = raw.rfind(marker)
    if start < 0:
        raise ValueError("artifact has no checksum record")
    unsigned = raw[:start]
    digest = hashlib.sha256(CHECKSUM_DOMAIN + unsigned).hexdigest().encode("ascii")
    return unsigned + marker + digest + b"\n"


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


def one_over_optional_fields() -> bytes:
    raw = encode_artifact(Identity(), minor=1)
    marker = b"optional-fields\t0\n"
    if raw.count(marker) != 1:
        raise ValueError("unexpected optional-field count record")
    return resign_artifact(raw.replace(marker, b"optional-fields\t4097\n", 1))


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

    optional_fields = tuple(
        (f"o{index:04d}", b"ab" * 64 if index == 4095 else b"00")
        for index in range(4096)
    )
    optional_names = tuple(name for name, _ in optional_fields)
    if (
        len(optional_fields) != 4096
        or optional_names != tuple(sorted(optional_names))
        or len(set(optional_names)) != 4096
        or len(bytes.fromhex(optional_fields[-1][1].decode("ascii"))) != 64
    ):
        raise SystemExit("optional-max generator invariants failed")
    optional_exact = encode_artifact(
        Identity(), minor=1, optional=optional_fields
    )
    if len(optional_exact) != 98_891:
        raise SystemExit(
            f"unexpected optional-max artifact size: {len(optional_exact)}"
        )
    optional_over = one_over_optional_fields()
    if len(optional_over) != 461 or len(optional_over) > 1_048_576:
        raise SystemExit(
            f"unexpected optional-one-over artifact size: {len(optional_over)}"
        )

    fixtures = {
        "exact-max.tsv": exact,
        "specials-one-over.tsv": one_over_specials(),
        "prefix-one-over.tsv": one_over_prefix(),
        "suffix-one-over.tsv": one_over_suffix(),
        "optional-max.tsv": optional_exact,
        "optional-one-over.tsv": optional_over,
    }
    for name, contents in fixtures.items():
        (args.output_directory / name).write_bytes(contents)

    print(
        "T1 LIMIT FIXTURES PASS: 6 deterministic artifacts; "
        "exact-max 472645 bytes; optional-max 98891 bytes"
    )


if __name__ == "__main__":
    main()
