"""Development-only DATA4 report oracle and deterministic shard corruption."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sys


def main() -> None:
    report = Path(sys.argv[1]).read_bytes()
    directory = Path(sys.argv[2])
    assert report.endswith(b"\n")
    prefix, trailer = report.rsplit(b"checksum\t", 1)
    assert trailer == hashlib.sha256(prefix).hexdigest().encode() + b"\n"
    expected_header = (
        b"format\teshkol-data4-corpus-report\n",
        b"version\t2\t0\n",
        b"filter\traw-byte-length-inclusive\n",
        b"dedup\traw-exact-first-passing\n",
        b"minimum-bytes\t2\n",
        b"maximum-bytes\t2\n",
    )
    expected_totals = (
        b"vocabulary-size\t257\n",
        b"input-count\t3\n",
        b"input-bytes\t6\n",
        b"kept-count\t2\n",
        b"kept-bytes\t4\n",
        b"shard-token-limit\t3\n",
        b"shard-count\t2\n",
        b"total-tokens\t6\n",
        b"manifest-sha256\t"
        + hashlib.sha256((directory / "manifest.etm").read_bytes()).hexdigest().encode()
        + b"\n",
    )
    lines = prefix.splitlines(keepends=True)
    a = hashlib.sha256(b"AB").hexdigest().encode()
    b = hashlib.sha256(b"CD").hexdigest().encode()
    expected_rows = (
        b"document\t0\ta\t2\t" + a + b"\tkeep\t\n",
        b"document\t1\tduplicate\t2\t" + a + b"\tduplicate\ta\n",
        b"document\t2\tb\t2\t" + b + b"\tkeep\t\n",
        b"checksum-algorithm\tsha256\n",
    )
    assert len(lines) == 20
    assert tuple(lines[:6]) == expected_header
    assert re.fullmatch(
        rb"tokenizer-fingerprint\tsha256:eshkol-bpe-tokenizer-v1:[0-9a-f]{64}\n",
        lines[6],
    )
    assert tuple(lines[7:16]) == expected_totals
    assert tuple(lines[16:]) == expected_rows

    shard = directory / "shard-0000000000000000.ets"
    raw = bytearray(shard.read_bytes())
    assert len(raw) > 33
    raw[-33] ^= 1  # Payload byte, not the SHA-256 trailer.
    shard.write_bytes(raw)


if __name__ == "__main__":
    main()
