#!/usr/bin/env python3
"""Create deterministic corrupt D1 shard fixtures for compiled T2 rejection."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


SHARD = "shard-0000000000000000.ets"


def clone(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination / SHARD


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--malformed", type=Path, required=True)
    parser.add_argument("--truncated", type=Path, required=True)
    parser.add_argument("--checksum-corrupt", type=Path, required=True)
    arguments = parser.parse_args()

    malformed = clone(arguments.template, arguments.malformed)
    data = bytearray(malformed.read_bytes())
    data[0] ^= 0x01
    malformed.write_bytes(data)

    truncated = clone(arguments.template, arguments.truncated)
    truncated.write_bytes(truncated.read_bytes()[:-1])

    checksum = clone(arguments.template, arguments.checksum_corrupt)
    data = bytearray(checksum.read_bytes())
    data[-1] ^= 0x01
    checksum.write_bytes(data)

    print("T2 D1 NEGATIVE FIXTURES PASS: 3 deterministic shard mutations")


if __name__ == "__main__":
    main()
