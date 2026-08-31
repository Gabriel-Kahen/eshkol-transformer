#!/usr/bin/env python3
"""Regenerate the frozen development-only T1 v1 format fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.t1.reference import Identity, Special, encode_artifact


def fixture_identity() -> Identity:
    return Identity(
        "raw",
        (
            Special(256, "bos", "omit"),
            Special(257, "eos", "omit"),
            Special(258, "invalid", "error"),
        ),
        (256,),
        (257,),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_bytes(encode_artifact(fixture_identity()))


if __name__ == "__main__":
    main()
