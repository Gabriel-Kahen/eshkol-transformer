#!/usr/bin/env python3
"""Generate the small frozen T1 development-format fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.t1.tokenizer_format import SpecialToken, TokenizerIdentity, encode_artifact


def fixture_identity() -> TokenizerIdentity:
    return TokenizerIdentity(
        utf8_policy="raw",
        specials=(
            SpecialToken(256, "bos", "omit"),
            SpecialToken(257, "eos", "omit"),
            SpecialToken(258, "invalid", "error"),
        ),
        prefix=(256,),
        suffix=(257,),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.output.write_bytes(encode_artifact(fixture_identity()))


if __name__ == "__main__":
    main()
