#!/usr/bin/env python3
"""Emit the deterministic development-only T2 golden tokenizer artifact."""

import argparse
from pathlib import Path

from tests.t2.reference import encode_artifact, train


DOCUMENTS = ((b"banana ", b"banana"), (b"bandana",), (b"banana",))
SPECIALS = (("bos", "omit"), ("eos", "omit"), ("invalid", "error"))


def fixture_bytes() -> bytes:
    core = train(DOCUMENTS, 8, 2, "raw", SPECIALS, ("bos",), ("eos",))
    return encode_artifact(core)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("fixtures") / "bpe_tokenizer_v1.tsv",
    )
    arguments = parser.parse_args()
    arguments.output.write_bytes(fixture_bytes())
