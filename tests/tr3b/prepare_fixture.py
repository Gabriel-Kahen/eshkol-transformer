from __future__ import annotations

import argparse
from pathlib import Path

from tests.d2.test_resources import write_d1_resource


FINGERPRINT = (
    "sha256:eshkol-byte-tokenizer-v1:"
    "aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", choices=("two", "one", "empty"), default="two")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    shards = {"two": ((3, 197, 41, 7),),
              "one": ((3, 197),), "empty": ()}[args.rows]
    write_d1_resource(args.output, shards, fingerprint=FINGERPRINT, vocab=256)


if __name__ == "__main__":
    main()
