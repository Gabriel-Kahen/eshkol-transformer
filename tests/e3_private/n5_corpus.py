"""Checked D1 one-over corpus and independent exact-boundary materialization."""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_reference.corpus import FINGERPRINT, VOCAB_SIZE

ROWS = 8193
RAW_TOKENS = 2 * ROWS + 1
SHARD_TOKENS = 2048
PATTERN = (3, 197, 0, 255, 7, 7)


def _materialize(root: Path, values: tuple[int, ...], rows: int) -> Path:
    root.mkdir(parents=True, exist_ok=False)
    shards = tuple(
        values[start : start + SHARD_TOKENS]
        for start in range(0, len(values), SHARD_TOKENS)
    )
    write_d1_resource(root, shards, fingerprint=FINGERPRINT, vocab=VOCAB_SIZE)
    loaded, digest = load_d1_resource(
        root, expected_fingerprint=FINGERPRINT, expected_vocab=VOCAB_SIZE,
        maximum_total_tokens=len(values),
    )
    if sum(map(len, loaded)) != len(values) or tuple(loaded) != shards:
        raise AssertionError("D1 materialization changed the token stream")
    reference = ReferenceDataset(
        loaded, ReferenceConfig(1, 2, None, 1, True),
        manifest_digest=digest, tokenizer_fingerprint=FINGERPRINT,
        vocab_size=VOCAB_SIZE,
    )
    if len(reference.rows) != rows or not all(
        row.loss_mask == (True, True) for row in reference.rows
    ):
        raise AssertionError("D2 reference produced an unexpected full-row count")
    if reference.rows[-1].targets[-1] != values[-1]:
        raise AssertionError("D2 terminal target changed")
    return root


def materialize(root: Path) -> Path:
    values = tuple(PATTERN[index % len(PATTERN)] for index in range(RAW_TOKENS))
    return _materialize(root, values, ROWS)


def materialize_boundary(root: Path) -> Path:
    values = tuple(PATTERN[index % len(PATTERN)] for index in range(2, RAW_TOKENS))
    return _materialize(root, values, ROWS - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundary", action="store_true")
    args = parser.parse_args()
    print(materialize_boundary(args.output) if args.boundary else materialize(args.output))


if __name__ == "__main__":
    main()
