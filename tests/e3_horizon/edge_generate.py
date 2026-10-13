"""Deterministic genuine D1/D2 fixtures for the private E3 edge gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_reference.corpus import FINGERPRINT, VOCAB_SIZE

CASES = {
    "empty": ((), (), 0),
    "singleton": (((3,),), (), 0),
    "entry-eos": (((3, 197, 0),), ((3, 197),), 1),
    "one-token-tail": (((3, 197, 0, 255),), ((3, 197), (0,)), 0),
}


def materialize(root: Path, case: str) -> tuple[bytes, tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[bool, ...]], ...]]:
    if case not in CASES:
        raise ValueError("unsupported E3 edge case")
    shards, expected_inputs, skip = CASES[case]
    root.mkdir(parents=True, exist_ok=False)
    write_d1_resource(root, shards, fingerprint=FINGERPRINT, vocab=VOCAB_SIZE)
    loaded, digest = load_d1_resource(
        root, expected_fingerprint=FINGERPRINT, expected_vocab=VOCAB_SIZE
    )
    reference = ReferenceDataset(
        loaded, ReferenceConfig(1, 2, None, 1, True, maximum_batch_bytes=34),
        manifest_digest=digest, tokenizer_fingerprint=FINGERPRINT,
        vocab_size=VOCAB_SIZE,
    )
    rows = reference.rows
    assert tuple(row.inputs[:sum(row.loss_mask)] for row in rows) == expected_inputs
    assert len(rows) == len(expected_inputs)
    for _ in range(skip):
        assert reference.next_batch() is not None
    cursor = reference.snapshot()
    if case == "entry-eos":
        assert reference.next_batch() is None
    emitted = tuple((row.inputs, row.targets, row.loss_mask) for row in rows)
    return cursor, emitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    args = parser.parse_args()
    cursor, rows = materialize(args.output, args.case)
    print(args.case, cursor.hex(), repr(rows), sep="\t")


if __name__ == "__main__":
    main()
