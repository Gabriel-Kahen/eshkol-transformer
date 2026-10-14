"""Generate a genuine packed D2 corpus exactly one row over E3's call cap."""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_horizon.generate import SHARD_TOKENS, TOKENS
from tests.e3_reference.corpus import FINGERPRINT, VOCAB_SIZE

CAP_ROWS = 8193


def materialize(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=False)
    values = tuple(TOKENS[index % len(TOKENS)] for index in range(2 * CAP_ROWS + 1))
    full_tokens = (2 * CAP_ROWS // SHARD_TOKENS) * SHARD_TOKENS
    shards = tuple(
        values[start : start + SHARD_TOKENS]
        for start in range(0, full_tokens, SHARD_TOKENS)
    ) + (values[full_tokens:],)
    write_d1_resource(root, shards, fingerprint=FINGERPRINT, vocab=VOCAB_SIZE)
    loaded, digest = load_d1_resource(
        root, expected_fingerprint=FINGERPRINT, expected_vocab=VOCAB_SIZE,
        maximum_total_tokens=len(values),
    )
    reference = ReferenceDataset(
        loaded, ReferenceConfig(1, 2, None, 1, True),
        manifest_digest=digest, tokenizer_fingerprint=FINGERPRINT,
        vocab_size=VOCAB_SIZE,
    )
    if len(reference.rows) != CAP_ROWS or any(
        row.loss_mask != (True, True) for row in reference.rows
    ):
        raise AssertionError("D2 did not emit exactly 8,193 full packed rows")
    return root / "shard-0000000000000000.ets"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(materialize(args.output))


if __name__ == "__main__":
    main()
