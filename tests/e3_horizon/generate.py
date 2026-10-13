"""Generate authentic D1 corpora for E3's 1,024/8,192/8,193-row boundary."""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_reference.corpus import FINGERPRINT, VOCAB_SIZE

HORIZONS = (1024, 8192, 8193)
TOKENS = (3, 197, 0, 255, 7, 7)
SHARD_TOKENS = 2048


def materialize(root: Path, horizon: int) -> Path:
    if horizon not in HORIZONS:
        raise ValueError("unsupported E3 horizon")
    root.mkdir(parents=True, exist_ok=False)
    # D2 packed T=2 emits ceil((N-1)/2) rows. Even-sized canonical shards
    # preserve the rows. H=8,193 adds two positions after the H=8,192 target.
    values = tuple(TOKENS[index % len(TOKENS)] for index in range(2 * horizon + 1))
    shards = tuple(
        values[start : start + SHARD_TOKENS]
        for start in range(0, len(values), SHARD_TOKENS)
    )
    write_d1_resource(root, shards, fingerprint=FINGERPRINT, vocab=VOCAB_SIZE)
    loaded, digest = load_d1_resource(
        root, expected_fingerprint=FINGERPRINT, expected_vocab=VOCAB_SIZE,
        maximum_total_tokens=len(values),
    )
    reference = ReferenceDataset(
        loaded, ReferenceConfig(1, 2, None, 1, True,
                                maximum_batch_bytes=34),
        manifest_digest=digest, tokenizer_fingerprint=FINGERPRINT,
        vocab_size=VOCAB_SIZE,
    )
    assert len(reference.rows) == horizon
    assert sum(sum(row.loss_mask) for row in reference.rows) == 2 * horizon
    assert reference.rows[-1].targets[-1] == values[-1]
    return root / f"shard-{len(shards) - 1:016d}.ets"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=HORIZONS, required=True)
    args = parser.parse_args()
    print(materialize(args.output, args.horizon))


if __name__ == "__main__":
    main()
