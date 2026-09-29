"""Generate authentic D1 corpora for one E3 call of 1,024 or 8,192 D2 rows."""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_reference.corpus import FINGERPRINT, VOCAB_SIZE

HORIZONS = (1024, 8192)
TOKENS = (3, 197, 0, 255, 7, 7)


def materialize(root: Path, horizon: int) -> Path:
    if horizon not in HORIZONS:
        raise ValueError("unsupported E3 horizon")
    root.mkdir(parents=True, exist_ok=False)
    # D2 packed T=2 emits ceil((N-1)/2) rows. The final target is the only
    # token in shard two, so its removal after D2 open fails at the last row.
    values = tuple(TOKENS[index % len(TOKENS)] for index in range(2 * horizon + 1))
    shards = (values[:-1], values[-1:])
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
    assert len(reference.rows) == horizon
    assert sum(sum(row.loss_mask) for row in reference.rows) == 2 * horizon
    assert reference.rows[-1].targets[-1] == values[-1]
    return root / "shard-0000000000000001.ets"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=HORIZONS, required=True)
    args = parser.parse_args()
    print(materialize(args.output, args.horizon))


if __name__ == "__main__":
    main()
