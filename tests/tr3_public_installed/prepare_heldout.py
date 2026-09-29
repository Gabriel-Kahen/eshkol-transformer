"""Generate the frozen, disjoint TR3 train/held-out D1 resources."""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.d2.test_resources import write_d1_resource
from tests.tr3_reference.constants import (
    HELDOUT_SHARDS,
    TOKENIZER_FINGERPRINT,
    TRAIN_SHARDS,
    VOCAB_SIZE,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    if output.exists():
        raise FileExistsError(output)
    transitions = lambda shards: {
        (left, right)
        for shard in shards
        for left, right in zip(shard, shard[1:])
    }
    if (transitions(TRAIN_SHARDS) != {(3, 197), (197, 41)}
            or transitions(HELDOUT_SHARDS)
            != {(5, 197), (197, 197), (6, 41)}
            or transitions(TRAIN_SHARDS) & transitions(HELDOUT_SHARDS)):
        raise AssertionError("training and validation transitions overlap")
    for name, shards in (("train", TRAIN_SHARDS), ("heldout", HELDOUT_SHARDS)):
        path = output / name
        path.mkdir(parents=True)
        write_d1_resource(path, shards, fingerprint=TOKENIZER_FINGERPRINT,
                          vocab=VOCAB_SIZE)


if __name__ == "__main__":
    main()
