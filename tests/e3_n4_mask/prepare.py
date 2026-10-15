"""Materialize checked raw-V256 D1 resources for the fixed T=2 mask census."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.d1.reference_format import parse_corpus
from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.t1.reference import Identity, encode_artifact, fingerprint

IDENTITY = Identity("raw", (), (), ())
FINGERPRINT = fingerprint(IDENTITY)
CASES = (("p2", (3, 197), True, [1, 0]),
         ("u2", (3, 197), False, [1, 0]),
         ("p3", (3, 197, 0), True, [1, 1]),
         ("u3", (3, 197, 0), False, [1, 1]))


def materialize(output: Path) -> dict[str, list[int]]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("N4 output must be fresh")
    output.mkdir(parents=True)
    (output / "tokenizer.etok").write_bytes(encode_artifact(IDENTITY))
    masks: dict[str, list[int]] = {}
    for name, tokens, packed, expected in CASES:
        directory = output / name
        directory.mkdir()
        write_d1_resource(directory, (tokens,), fingerprint=FINGERPRINT, vocab=256)
        loaded, digest = load_d1_resource(
            directory, expected_fingerprint=FINGERPRINT,
            expected_vocab=256, maximum_total_tokens=3)
        if tuple(parse_corpus(directory)["tokens"]) != tokens:
            raise AssertionError("checked D1 tokens changed")
        dataset = ReferenceDataset(
            loaded, ReferenceConfig(1, 2, None, 1, packed),
            manifest_digest=digest, tokenizer_fingerprint=FINGERPRINT,
            vocab_size=256)
        rows = []
        while (batch := dataset.next_batch()) is not None:
            rows.append([int(bit) for bit in batch.loss_mask[0]])
        if rows != [expected]:
            raise AssertionError(f"{name}: D2 row equation changed: {rows}")
        masks[name] = rows[0]
    (output / "masks.json").write_text(
        json.dumps(masks, sort_keys=True, separators=(",", ":")) + "\n")
    return masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.output)


if __name__ == "__main__":
    main()
