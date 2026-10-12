"""Generate authentic raw-T1/V256 D1 resources for the E3 N5 one-over gate."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_reference.corpus import FINGERPRINT, IDENTITY, SOURCE_BYTES, VOCAB_SIZE
from tests.e3_reference.transcript import encode
from tests.t1.reference import encode_artifact, encode_bytes

WIDTH = 2
ROWS = 8193
EXACT_ROWS = ROWS - 1
SHARD_TOKENS = 2048
TOTAL_TOKENS = 2 * ROWS + 1
EXACT_TOKENS = 2 * EXACT_ROWS


def _digest(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"path": path.name, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def materialize(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=False)
    assert encode_bytes(IDENTITY, SOURCE_BYTES) == tuple(SOURCE_BYTES)
    tokenizer = root / "tokenizer.etok"
    tokenizer.write_bytes(encode_artifact(IDENTITY))
    config = ReferenceConfig(1, WIDTH, None, 1, True)
    cases = []
    for name, count in (("one-over", TOTAL_TOKENS),
                        ("exact-limit", EXACT_TOKENS),
                        ("retry-small", len(SOURCE_BYTES))):
        values = tuple(SOURCE_BYTES[i % len(SOURCE_BYTES)] for i in range(count))
        shards = tuple(values[start:start + SHARD_TOKENS]
                       for start in range(0, count, SHARD_TOKENS))
        directory = root / name
        directory.mkdir()
        write_d1_resource(directory, shards, fingerprint=FINGERPRINT, vocab=VOCAB_SIZE)
        loaded, digest = load_d1_resource(
            directory, expected_fingerprint=FINGERPRINT,
            expected_vocab=VOCAB_SIZE, maximum_total_tokens=TOTAL_TOKENS,
        )
        reference = ReferenceDataset(
            loaded, config, manifest_digest=digest,
            tokenizer_fingerprint=FINGERPRINT, vocab_size=VOCAB_SIZE,
        )
        independent_count = (count - 1 + WIDTH - 1) // WIDTH
        assert len(reference.rows) == independent_count
        assert sum(sum(row.loss_mask) for row in reference.rows) == count - 1
        emitted = []
        for ordinal in range(independent_count):
            start = ordinal * WIDTH
            active = min(WIDTH, count - 1 - start)
            row = {
                "inputs": list(values[start:start + active]) + [0] * (WIDTH - active),
                "targets": list(values[start + 1:start + 1 + active])
                           + [0] * (WIDTH - active),
                "mask": [True] * active + [False] * (WIDTH - active),
            }
            actual = reference.rows[ordinal]
            assert row == {"inputs": list(actual.inputs),
                           "targets": list(actual.targets),
                           "mask": list(actual.loss_mask)}
            emitted.append(row)
        if name == "one-over":
            assert independent_count == ROWS
            assert len(shards) == 9 and len(shards[-1]) == 3
            assert reference.rows[-1].inputs == values[-3:-1]
            assert reference.rows[-1].targets == values[-2:]
        elif name == "exact-limit":
            assert independent_count == EXACT_ROWS
            assert len(shards) == 8 and len(shards[-1]) == SHARD_TOKENS
            assert reference.rows[-1].inputs == (values[-2], 0)
            assert reference.rows[-1].targets == (values[-1], 0)
            assert reference.rows[-1].loss_mask == (True, False)
        else:
            assert independent_count == 3
        cases.append({
            "name": name, "source_bytes": bytes(values).hex(),
            "tokens": count, "expected_rows": independent_count,
            "active_positions": count - 1,
            "d2": {"batch_size": 1, "sequence_length": 2,
                   "shuffle_seed": None, "shuffle_window_rows": 1,
                   "packing": True, "start_cursor": reference.snapshot().hex()},
            "manifest_trailer_sha256": digest.hex(),
            "files": [_digest(path) for path in sorted(directory.iterdir())],
            "rows": emitted,
        })
    result = {
        "kind": "e3-n5-private-corpus-v1", "profile": "raw-t1-v256-n1-t2-d4",
        "tokenizer": {"fingerprint": FINGERPRINT, **_digest(tokenizer)},
        "cases": cases,
    }
    (root / "corpus-manifest.json").write_bytes(encode(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    materialize(arguments.output)
    print(arguments.output / "corpus-manifest.json")


if __name__ == "__main__":
    main()
