"""Independent T1/D1/D2 resources for the private E3 cursor boundary gate."""

from __future__ import annotations

from pathlib import Path

from tests.d2.reference import ReferenceConfig, ReferenceDataset, packed_row_count
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_reference.corpus import FINGERPRINT, IDENTITY, VOCAB_SIZE
from tests.t1.reference import encode_artifact, encode_bytes


# Keep these outside the frozen five-case transcript. The first three have no
# evaluated target; the fourth starts on the padded, one-valid-token final row.
# The fifth is 16,387 authentic raw bytes: D2 T=2 packing emits 8,193 rows.
ONE_OVER_TOKENS = tuple((index * 37 + 11) & 255 for index in range(16_387))
CASES = (
    ("empty", (), 0),
    ("singleton", ((7,),), 0),
    ("entry-eos", ((7, 8),), 1),
    ("final-suffix", ((3, 197, 0, 255, 7, 7),), 2),
    ("one-over", (ONE_OVER_TOKENS,), 0),
)


def materialize(root: Path) -> dict[str, dict[str, object]]:
    """Write fresh authentic resources and return independently emitted rows/cursors."""

    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (root / "tokenizer.etok").write_bytes(encode_artifact(IDENTITY))
    result: dict[str, dict[str, object]] = {}
    for name, shards, start_ordinal in CASES:
        if tuple(encode_bytes(IDENTITY, bytes(token for shard in shards for token in shard))) != tuple(
            token for shard in shards for token in shard
        ):
            raise AssertionError("raw T1 byte identity changed")
        directory = root / name
        directory.mkdir()
        write_d1_resource(directory, shards, fingerprint=FINGERPRINT, vocab=VOCAB_SIZE)
        loaded, digest = load_d1_resource(
            directory, expected_fingerprint=FINGERPRINT,
            expected_vocab=VOCAB_SIZE, maximum_total_tokens=65536,
        )
        dataset = ReferenceDataset(
            loaded, ReferenceConfig(1, 2, None, 1, True),
            manifest_digest=digest, tokenizer_fingerprint=FINGERPRINT,
            vocab_size=VOCAB_SIZE,
        )
        for _ in range(start_ordinal):
            if dataset.next_batch() is None:
                raise AssertionError("cursor starts beyond EOS")
        saved = dataset.snapshot()
        batches: list[dict[str, object]] = []
        if name == "one-over":
            if len(dataset.rows) != packed_row_count(len(ONE_OVER_TOKENS), 2):
                raise AssertionError("independent D2 row count changed")
        else:
            while (batch := dataset.next_batch()) is not None:
                batches.append({
                    "inputs": list(batch.inputs[0]),
                    "targets": list(batch.targets[0]),
                    "mask": list(batch.loss_mask[0]),
                })
        result[name] = {
            "directory": directory,
            "start_ordinal": start_ordinal,
            "total_rows": len(dataset.rows),
            "saved_cursor": saved,
            "batches": batches,
        }
    return result
