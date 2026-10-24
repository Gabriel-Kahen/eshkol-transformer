"""Prepare deterministic, checksummed N5 D1 resources and D2 cursor facts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tests.d2.reference import ReferenceConfig, ReferenceDataset, decode_reference_cursor
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_reference.corpus import FINGERPRINT, IDENTITY, VOCAB_SIZE
from tests.t1.reference import encode_artifact

TOKENS = (3, 197, 0, 255, 7, 7)
SHARD_TOKENS = 2048
CASES = ("empty", "singleton", "entry-eos", "one-token", "cap-8192", "cap-8193")


def shards_for(case: str) -> tuple[tuple[int, ...], ...]:
    if case == "empty":
        return ()
    if case == "singleton":
        return ((3,),)
    if case in ("entry-eos", "one-token"):
        return ((3, 197),)
    if case in ("cap-8192", "cap-8193"):
        horizon = int(case.removeprefix("cap-"))
        values = tuple(TOKENS[index % len(TOKENS)] for index in range(2 * horizon + 1))
        return tuple(values[start:start + SHARD_TOKENS]
                     for start in range(0, len(values), SHARD_TOKENS))
    raise ValueError("unknown N5 case")


def _row(row: object) -> dict[str, list[int] | list[bool]]:
    return {"inputs": list(row.inputs[0]), "targets": list(row.targets[0]),
            "mask": list(row.loss_mask[0])}


def materialize(root: Path) -> dict[str, object]:
    ancestor = root.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        if ancestor.is_symlink():
            raise ValueError("N5 output ancestors must not be symlinks")
        ancestor = ancestor.parent
    if ancestor.is_symlink() or ancestor.resolve() != ancestor.absolute():
        raise ValueError("N5 output ancestors must not be symlinks")
    if root.exists() or root.is_symlink():
        raise FileExistsError("N5 output path must be fresh")
    root.mkdir(parents=True)
    tokenizer = encode_artifact(IDENTITY)
    (root / "tokenizer.etok").write_bytes(tokenizer)
    cases = []
    for name in CASES:
        directory = root / name
        directory.mkdir()
        source_shards = shards_for(name)
        write_d1_resource(directory, source_shards,
                          fingerprint=FINGERPRINT, vocab=VOCAB_SIZE)
        loaded, digest = load_d1_resource(
            directory, expected_fingerprint=FINGERPRINT,
            expected_vocab=VOCAB_SIZE, maximum_manifest_bytes=1048576,
            maximum_shard_bytes=1048576, maximum_total_tokens=65536)
        if loaded != source_shards:
            raise AssertionError("D1 readback differs from generated source")
        config = ReferenceConfig(1, 2, None, 1, True,
                                 maximum_batch_bytes=34)
        dataset = ReferenceDataset(loaded, config, manifest_digest=digest,
                                   tokenizer_fingerprint=FINGERPRINT,
                                   vocab_size=VOCAB_SIZE)
        total_rows = len(dataset.rows)
        skip = total_rows if name == "entry-eos" else 0
        for _ in range(skip):
            if dataset.next_batch() is None:
                raise AssertionError("D2 EOS setup stopped early")
        entry = dataset.snapshot()
        emitted = 0
        active = 0
        first = last = None
        rows = []
        while (batch := dataset.next_batch()) is not None:
            row = _row(batch)
            rows.append(row)
            if first is None:
                first = row
            last = row
            emitted += 1
            active += sum(row["mask"])
        end = dataset.snapshot()
        entry_fields = decode_reference_cursor(entry)
        end_fields = decode_reference_cursor(end)
        if (entry_fields.total_rows != total_rows
                or entry_fields.next_ordinal != skip
                or end_fields.next_ordinal != total_rows
                or emitted != total_rows - skip):
            raise AssertionError("D2 cursor or emitted row count differs")
        replay = ReferenceDataset(loaded, config, manifest_digest=digest,
                                  tokenizer_fingerprint=FINGERPRINT,
                                  vocab_size=VOCAB_SIZE)
        replay.seek(entry)
        if replay.snapshot() != entry:
            raise AssertionError("D2 entry cursor does not replay")
        if [_row(batch) for batch in iter(replay.next_batch, None)] != rows:
            raise AssertionError("D2 entry cursor replay differs from emitted rows")
        if replay.snapshot() != end:
            raise AssertionError("D2 end cursor differs after replay")
        files = [{"path": path.name, "bytes": path.stat().st_size,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                 for path in sorted(directory.iterdir())]
        cases.append({"name": name, "shard_tokens": list(map(len, loaded)),
                      "total_rows": total_rows, "skip": skip,
                      "remaining_rows": emitted, "active_tokens": active,
                      "first_row": first, "last_row": last, "rows": rows,
                      "entry_cursor": entry.hex(), "end_cursor": end.hex(),
                      "manifest_digest": digest.hex(), "files": files})
    receipt = {"kind": "e3-n5-d2-preflight-v1", "tokenizer_fingerprint": FINGERPRINT,
               "tokenizer_sha256": hashlib.sha256(tokenizer).hexdigest(),
               "cases": cases}
    (root / "preflight.json").write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.output)
    print(args.output / "preflight.json")


if __name__ == "__main__":
    main()
