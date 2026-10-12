"""Versioned, checksummed D1/D2 edge cases for the private E3 test route."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from tests.d2.reference import ReferenceConfig, ReferenceDataset, decode_reference_cursor
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_reference.corpus import FINGERPRINT, IDENTITY, _reject_symlink_ancestors
from tests.e3_reference.transcript import canonical
from tests.t1.reference import encode_artifact


KIND = "e3-edge-corpus-v1"
CONFIG = ReferenceConfig(1, 2, None, 1, True)
CASES = (
    ("empty", (), 0, "invalid-state"),
    ("singleton", ((7,),), 0, "invalid-state"),
    ("entry-eos", ((7, 8),), 1, "invalid-state"),
    ("one-valid-final", ((3, 197, 0, 255),), 0, "success"),
)


def _rows(dataset: ReferenceDataset) -> list[dict[str, object]]:
    result = []
    while (batch := dataset.next_batch()) is not None:
        result.append({"inputs": list(batch.inputs[0]),
                       "targets": list(batch.targets[0]),
                       "mask": list(batch.loss_mask[0])})
    return result


def _case(directory: Path, name: str, shards: tuple[tuple[int, ...], ...],
          start: int, outcome: str) -> dict[str, object]:
    loaded, digest = load_d1_resource(
        directory, expected_fingerprint=FINGERPRINT, expected_vocab=256,
        maximum_manifest_bytes=1048576, maximum_shard_bytes=1048576,
        maximum_total_tokens=65536,
    )
    if loaded != shards:
        raise ValueError(f"{name}: D1 shard contents changed")
    expected_names = ["manifest.etm"] + [
        f"shard-{index:016d}.ets" for index in range(len(shards))]
    if sorted(path.name for path in directory.iterdir()) != sorted(expected_names):
        raise ValueError(f"{name}: D1 member inventory changed")
    dataset = ReferenceDataset(loaded, CONFIG, manifest_digest=digest,
                               tokenizer_fingerprint=FINGERPRINT, vocab_size=256)
    if start > len(dataset.rows):
        raise ValueError(f"{name}: start cursor past EOS")
    for _ in range(start):
        dataset.next_batch()
    cursor = dataset.snapshot().hex()
    batches = _rows(dataset)
    traversal_end = dataset.snapshot().hex()
    files = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{name}: unexpected D1 member")
        raw = path.read_bytes()
        files.append({"path": path.name, "bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest()})
    return {"name": name, "shard_token_counts": [len(s) for s in shards],
            "start_ordinal": start, "total_rows": len(dataset.rows),
            "outcome": outcome, "start_cursor": cursor,
            "traversal_end_cursor": traversal_end, "batches": batches,
            "manifest_sha256": hashlib.sha256((directory / "manifest.etm").read_bytes()).hexdigest(),
            "files": files}


def validate(root: Path, payload: object) -> None:
    if not isinstance(payload, dict) or set(payload) != {"kind", "tokenizer_sha256", "cases"}:
        raise ValueError("edge corpus schema mismatch")
    if payload["kind"] != KIND or type(payload["tokenizer_sha256"]) is not str:
        raise ValueError("edge corpus identity mismatch")
    artifact = (root / "tokenizer.etok").read_bytes()
    if artifact != encode_artifact(IDENTITY) or hashlib.sha256(artifact).hexdigest() != payload["tokenizer_sha256"]:
        raise ValueError("edge tokenizer identity mismatch")
    cases = payload["cases"]
    if type(cases) is not list or len(cases) != len(CASES):
        raise ValueError("edge case inventory mismatch")
    for actual, (name, shards, start, outcome) in zip(cases, CASES):
        if not isinstance(actual, dict):
            raise ValueError("edge case schema mismatch")
        expected = _case(root / name, name, shards, start, outcome)
        if set(actual) != set(expected):
            raise ValueError(f"{name}: edge case schema mismatch")
        # Equality alone treats True as 1; canonical bytes retain the exact JSON types.
        if canonical(actual) != canonical(expected):
            raise ValueError(f"{name}: D1/D2, manifest or cursor mismatch")
        fields = decode_reference_cursor(bytes.fromhex(actual["start_cursor"]))
        if fields.next_ordinal != start or fields.total_rows != expected["total_rows"]:
            raise ValueError(f"{name}: cursor position mismatch")


def materialize(root: Path) -> dict[str, object]:
    _reject_symlink_ancestors(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists() or root.is_symlink():
        raise FileExistsError("edge corpus output must be fresh")
    root.mkdir()
    artifact = encode_artifact(IDENTITY)
    (root / "tokenizer.etok").write_bytes(artifact)
    cases = []
    for name, shards, start, outcome in CASES:
        directory = root / name
        directory.mkdir()
        write_d1_resource(directory, shards, fingerprint=FINGERPRINT, vocab=256)
        cases.append(_case(directory, name, shards, start, outcome))
    result = {"kind": KIND, "tokenizer_sha256": hashlib.sha256(artifact).hexdigest(),
              "cases": cases}
    validate(root, result)
    (root / "edge-corpus.json").write_bytes(canonical(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    materialize(args.output)
