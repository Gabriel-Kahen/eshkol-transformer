"""Generate tiny authentic T1/D1 resources and independent D2 emissions."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from tests.d1.reference_format import parse_corpus
from tests.d2.reference import ReferenceConfig, ReferenceDataset, decode_reference_cursor
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_reference.transcript import encode
from tests.t1.reference import Identity, encode_artifact, encode_bytes, fingerprint


SOURCE_BYTES = bytes((3, 197, 0, 255, 7, 7))
IDENTITY = Identity("raw", (), (), ())
FINGERPRINT = fingerprint(IDENTITY)
VOCAB_SIZE = 256
PROFILE = "e3-private-n1-t2-v256-d4-v1"
CASES = (
    ("packed-single", ((3, 197, 0, 255, 7, 7),), True, 0),
    ("packed-sharded", ((3, 197), (0, 255), (7, 7)), True, 0),
    ("packed-suffix", ((3, 197), (0, 255), (7, 7)), True, 1),
    ("unpacked-three", ((3, 197, 0), (255, 7, 7)), False, 0),
    ("unpacked-pairs", ((3, 197), (0, 255), (7, 7)), False, 0),
)
MAX_CASES = len(CASES)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _batch(batch) -> dict[str, object]:
    return {
        "inputs": list(batch.inputs[0]),
        "mask": list(batch.loss_mask[0]),
        "targets": list(batch.targets[0]),
    }


def _exact_int(value: object, where: str, minimum: int = 0, maximum: int = (1 << 63) - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{where}: expected exact integer in range")
    return value


def _digest_word(value: object, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        ch not in "0123456789abcdef" for ch in value
    ):
        raise ValueError(f"{where}: expected lowercase SHA-256")
    return value


def _reject_symlink_ancestors(path: Path) -> None:
    ancestor = path.parent
    while not ancestor.exists():
        if ancestor.is_symlink():
            raise ValueError("E3 reference output ancestors must not be symlinks")
        if ancestor == ancestor.parent:
            break
        ancestor = ancestor.parent
    if ancestor.is_symlink() or ancestor.resolve() != ancestor.absolute():
        raise ValueError("E3 reference output ancestors must not be symlinks")


def validate_payload(payload: dict[str, object]) -> None:
    """Admit only the fixed corpus records this generator can produce."""

    if not isinstance(payload, dict) or set(payload) != {
        "cases", "kind", "profile", "source_bytes", "tokenizer"
    }:
        raise ValueError("corpus reference fields differ from version 1")
    if payload["kind"] != "corpus-reference" or payload["profile"] != PROFILE:
        raise ValueError("corpus reference kind/profile mismatch")
    if payload["source_bytes"] != SOURCE_BYTES.hex():
        raise ValueError("corpus reference bytes differ from the fixed case")
    tokenizer = payload["tokenizer"]
    if not isinstance(tokenizer, dict) or set(tokenizer) != {
        "artifact_sha256", "fingerprint", "specials", "utf8_policy", "vocab_size"
    }:
        raise ValueError("tokenizer record fields differ from version 1")
    if tokenizer != {
        "artifact_sha256": hashlib.sha256(encode_artifact(IDENTITY)).hexdigest(),
        "fingerprint": FINGERPRINT,
        "specials": 0,
        "utf8_policy": "raw",
        "vocab_size": VOCAB_SIZE,
    }:
        raise ValueError("tokenizer record differs from canonical raw V256 identity")
    cases = payload["cases"]
    if not isinstance(cases, list) or len(cases) != MAX_CASES:
        raise ValueError("corpus reference case count mismatch")
    for case, (name, shards, packing, start_ordinal) in zip(cases, CASES):
        if not isinstance(case, dict) or set(case) != {
            "batches", "corpus", "name", "packed", "start_cursor", "start_ordinal",
            "total_rows",
        }:
            raise ValueError(f"{name}: corpus case fields differ from version 1")
        if case["name"] != name or type(case["packed"]) is not bool or case["packed"] != packing:
            raise ValueError(f"{name}: case identity mismatch")
        if _exact_int(case["start_ordinal"], name + ".start_ordinal") != start_ordinal:
            raise ValueError(f"{name}: start ordinal mismatch")
        expected_rows = ReferenceDataset(
            shards,
            ReferenceConfig(1, 2, None, 1, packing),
            manifest_digest=bytes(32),
            tokenizer_fingerprint=FINGERPRINT,
            vocab_size=VOCAB_SIZE,
        ).rows
        if _exact_int(case["total_rows"], name + ".total_rows") != len(expected_rows):
            raise ValueError(f"{name}: row count mismatch")
        expected_batches = [
            {"inputs": list(row.inputs), "mask": list(row.loss_mask), "targets": list(row.targets)}
            for row in expected_rows[start_ordinal:]
        ]
        batches = case["batches"]
        if not isinstance(batches, list) or len(batches) != len(expected_batches):
            raise ValueError(f"{name}: batch count mismatch")
        for index, (batch, expected) in enumerate(zip(batches, expected_batches)):
            if not isinstance(batch, dict) or set(batch) != {"inputs", "mask", "targets"}:
                raise ValueError(f"{name}.batches[{index}]: fields differ from version 1")
            for field in ("inputs", "targets"):
                values = batch[field]
                if not isinstance(values, list) or len(values) != 2:
                    raise ValueError(f"{name}.batches[{index}].{field}: expected length 2")
                normalized = [
                    _exact_int(value, f"{name}.batches[{index}].{field}", 0, 255)
                    for value in values
                ]
                if normalized != expected[field]:
                    raise ValueError(f"{name}: batches differ from independent D2 row equations")
            mask = batch["mask"]
            if not isinstance(mask, list) or len(mask) != 2 or any(type(value) is not bool for value in mask):
                raise ValueError(f"{name}.batches[{index}].mask: expected two Booleans")
            if mask != expected["mask"]:
                raise ValueError(f"{name}: batches differ from independent D2 row equations")
        cursor = case["start_cursor"]
        if not isinstance(cursor, str) or len(cursor) > 800 or len(cursor) % 2 or any(
            ch not in "0123456789abcdef" for ch in cursor
        ):
            raise ValueError(f"{name}: start cursor is not canonical hex")
        fields = decode_reference_cursor(bytes.fromhex(cursor))
        if (
            fields.next_ordinal != start_ordinal
            or fields.total_rows != len(expected_rows)
            or fields.tokenizer_fingerprint != FINGERPRINT
            or fields.vocab_size != VOCAB_SIZE
            or fields.config != ReferenceConfig(1, 2, None, 1, packing)
        ):
            raise ValueError(f"{name}: start cursor identity or position mismatch")
        corpus = case["corpus"]
        if not isinstance(corpus, dict) or set(corpus) != {
            "files", "manifest_trailer_sha256", "shard_token_counts"
        }:
            raise ValueError(f"{name}: physical corpus fields differ from version 1")
        if corpus["shard_token_counts"] != [len(shard) for shard in shards]:
            raise ValueError(f"{name}: shard counts mismatch")
        trailer = _digest_word(
            corpus["manifest_trailer_sha256"], name + ".manifest_trailer_sha256"
        )
        if fields.manifest_digest.hex() != trailer:
            raise ValueError(f"{name}: cursor and manifest digest mismatch")
        files = corpus["files"]
        if not isinstance(files, list) or len(files) != len(shards) + 1:
            raise ValueError(f"{name}: physical file count mismatch")
        expected_names = ["manifest.etm"] + [
            f"shard-{index:016d}.ets" for index in range(len(shards))
        ]
        if [item.get("path") if isinstance(item, dict) else None for item in files] != expected_names:
            raise ValueError(f"{name}: physical filenames mismatch")
        for item in files:
            if not isinstance(item, dict) or set(item) != {"bytes", "path", "sha256"}:
                raise ValueError(f"{name}: physical file record mismatch")
            _exact_int(item["bytes"], name + ".file.bytes", 1, 1_048_576)
            if not isinstance(item["path"], str) or "/" in item["path"] or "\\" in item["path"]:
                raise ValueError(f"{name}: file path is not relative basename")
            _digest_word(item["sha256"], name + ".file.sha256")


def materialize(output: Path) -> dict[str, object]:
    _reject_symlink_ancestors(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError("E3 reference output must be a fresh path")
    output.mkdir()
    artifact = encode_artifact(IDENTITY)
    if encode_bytes(IDENTITY, SOURCE_BYTES) != tuple(SOURCE_BYTES):
        raise AssertionError("raw/empty tokenizer changed source byte IDs")
    (output / "tokenizer.etok").write_bytes(artifact)
    cases: list[dict[str, object]] = []
    for name, shards, packing, start_ordinal in CASES:
        directory = output / name
        directory.mkdir()
        manifest = write_d1_resource(
            directory, shards, fingerprint=FINGERPRINT, vocab=VOCAB_SIZE
        )
        loaded, digest = load_d1_resource(
            directory,
            expected_fingerprint=FINGERPRINT,
            expected_vocab=VOCAB_SIZE,
            maximum_total_tokens=len(SOURCE_BYTES),
        )
        parsed = parse_corpus(directory)
        if tuple(parsed["tokens"]) != tuple(SOURCE_BYTES):
            raise AssertionError("D1 resource does not contain the canonical byte IDs")
        config = ReferenceConfig(1, 2, None, 1, packing)
        dataset = ReferenceDataset(
            loaded,
            config,
            manifest_digest=digest,
            tokenizer_fingerprint=FINGERPRINT,
            vocab_size=VOCAB_SIZE,
        )
        for _ in range(start_ordinal):
            if dataset.next_batch() is None:
                raise AssertionError("start ordinal exceeds the D2 row count")
        start_cursor = dataset.snapshot()
        batches = []
        while (batch := dataset.next_batch()) is not None:
            batches.append(_batch(batch))
        end_cursor = dataset.snapshot()
        replay = ReferenceDataset(
            loaded,
            config,
            manifest_digest=digest,
            tokenizer_fingerprint=FINGERPRINT,
            vocab_size=VOCAB_SIZE,
        )
        replay.seek(start_cursor)
        replay_batches = []
        while (batch := replay.next_batch()) is not None:
            replay_batches.append(_batch(batch))
        if replay_batches != batches or replay.snapshot() != end_cursor:
            raise AssertionError("nonzero D2 cursor replay differs from suffix traversal")
        files = []
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            files.append(
                {"bytes": path.stat().st_size, "path": path.name, "sha256": _sha256(path)}
            )
        cases.append(
            {
                "batches": batches,
                "corpus": {
                    "files": files,
                    "manifest_trailer_sha256": manifest[-32:].hex(),
                    "shard_token_counts": [len(shard) for shard in shards],
                },
                "name": name,
                "packed": packing,
                "start_cursor": start_cursor.hex(),
                "start_ordinal": start_ordinal,
                "total_rows": len(dataset.rows),
            }
        )
    payload = {
        "cases": cases,
        "kind": "corpus-reference",
        "profile": PROFILE,
        "source_bytes": SOURCE_BYTES.hex(),
        "tokenizer": {
            "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
            "fingerprint": FINGERPRINT,
            "specials": 0,
            "utf8_policy": "raw",
            "vocab_size": VOCAB_SIZE,
        },
    }
    validate_payload(payload)
    (output / "corpus-reference.json").write_bytes(encode(payload))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    materialize(arguments.output)


if __name__ == "__main__":
    main()
