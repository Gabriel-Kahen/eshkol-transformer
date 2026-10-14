"""Frozen five-case input and cursor identity for E3 section 9 source gates.

This development-only sidecar is derived before any native observation. It does
not extend the accepted v1 corpus, reference-bundle, or observed schemas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource
from tests.e3_reference.corpus import CASES, FINGERPRINT, SOURCE_BYTES, VOCAB_SIZE
from tests.e3_reference.corpus import validate_payload as validate_corpus
from tests.e3_reference.reference import _read_limited, _write_exclusive, validate_bundle
from tests.e3_reference.transcript import canonical, decode as decode_reference
from tests.e3_reference.transcript import encode as encode_reference
from tests.e3_reference.transcript import validate_observed_case


FORMAT = "eshkol-e3-section9-case-manifest"
VERSION = 1
MAX_BYTES = 65_536


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _row(batch: object) -> dict[str, object]:
    return {
        "inputs": list(batch.inputs[0]),
        "mask": list(batch.loss_mask[0]),
        "targets": list(batch.targets[0]),
    }


def _physical_files(directory: Path, recorded: list[dict[str, object]]) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("case directory is absent or a symlink")
    if sorted(path.name for path in directory.iterdir()) != sorted(item["path"] for item in recorded):
        raise ValueError("D1 physical file set differs from corpus reference")
    for item in recorded:
        path = directory / item["path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError("D1 physical file is absent or a symlink")
        if path.stat().st_size != item["bytes"] or _sha(path.read_bytes()) != item["sha256"]:
            raise ValueError("D1 physical file length or SHA-256 differs from corpus reference")


def expected_payload(
    corpus: dict[str, object], bundle: dict[str, object], directory: Path,
) -> dict[str, object]:
    """Reconstruct every D2 row and both cursor positions from physical D1 bytes."""

    validate_corpus(corpus)
    validate_bundle(bundle)
    if [item["name"] for item in bundle["cases"]] != [name for name, *_ in CASES]:
        raise ValueError("reference bundle case order differs from the fixed five cases")
    if canonical([item["corpus"] for item in bundle["cases"]]) != canonical(corpus["cases"]):
        raise ValueError("reference bundle and corpus records differ")
    if directory.is_symlink() or directory.resolve() != directory.absolute() or not directory.is_dir():
        raise ValueError("corpus directory is absent or a symlink")
    artifact = directory / "tokenizer.etok"
    if artifact.is_symlink() or not artifact.is_file():
        raise ValueError("T1 artifact is absent or a symlink")
    if artifact.stat().st_size > 1_048_576:
        raise ValueError("T1 artifact exceeds the one-MiB development limit")
    if _sha(artifact.read_bytes()) != corpus["tokenizer"]["artifact_sha256"]:
        raise ValueError("T1 artifact SHA-256 differs from corpus reference")

    cases = []
    for source, (name, shards, packing, start_ordinal) in zip(corpus["cases"], CASES):
        if source["name"] != name:
            raise ValueError("corpus case order differs from the fixed five cases")
        case_directory = directory / name
        files = source["corpus"]["files"]
        _physical_files(case_directory, files)
        loaded, digest = load_d1_resource(
            case_directory, expected_fingerprint=FINGERPRINT,
            expected_vocab=VOCAB_SIZE, maximum_manifest_bytes=1_048_576,
            maximum_shard_bytes=1_048_576, maximum_total_tokens=len(SOURCE_BYTES),
        )
        if loaded != shards or digest.hex() != source["corpus"]["manifest_trailer_sha256"]:
            raise ValueError(f"{name}: physical D1 tokens or manifest trailer differ")
        config = ReferenceConfig(1, 2, None, 1, packing)
        traversal = ReferenceDataset(
            loaded, config, manifest_digest=digest,
            tokenizer_fingerprint=FINGERPRINT, vocab_size=VOCAB_SIZE,
        )
        for _ in range(start_ordinal):
            if traversal.next_batch() is None:
                raise ValueError(f"{name}: start ordinal exceeds D2 rows")
        start = traversal.snapshot()
        rows = []
        while (batch := traversal.next_batch()) is not None:
            rows.append(_row(batch))
        end = traversal.snapshot()
        replay = ReferenceDataset(
            loaded, config, manifest_digest=digest,
            tokenizer_fingerprint=FINGERPRINT, vocab_size=VOCAB_SIZE,
        )
        replay.seek(start)
        replay_rows = []
        while (batch := replay.next_batch()) is not None:
            replay_rows.append(_row(batch))
        if replay_rows != rows or replay.snapshot() != end:
            raise ValueError(f"{name}: D2 cursor replay differs")
        if source["start_cursor"] != start.hex() or source["batches"] != rows:
            raise ValueError(f"{name}: frozen start cursor or rows differ from independent D2")
        cases.append({
            "case_id": name,
            "d1": {
                "files": files,
                "manifest_trailer_sha256": digest.hex(),
            },
            "d2": {
                "batch_size": 1, "sequence_length": 2, "shuffle_seed": None,
                "shuffle_window_rows": 1, "packed": packing,
                "start_cursor": start.hex(), "end_cursor": end.hex(),
                "start_ordinal": start_ordinal,
            },
            "expected_batch_count": len(rows),
            "rows": rows,
        })
    return {
        "kind": "section9-case-identity",
        "version": VERSION,
        "corpus_reference_sha256": _sha(encode_reference(corpus)),
        "reference_bundle_sha256": _sha(encode_reference(bundle)),
        "model": bundle["model"],
        "t1": corpus["tokenizer"],
        "cases": cases,
    }


def encode(payload: dict[str, object]) -> bytes:
    body = canonical(payload)
    raw = canonical({
        "format": FORMAT, "version": VERSION, "payload": payload,
        "payload_sha256": _sha(body),
    })
    if len(raw) > MAX_BYTES:
        raise ValueError("section 9 case manifest exceeds 65536 bytes")
    return raw


def decode(
    raw: bytes, corpus: dict[str, object], bundle: dict[str, object], directory: Path,
) -> dict[str, object]:
    """Reject malformed sidecars before any observed transcript is inspected."""

    if not isinstance(raw, bytes) or not 1 <= len(raw) <= MAX_BYTES:
        raise ValueError("section 9 case manifest exceeds 65536 bytes")
    try:
        document = json.loads(raw, object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("section 9 case manifest is not JSON") from error
    if not isinstance(document, dict) or set(document) != {
        "format", "version", "payload", "payload_sha256",
    }:
        raise ValueError("section 9 case manifest envelope fields differ")
    if document["format"] != FORMAT or type(document["version"]) is not int or document["version"] != VERSION:
        raise ValueError("section 9 case manifest format or version differs")
    payload = document["payload"]
    digest = document["payload_sha256"]
    if not isinstance(payload, dict) or not isinstance(digest, str) or digest != _sha(canonical(payload)):
        raise ValueError("section 9 case manifest checksum differs")
    if encode(payload) != raw:
        raise ValueError("section 9 case manifest JSON is not canonical")
    if canonical(payload) != canonical(expected_payload(corpus, bundle, directory)):
        raise ValueError("section 9 case manifest fields or identity differ")
    return payload


def accept_observed(
    raw: bytes, corpus: dict[str, object], bundle: dict[str, object],
    directory: Path, case_id: str, observed: dict[str, object],
) -> None:
    manifest = decode(raw, corpus, bundle, directory)
    matches = [case for case in manifest["cases"] if case["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError("unknown section 9 case ID")
    validate_observed_case(observed, tuple(matches[0]["rows"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--reference-bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    corpus = decode_reference(_read_limited(arguments.corpus_dir / "corpus-reference.json"))
    bundle = decode_reference(_read_limited(arguments.reference_bundle))
    _write_exclusive(
        arguments.output,
        encode(expected_payload(corpus, bundle, arguments.corpus_dir)),
    )


if __name__ == "__main__":
    main()
