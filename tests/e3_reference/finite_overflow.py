"""Deterministic development-only C1/D1 fixture for a real E3 finalizer cut."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

from tests.c1.test_checkpoint_format import build_container, parse_golden
from tests.d1.reference_format import parse_corpus
from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_reference.corpus import FINGERPRINT, IDENTITY, VOCAB_SIZE
from tests.e3_reference.uniform_tie import ALIAS, PROVIDER, SHAPES
from tests.t1.reference import encode_artifact, encode_bytes


RAW = bytes.fromhex("0304")
GAMMAS = {"finite": 0x41B10000, "overflow": 0x41B20000}
HEAD = (0x3F800000, 0xBF800000, 0x3F800000, 0xBF800000)
PROFILE = ReferenceConfig(1, 2, None, 1, True)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parameter_words(path: str, shape: tuple[int, ...], gamma: int) -> tuple[int, ...]:
    count = 1
    for extent in shape:
        count *= extent
    words = [0] * count
    if path in ("head/weight", "token_embedding/weight"):
        if shape != (256, 4):
            raise ValueError("tied head geometry changed")
        words[12:16] = HEAD
    elif path == "norm_final/weight":
        if shape != (4,):
            raise ValueError("final norm geometry changed")
        words[:] = [gamma] * 4
    return tuple(words)


def checkpoint(gamma: int) -> bytes:
    if gamma not in GAMMAS.values() or len(SHAPES) != 15 or ALIAS != (10, 14):
        raise ValueError("finite overflow fixture topology changed")
    entries = []
    for path, shape in SHAPES:
        words = parameter_words(path, shape, gamma)
        payload = struct.pack("<" + "I" * len(words), *words)
        entries.append((tuple(part.encode("ascii") for part in path.split("/")),
                        1, 3, shape, payload))
    if entries[10][-1] != entries[14][-1]:
        raise AssertionError("tied checkpoint payloads diverged")
    encoded = build_container(tuple(entries), (ALIAS,), PROVIDER)
    parsed = parse_golden(encoded, PROVIDER)
    if len(parsed.records) != 15 or len(parsed.aliases) != 1 or \
            tuple(parsed.aliases[0].members) != ALIAS:
        raise AssertionError("C1 tie topology changed")
    for row, expected in zip(parsed.records, entries, strict=True):
        payload = encoded[row.payload_start:row.payload_start + row.payload_bytes]
        if payload != expected[-1]:
            raise AssertionError("C1 payload differed from exact parameter bits")
    return encoded


def _regular_tree(root: Path) -> None:
    if root.is_symlink() or root.resolve() != root.absolute():
        raise ValueError("fixture path is not canonical")
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("fixture contains nonregular path")


def materialize(root: Path) -> dict:
    if root.exists() or root.is_symlink() or root.parent.resolve() != root.parent.absolute():
        raise ValueError("fixture destination must be fresh and canonical")
    if tuple(encode_bytes(IDENTITY, RAW)) != tuple(RAW):
        raise AssertionError("T1 raw-byte identity changed")
    root.mkdir()
    tokenizer = encode_artifact(IDENTITY)
    (root / "tokenizer.etok").write_bytes(tokenizer)
    files = {f"{name}.etcp": checkpoint(gamma) for name, gamma in GAMMAS.items()}
    for name, data in files.items():
        (root / name).write_bytes(data)
    corpus = root / "finite-overflow"
    corpus.mkdir()
    manifest = write_d1_resource(corpus, (tuple(RAW),),
                                 fingerprint=FINGERPRINT, vocab=VOCAB_SIZE)
    loaded, manifest_digest = load_d1_resource(
        corpus, expected_fingerprint=FINGERPRINT, expected_vocab=VOCAB_SIZE,
        maximum_total_tokens=2)
    if tuple(parse_corpus(corpus)["tokens"]) != tuple(RAW):
        raise AssertionError("D1 raw tokens changed")
    dataset = ReferenceDataset(loaded, PROFILE, manifest_digest=manifest_digest,
                               tokenizer_fingerprint=FINGERPRINT,
                               vocab_size=VOCAB_SIZE)
    start = dataset.snapshot().hex()
    row = dataset.next_batch()
    if row is None or dataset.next_batch() is not None or \
            list(row.inputs[0]) != [3, 0] or list(row.targets[0]) != [4, 0] or \
            list(row.loss_mask[0]) != [True, False]:
        raise AssertionError("authentic D2 padded row changed")
    payload = {
        "kind": "e3-finite-overflow-fixture-v1", "raw": RAW.hex(),
        "tokenizer_sha256": digest(tokenizer), "provider": PROVIDER.decode(),
        "alias_indices": list(ALIAS), "entry_count": 15,
        "checkpoints": {name: {"sha256": digest(data), "bytes": len(data),
                               "gamma_word": f"{GAMMAS[name.removesuffix('.etcp')]:08x}"}
                        for name, data in files.items()},
        "corpus": {"name": corpus.name, "manifest_sha256": digest(manifest),
                   "start_cursor": start, "end_cursor": dataset.snapshot().hex(),
                   "files": {p.name: digest(p.read_bytes()) for p in sorted(corpus.iterdir())}},
    }
    (root / "fixture.json").write_text(json.dumps(payload, sort_keys=True,
                                                  separators=(",", ":")) + "\n")
    validate(root)
    return payload


def validate(root: Path) -> dict:
    _regular_tree(root)
    payload = json.loads((root / "fixture.json").read_text())
    if set(payload) != {"kind", "raw", "tokenizer_sha256", "provider",
                        "alias_indices", "entry_count", "checkpoints", "corpus"} or \
            payload["kind"] != "e3-finite-overflow-fixture-v1" or \
            payload["raw"] != RAW.hex() or payload["provider"] != PROVIDER.decode() or \
            payload["alias_indices"] != list(ALIAS) or payload["entry_count"] != 15:
        raise ValueError("fixture manifest contract changed")
    if payload["tokenizer_sha256"] != digest((root / "tokenizer.etok").read_bytes()) or \
            (root / "tokenizer.etok").read_bytes() != encode_artifact(IDENTITY):
        raise ValueError("tokenizer mismatch")
    if set(payload["checkpoints"]) != {"finite.etcp", "overflow.etcp"}:
        raise ValueError("checkpoint membership mismatch")
    for name, gamma in GAMMAS.items():
        path = root / f"{name}.etcp"
        expected = checkpoint(gamma)
        if path.read_bytes() != expected or payload["checkpoints"][f"{name}.etcp"] != {
                "sha256": digest(expected), "bytes": len(expected),
                "gamma_word": f"{gamma:08x}"}:
            raise ValueError("C1 state mismatch")
    corpus = root / "finite-overflow"
    if set(payload["corpus"]) != {"name", "manifest_sha256", "start_cursor",
                                  "end_cursor", "files"} or \
            payload["corpus"]["name"] != corpus.name:
        raise ValueError("corpus path mismatch")
    observed = {p.name: digest(p.read_bytes()) for p in sorted(corpus.iterdir())}
    if set(observed) != {"manifest.etm", "shard-0000000000000000.ets"} or \
            payload["corpus"]["files"] != observed:
        raise ValueError("corpus checksum mismatch")
    loaded, manifest_digest = load_d1_resource(
        corpus, expected_fingerprint=FINGERPRINT, expected_vocab=VOCAB_SIZE,
        maximum_total_tokens=2)
    dataset = ReferenceDataset(loaded, PROFILE, manifest_digest=manifest_digest,
                               tokenizer_fingerprint=FINGERPRINT,
                               vocab_size=VOCAB_SIZE)
    start = dataset.snapshot().hex()
    row = dataset.next_batch()
    if row is None or dataset.next_batch() is not None or \
            list(row.inputs[0]) != [3, 0] or list(row.targets[0]) != [4, 0] or \
            list(row.loss_mask[0]) != [True, False] or \
            payload["corpus"]["start_cursor"] != start or \
            payload["corpus"]["end_cursor"] != dataset.snapshot().hex() or \
            payload["corpus"]["manifest_sha256"] != digest((corpus / "manifest.etm").read_bytes()):
        raise ValueError("D2 row or cursor mismatch")
    listed = {"tokenizer.etok", "finite.etcp", "overflow.etcp", "fixture.json",
              "finite-overflow"}
    if {p.name for p in root.iterdir()} != listed:
        raise ValueError("fixture has extra or missing members")
    return payload


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m tests.e3_reference.finite_overflow FRESH_DIR")
    print(json.dumps(materialize(Path(sys.argv[1])), sort_keys=True,
                     separators=(",", ":")))
