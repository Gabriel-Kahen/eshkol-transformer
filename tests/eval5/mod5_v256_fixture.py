"""Compiler-free, data-only T1/D1/C1 fixture for MOD5 EVAL5 development."""

import hashlib
import json
import math
from pathlib import Path
import struct
import sys

from tests.d1.reference_format import parse_corpus
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.t1.reference import (Identity, decode_artifact, encode_artifact,
                                encode_bytes, fingerprint)
from tests.eval5 import mod5_reference_oracle as oracle

SOURCE = {
    "train": (bytes((1, 4, 1, 5, 9, 2, 6, 5, 3)),
              bytes((2, 7, 1, 8, 2, 8, 1, 8, 2))),
    "eval-clean": (bytes((200, 3, 10, 11, 12, 13, 14, 15, 201)),
                   bytes((201, 15, 14, 13, 12, 11, 10, 3, 200))),
}
IDENTITY = Identity()
FINGERPRINT = fingerprint(IDENTITY)
SHARD = "shard-{index:016d}.ets"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("ascii")


def model_state():
    tensors = []
    for p, (name, shape) in enumerate(zip(oracle.PATHS, oracle.SHAPES_256, strict=True)):
        count = math.prod(shape)
        base = 1.0 if len(shape) == 1 else 0.0
        values = [base + ((29 * p + 11 * i + i // 5) % 37 - 18) / 128
                  for i in range(count)]
        tensors.append(oracle.safe.Tensor(tuple(name.split(".")), "parameter", shape,
                                          struct.pack("<" + "f" * count, *values)))
    return oracle.safe.State("i2-dense-cpu-f32-v1", tuple(tensors))


def split_record(directory, name, write=False):
    docs = SOURCE[name]
    tokens = tuple(encode_bytes(IDENTITY, b"".join(docs)))
    if write:
        directory.mkdir()
        for index, doc in enumerate(docs):
            (directory / f"doc-{index}.bin").write_bytes(doc)
        write_d1_resource(directory, (tokens[:6], tokens[6:12], tokens[12:]),
                          fingerprint=FINGERPRINT, vocab=256)
    if tuple((directory / f"doc-{i}.bin").read_bytes() for i in range(2)) != docs:
        raise ValueError(f"{name}: original documents differ from fixture")
    loaded, trailer = load_d1_resource(
        directory, expected_fingerprint=FINGERPRINT, expected_vocab=256,
        maximum_manifest_bytes=1024, maximum_shard_bytes=1024,
        maximum_total_tokens=18)
    parsed = parse_corpus(directory)
    if loaded != (tokens[:6], tokens[6:12], tokens[12:]) or tuple(parsed["tokens"]) != tokens:
        raise ValueError(f"{name}: D1 tokens differ from T1 encoding of original bytes")
    manifest = (directory / "manifest.etm").read_bytes()
    if trailer != manifest[-32:]:
        raise ValueError(f"{name}: D1 manifest trailer mismatch")
    return {"dataset_sha256": sha(manifest),
            "manifest_trailer_sha256": trailer.hex(),
            "ordered_shard_sha256": [sha((directory / SHARD.format(index=i)).read_bytes())
                                     for i in range(3)],
            "ordered_document_sha256": [sha(doc) for doc in docs],
            "original_byte_count": sum(map(len, docs))}


def expected_record(root, *, write=False):
    artifact = encode_artifact(IDENTITY)
    model = oracle.safe.encode_c1(model_state())
    if write:
        root.mkdir()
        (root / "tokenizer.etok").write_bytes(artifact)
        (root / "model.etcp").write_bytes(model)
    actual_artifact = (root / "tokenizer.etok").read_bytes()
    parsed = decode_artifact(actual_artifact)
    if actual_artifact != artifact or parsed.identity != IDENTITY or parsed.fingerprint != FINGERPRINT:
        raise ValueError("T1 artifact is not the canonical raw byte tokenizer")
    actual_model = (root / "model.etcp").read_bytes()
    weights = oracle.weights_from_c1(actual_model)
    if actual_model != model:
        raise ValueError("C1 model differs from deterministic V=256 fixture")
    splits = {name: split_record(root / name, name, write=write) for name in SOURCE}
    train = tuple(tuple(doc) for doc in SOURCE["train"])
    evaluation = tuple(tuple(doc) for doc in SOURCE["eval-clean"])
    score = oracle.evaluate(weights, train, evaluation, SOURCE["train"],
                            SOURCE["eval-clean"], byte_count=16)
    if score["status"] != "clean":
        raise ValueError("held-out fixture failed contamination gate")
    return {"version": 1, "scope": "development-reference-only",
            "tokenizer_fingerprint": FINGERPRINT,
            "tokenizer_sha256": FINGERPRINT.rsplit(":", 1)[1],
            "tokenizer_artifact_sha256": sha(actual_artifact),
            "checkpoint_sha256": sha(actual_model),
            "splits": splits, "score_reference": score}


def build(root):
    if root.exists() or root.is_symlink():
        raise FileExistsError("fixture destination must be fresh")
    record = expected_record(root, write=True)
    (root / "fixture.json").write_bytes(canonical(record))
    return record


def verify(root):
    raw = (root / "fixture.json").read_bytes()
    record = json.loads(raw)
    if raw != canonical(record) or record != expected_record(root):
        raise ValueError("fixture provenance or reference score mismatch")
    return record


def main(argv):
    if len(argv) != 3 or argv[1] not in ("build", "verify"):
        raise SystemExit("usage: python3 -m tests.eval5.mod5_v256_fixture build|verify DIR")
    result = build(Path(argv[2])) if argv[1] == "build" else verify(Path(argv[2]))
    print(canonical(result).decode("ascii"), end="")


if __name__ == "__main__":
    main(sys.argv)
