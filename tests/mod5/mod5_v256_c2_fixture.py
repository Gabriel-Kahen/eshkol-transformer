"""Generate data and independent cursor expectations for the MOD5 V256 C2 gate."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

from tests.d2.reference import BATCH_ELEMENT_BYTES, ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.t1.reference import Identity, decode_artifact, encode_artifact, encode_bytes, fingerprint
from tests.x1 import manifest_format as x1


CONTEXT = 8
VOCAB = 256
SPLIT_UPDATES = 2
PHILOX_KEY = 1729
DOCUMENTS = (
    bytes((0, 34, 34, 197, 4, 7, 219, 12, 255)),
    bytes((163, 5, 13, 230, 17, 5, 111, 27, 31)),
    bytes((127, 140, 253, 23, 129, 94, 23, 89, 41)),
    bytes((54, 67, 92, 54, 203, 109, 126, 9, 190)),
    bytes((200, 240, 120, 16, 120, 72, 36, 18, 1)),
    bytes((85, 186, 3, 222, 245, 56, 245, 251, 2)),
)
TOKENIZER = Identity()
D2_CONFIG = ReferenceConfig(
    batch_size=1,
    sequence_length=CONTEXT,
    packing=False,
    directory="train",
    maximum_manifest_bytes=1024,
    maximum_shard_bytes=1024,
    maximum_total_tokens=sum(map(len, DOCUMENTS)),
    maximum_batch_bytes=CONTEXT * BATCH_ELEMENT_BYTES,
)
X1_INPUT = {
    "config-schema-major": 1,
    "config-schema-minor": 0,
    "model.context-length": CONTEXT,
    "model.device": "cpu",
    "model.dtype": "f32",
    "model.head-size": 4,
    "model.hidden-size": 16,
    "model.kv-head-count": 2,
    "model.layer-count": 1,
    "model.query-head-count": 4,
    "model.vocabulary-size": VOCAB,
    "run.deterministic": True,
    "run.seed": PHILOX_KEY,
    "training.accumulation-steps": 1,
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def expected_resolved_x1():
    return x1.canonical_bytes({
        "canonicalization": x1.CANONICALIZATION,
        "checksum-algorithm": "sha256",
        "checksum-coverage": "whole-document-including-final-lf",
        "config-schema-version": x1.SCHEMA_VERSION,
        "format": x1.FORMAT,
        "format-version": x1.FORMAT_VERSION,
        "limits": x1.LIMITS,
        "provenance": dict.fromkeys(X1_INPUT, "input"),
        "required-features": [],
        "resolved": X1_INPUT,
    })


def corpus_shards(documents):
    return tuple(encode_bytes(TOKENIZER, document) for document in documents)


def cursor_plan(shards, manifest_trailer, fingerprint):
    dataset = ReferenceDataset(
        shards, D2_CONFIG, manifest_digest=manifest_trailer,
        tokenizer_fingerprint=fingerprint, vocab_size=VOCAB,
    )
    cursor_hashes = [digest(dataset.snapshot())]
    for tokens in shards:
        batch = dataset.next_batch()
        if (len(tokens) != CONTEXT + 1 or batch is None
                or batch.inputs != (tokens[:-1],) or batch.targets != (tokens[1:],)
                or batch.loss_mask != ((True,) * CONTEXT,)):
            raise ValueError("expected one complete unpacked D2 row per document")
        cursor_hashes.append(digest(dataset.snapshot()))
    if dataset.next_batch() is not None:
        raise ValueError("unexpected D2 rows after the last document")
    return cursor_hashes


def expected_record(root):
    artifact = (root / "tokenizer.etok").read_bytes()
    tokenizer = decode_artifact(artifact)
    if artifact != encode_artifact(TOKENIZER):
        raise ValueError("expected canonical V256 T1 raw byte tokenizer")

    source = (root / "config.json").read_bytes()
    resolved = (root / "expected-x1.json").read_bytes()
    if source != x1.canonical_bytes(X1_INPUT) or resolved != expected_resolved_x1():
        raise ValueError("X1 fixture differs from the fixed MOD5 V256 profile")
    x1.load_manifest_bytes(resolved)

    corpus = root / D2_CONFIG.directory
    documents = tuple(
        (corpus / f"doc-{i}.bin").read_bytes() for i in range(len(DOCUMENTS))
    )
    if documents != DOCUMENTS:
        raise ValueError("original MOD5 training documents differ from the fixture")
    shards, trailer = load_d1_resource(
        corpus, expected_fingerprint=tokenizer.fingerprint, expected_vocab=VOCAB,
        maximum_manifest_bytes=D2_CONFIG.maximum_manifest_bytes,
        maximum_shard_bytes=D2_CONFIG.maximum_shard_bytes,
        maximum_total_tokens=D2_CONFIG.maximum_total_tokens,
    )
    if shards != corpus_shards(documents):
        raise ValueError("D1 shards differ from the T1 encodings of original documents")

    return {
        "version": 1,
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "tokenizer_artifact_sha256": digest(artifact),
        "manifest_trailer_sha256": trailer.hex(),
        "document_sha256": [digest(document) for document in documents],
        "x1_fingerprint": x1.fingerprint(resolved),
        "d2": asdict(D2_CONFIG),
        "prefix_updates": SPLIT_UPDATES,
        "total_updates": len(shards),
        "oracle_cursor_sha256_by_next_ordinal": cursor_plan(
            shards, trailer, tokenizer.fingerprint),
    }


def build(root):
    root.mkdir()
    (root / "tokenizer.etok").write_bytes(encode_artifact(TOKENIZER))
    (root / "config.json").write_bytes(x1.canonical_bytes(X1_INPUT))
    (root / "expected-x1.json").write_bytes(expected_resolved_x1())
    corpus = root / D2_CONFIG.directory
    corpus.mkdir()
    for index, document in enumerate(DOCUMENTS):
        (corpus / f"doc-{index}.bin").write_bytes(document)
    write_d1_resource(
        corpus, corpus_shards(DOCUMENTS),
        fingerprint=fingerprint(TOKENIZER), vocab=VOCAB,
    )
    record = expected_record(root)
    (root / "fixture.json").write_bytes(x1.canonical_bytes(record))
    return record


def verify(root):
    record = expected_record(root)
    if (root / "fixture.json").read_bytes() != x1.canonical_bytes(record):
        raise ValueError("MOD5 C2 fixture record differs from its verified inputs")
    return record


def main(argv):
    if len(argv) != 3 or argv[1] not in {"build", "verify"}:
        raise SystemExit("usage: python3 -m tests.mod5.mod5_v256_c2_fixture build|verify DIR")
    command = build if argv[1] == "build" else verify
    print(json.dumps(command(Path(argv[2])), sort_keys=True))


if __name__ == "__main__":
    main(sys.argv)
