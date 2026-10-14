"""Authentic one-valid-token N5 corpus and independent numerical oracle.

This prepares a source candidate; only a later compiled numerical-tuple run can
establish native E3 behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

from tests.d1.reference_format import parse_corpus
from tests.d2.reference import ReferenceConfig, ReferenceDataset, decode_reference_cursor
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_reference.corpus import FINGERPRINT, IDENTITY, PROFILE, VOCAB_SIZE
from tests.e3_reference.reference import _synthetic_case, validate_bundle
from tests.e3_reference.transcript import canonical, encode, f32_bits, validate_observed_case
from tests.m3 import reference as m3
from tests.t1.reference import encode_artifact, encode_bytes


ROOT = Path(__file__).resolve().parents[2]
SOURCE = bytes((7, 8))
NAME = "one-final-token"
KIND = "e3-n5-one-token-corpus-v1"
CONFIG = ReferenceConfig(1, 2, None, 1, True, maximum_batch_bytes=34)
ROW = {"inputs": [7, 0], "targets": [8, 0], "mask": [True, False]}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fresh(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("N5 one-token output must be fresh")
    ancestor = path.parent
    while not ancestor.exists():
        if ancestor.is_symlink():
            raise ValueError("N5 output ancestor is a symlink")
        ancestor = ancestor.parent
    if ancestor.is_symlink() or ancestor.resolve() != ancestor.absolute():
        raise ValueError("N5 output ancestor is a symlink")


def _cursor(word: object, digest: str, ordinal: int) -> None:
    if not isinstance(word, str) or not 0 < len(word) <= 800 or len(word) % 2:
        raise ValueError("N5 cursor encoding mismatch")
    try:
        fields = decode_reference_cursor(bytes.fromhex(word))
    except (ValueError, TypeError) as error:
        raise ValueError("N5 cursor encoding mismatch") from error
    if (fields.next_ordinal != ordinal or fields.total_rows != 1
            or fields.manifest_digest.hex() != digest
            or fields.tokenizer_fingerprint != FINGERPRINT
            or fields.vocab_size != VOCAB_SIZE or fields.config != CONFIG):
        raise ValueError("N5 cursor identity/count mismatch")


def validate_manifest(payload: object) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "kind", "profile", "source_bytes", "tokenizer", "batches",
        "start_cursor", "end_cursor", "total_rows", "corpus",
    }:
        raise ValueError("N5 manifest schema mismatch")
    if (payload["kind"] != KIND or payload["profile"] != PROFILE
            or payload["source_bytes"] != SOURCE.hex()
            or payload["tokenizer"] != {
                "artifact_sha256": hashlib.sha256(encode_artifact(IDENTITY)).hexdigest(),
                "fingerprint": FINGERPRINT, "vocab_size": VOCAB_SIZE,
            }):
        raise ValueError("N5 corpus/tokenizer identity mismatch")
    corpus = payload["corpus"]
    if not isinstance(corpus, dict) or set(corpus) != {"files", "manifest_trailer_sha256"}:
        raise ValueError("N5 corpus schema mismatch")
    digest = corpus["manifest_trailer_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("N5 manifest digest mismatch")
    files = corpus["files"]
    if (not isinstance(files, list) or len(files) != 2
            or [f.get("path") if isinstance(f, dict) else None for f in files]
            != ["manifest.etm", "shard-0000000000000000.ets"]):
        raise ValueError("N5 physical file inventory mismatch")
    for file in files:
        if (set(file) != {"path", "bytes", "sha256"}
                or type(file["bytes"]) is not int or not 1 <= file["bytes"] <= 1_048_576
                or not isinstance(file["sha256"], str) or len(file["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in file["sha256"])):
            raise ValueError("N5 physical file record mismatch")
    rows = ReferenceDataset((tuple(SOURCE),), CONFIG, manifest_digest=bytes.fromhex(digest),
                            tokenizer_fingerprint=FINGERPRINT, vocab_size=VOCAB_SIZE).rows
    expected = [{"inputs": list(row.inputs), "targets": list(row.targets),
                 "mask": list(row.loss_mask)} for row in rows]
    if (type(payload["total_rows"]) is not int or payload["total_rows"] != 1
            or expected != [ROW] or payload["batches"] != expected):
        raise ValueError("N5 one-valid-token D2 row mismatch")
    _cursor(payload["start_cursor"], digest, 0)
    _cursor(payload["end_cursor"], digest, 1)


def materialize(output: Path) -> dict[str, object]:
    _fresh(output)
    output.mkdir(parents=True)
    artifact = encode_artifact(IDENTITY)
    if encode_bytes(IDENTITY, SOURCE) != tuple(SOURCE):
        raise AssertionError("raw tokenizer changed N5 IDs")
    (output / "tokenizer.etok").write_bytes(artifact)
    directory = output / NAME
    directory.mkdir()
    manifest = write_d1_resource(directory, (tuple(SOURCE),),
                                 fingerprint=FINGERPRINT, vocab=VOCAB_SIZE)
    loaded, digest = load_d1_resource(directory, expected_fingerprint=FINGERPRINT,
                                      expected_vocab=VOCAB_SIZE, maximum_total_tokens=2)
    if tuple(parse_corpus(directory)["tokens"]) != tuple(SOURCE):
        raise AssertionError("N5 D1 bytes changed")
    dataset = ReferenceDataset(loaded, CONFIG, manifest_digest=digest,
                               tokenizer_fingerprint=FINGERPRINT, vocab_size=VOCAB_SIZE)
    start = dataset.snapshot().hex()
    batches = []
    while (batch := dataset.next_batch()) is not None:
        batches.append({"inputs": list(batch.inputs[0]), "targets": list(batch.targets[0]),
                        "mask": list(batch.loss_mask[0])})
    payload = {
        "kind": KIND, "profile": PROFILE, "source_bytes": SOURCE.hex(),
        "tokenizer": {"artifact_sha256": hashlib.sha256(artifact).hexdigest(),
                      "fingerprint": FINGERPRINT, "vocab_size": VOCAB_SIZE},
        "batches": batches, "start_cursor": start,
        "end_cursor": dataset.snapshot().hex(), "total_rows": len(dataset.rows),
        "corpus": {
            "manifest_trailer_sha256": manifest[-32:].hex(),
            "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": _hash(p)}
                      for p in sorted(directory.iterdir())],
        },
    }
    validate_manifest(payload)
    (output / "n5-one-token-manifest.json").write_bytes(canonical(payload))
    return payload


def validate_materialized(root: Path, payload: dict[str, object]) -> None:
    validate_manifest(payload)
    if root.is_symlink() or root.resolve() != root.absolute():
        raise ValueError("N5 resource root is a symlink")
    expected = {"tokenizer.etok", "n5-one-token-manifest.json", NAME}
    for path, digest in ((root / "tokenizer.etok", payload["tokenizer"]["artifact_sha256"]),
                         (root / "n5-one-token-manifest.json", None)):
        if path.is_symlink() or (digest is not None and _hash(path) != digest):
            raise ValueError("N5 top-level artifact identity mismatch")
    directory = root / NAME
    if directory.is_symlink():
        raise ValueError("N5 corpus directory is a symlink")
    for record in payload["corpus"]["files"]:
        path = directory / record["path"]
        expected.add(f"{NAME}/{record['path']}")
        if path.is_symlink() or path.stat().st_size != record["bytes"] or _hash(path) != record["sha256"]:
            raise ValueError("N5 physical file identity mismatch")
    if {str(path.relative_to(root)) for path in root.rglob("*")} != expected:
        raise ValueError("N5 physical file inventory mismatch")
    if (root / "n5-one-token-manifest.json").read_bytes() != canonical(payload):
        raise ValueError("N5 manifest bytes mismatch")
    loaded, digest = load_d1_resource(directory, expected_fingerprint=FINGERPRINT,
                                      expected_vocab=VOCAB_SIZE, maximum_total_tokens=2)
    if loaded != (tuple(SOURCE),) or digest.hex() != payload["corpus"]["manifest_trailer_sha256"]:
        raise ValueError("N5 physical D1 identity mismatch")


def build_reference(payload: dict[str, object]) -> dict[str, object]:
    validate_manifest(payload)
    m3.setup()
    parameters, successor = m3.parameters(1729)
    synthetic, mathematical, loss = _synthetic_case(payload["batches"], parameters)
    parameter_words = b"".join(bytes.fromhex(f32_bits(item)) for path, _ in m3.PARAMETERS
                               for item in parameters[path].to(m3.torch.float32).detach().reshape(-1).tolist())
    source_hash = _hash(Path(__file__))
    result = {
        "kind": "reference-bundle", "profile": m3.PROFILE,
        "cases": [{"name": NAME, "corpus": {"name": NAME, "batches": payload["batches"]},
                   "synthetic": synthetic, "mathematical": mathematical,
                   "mathematical_loss_f64": loss}],
        "model": {"initializer_seed": 1729, "initializer_successor": list(successor),
                  "parameter_f32_sha256": hashlib.sha256(parameter_words).hexdigest(),
                  "reference_manifest_sha256": _hash(ROOT / "tests/m3/reference_manifest.json")},
        "provenance": {
            "corpus_source_sha256": source_hash, "generator_source_sha256": source_hash,
            "m3_reference_source_sha256": _hash(ROOT / "tests/m3/reference.py"),
            "initializer_source_sha256": _hash(ROOT / "tests/n3k/test_initializer_reference.py"),
            "rational_reference_source_sha256": _hash(ROOT / "tests/e3_metrics/reference_probe.py"),
            "q0_lock_sha256": _hash(ROOT / "tests/q0/requirements-oracle.lock"),
            "python": platform.python_version(), "torch": m3.torch.__version__,
        },
    }
    validate_bundle(result)
    return result


def validate_observed(payload: dict[str, object], observed: dict[str, object]) -> None:
    validate_manifest(payload)
    validate_observed_case(observed, tuple(payload["batches"]))
    if observed["final"]["counters"] != [1, 1]:
        raise ValueError("N5 observed one-token counters mismatch")
    if observed["final"]["metrics_f32"][1] != "3f800000":
        raise ValueError("N5 observed one-token weight mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.corpus is None and args.reference is None:
        materialize(args.output)
        return
    if args.output is not None or args.corpus is None or args.reference is None:
        parser.error("use --output or --corpus with --reference")
    payload = json.loads((args.corpus / "n5-one-token-manifest.json").read_bytes())
    validate_materialized(args.corpus, payload)
    _fresh(args.reference)
    args.reference.parent.mkdir(parents=True, exist_ok=True)
    with args.reference.open("xb") as stream:
        stream.write(encode(build_reference(payload)))


if __name__ == "__main__":
    main()
