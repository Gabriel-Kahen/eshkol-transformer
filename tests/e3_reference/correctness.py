"""Independent two-case E3 correctness corpus and mathematical transcript.

This versioned leaf leaves the five-case numerical corpus and its schema intact.
The declared classes are checked against freshly evaluated seed-1729 M3 logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import struct

from tests.d1.reference_format import parse_corpus
from tests.d2.reference import ReferenceConfig, ReferenceDataset, decode_reference_cursor
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_reference.corpus import FINGERPRINT, IDENTITY, PROFILE, VOCAB_SIZE
from tests.e3_reference.reference import _synthetic_case, validate_bundle
from tests.e3_reference.transcript import canonical, encode, f32_bits, validate_observed_case
from tests.m3 import reference as m3
from tests.t1.reference import encode_artifact, encode_bytes


ROOT = Path(__file__).resolve().parents[2]
CASES = (
    ("all-correct", bytes.fromhex("030e1f0e1f0e"), True),
    ("all-wrong", bytes.fromhex("030f39c5e7c5"), False),
)
KIND = "e3-correctness-corpus-v1"
CONFIG = ReferenceConfig(1, 2, None, 1, True, maximum_batch_bytes=34)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_fresh(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("correctness output must be fresh")
    ancestor = path.parent
    while not ancestor.exists():
        if ancestor.is_symlink():
            raise ValueError("correctness output ancestor is a symlink")
        ancestor = ancestor.parent
    if ancestor.is_symlink() or ancestor.resolve() != ancestor.absolute():
        raise ValueError("correctness output ancestor is a symlink")


def _batch(batch) -> dict[str, object]:
    return {
        "inputs": list(batch.inputs[0]),
        "targets": list(batch.targets[0]),
        "mask": list(batch.loss_mask[0]),
    }


def validate_manifest(payload: object) -> None:
    if not isinstance(payload, dict) or set(payload) != {"kind", "profile", "tokenizer", "cases"}:
        raise ValueError("correctness manifest schema mismatch")
    if payload["kind"] != KIND or payload["profile"] != PROFILE:
        raise ValueError("correctness manifest kind/profile mismatch")
    artifact = encode_artifact(IDENTITY)
    if payload["tokenizer"] != {
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "fingerprint": FINGERPRINT,
        "vocab_size": VOCAB_SIZE,
    }:
        raise ValueError("correctness tokenizer identity mismatch")
    cases = payload["cases"]
    if not isinstance(cases, list) or len(cases) != len(CASES):
        raise ValueError("correctness case count mismatch")
    for item, (name, raw, correct) in zip(cases, CASES):
        if not isinstance(item, dict) or set(item) != {
            "name", "source_bytes", "declared_correct", "batches", "start_cursor",
            "end_cursor", "total_rows", "corpus",
        }:
            raise ValueError("correctness case schema mismatch")
        if (item["name"] != name or item["source_bytes"] != raw.hex()
                or type(item["declared_correct"]) is not bool
                or item["declared_correct"] != correct):
            raise ValueError("correctness case identity/class mismatch")
        corpus = item["corpus"]
        if not isinstance(corpus, dict) or set(corpus) != {"files", "manifest_trailer_sha256"}:
            raise ValueError("correctness corpus schema mismatch")
        digest = corpus["manifest_trailer_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("correctness manifest digest mismatch")
        files = corpus["files"]
        if not isinstance(files, list) or len(files) != 2 or [f.get("path") if isinstance(f, dict) else None for f in files] != ["manifest.etm", "shard-0000000000000000.ets"]:
            raise ValueError("correctness file inventory mismatch")
        for file in files:
            if set(file) != {"path", "bytes", "sha256"} or type(file["bytes"]) is not int or not 1 <= file["bytes"] <= 1_048_576:
                raise ValueError("correctness file record mismatch")
            word = file["sha256"]
            if not isinstance(word, str) or len(word) != 64 or any(c not in "0123456789abcdef" for c in word):
                raise ValueError("correctness file checksum mismatch")
        dataset = ReferenceDataset((tuple(raw),), CONFIG, manifest_digest=bytes.fromhex(digest), tokenizer_fingerprint=FINGERPRINT, vocab_size=VOCAB_SIZE)
        expected = [{"inputs": list(row.inputs), "targets": list(row.targets),
                     "mask": list(row.loss_mask)} for row in dataset.rows]
        if type(item["total_rows"]) is not int or item["total_rows"] != 3 or item["total_rows"] != len(expected):
            raise ValueError("correctness row count mismatch")
        if item["batches"] != expected or any(
            type(value) is not int or not 0 <= value < 256
            for batch in item["batches"] for field in ("inputs", "targets") for value in batch[field]
        ) or any(type(value) is not bool for batch in item["batches"] for value in batch["mask"]):
            raise ValueError("correctness D2 rows/IDs mismatch")
        if [batch["mask"] for batch in item["batches"]] != [[True, True], [True, True], [True, False]]:
            raise ValueError("correctness mask/count mismatch")
        for field, ordinal in (("start_cursor", 0), ("end_cursor", 3)):
            word = item[field]
            if not isinstance(word, str) or len(word) > 800 or len(word) % 2 or any(c not in "0123456789abcdef" for c in word):
                raise ValueError("correctness cursor encoding mismatch")
            cursor = decode_reference_cursor(bytes.fromhex(word))
            if (cursor.next_ordinal != ordinal or cursor.total_rows != 3
                    or cursor.manifest_digest.hex() != digest
                    or cursor.tokenizer_fingerprint != FINGERPRINT
                    or cursor.vocab_size != VOCAB_SIZE or cursor.config != CONFIG):
                raise ValueError("correctness cursor identity/count mismatch")


def materialize(output: Path) -> dict[str, object]:
    _safe_fresh(output)
    output.mkdir(parents=True)
    artifact = encode_artifact(IDENTITY)
    (output / "tokenizer.etok").write_bytes(artifact)
    cases = []
    for name, raw, correct in CASES:
        if encode_bytes(IDENTITY, raw) != tuple(raw):
            raise AssertionError("raw tokenizer changed correctness IDs")
        directory = output / name
        directory.mkdir()
        manifest = write_d1_resource(directory, (tuple(raw),), fingerprint=FINGERPRINT, vocab=VOCAB_SIZE)
        loaded, digest = load_d1_resource(directory, expected_fingerprint=FINGERPRINT, expected_vocab=VOCAB_SIZE, maximum_total_tokens=6)
        if tuple(parse_corpus(directory)["tokens"]) != tuple(raw):
            raise AssertionError(f"{name}: D1 bytes changed")
        dataset = ReferenceDataset(loaded, CONFIG, manifest_digest=digest, tokenizer_fingerprint=FINGERPRINT, vocab_size=VOCAB_SIZE)
        start = dataset.snapshot().hex()
        batches = []
        while (batch := dataset.next_batch()) is not None:
            batches.append(_batch(batch))
        end = dataset.snapshot().hex()
        cases.append({
            "name": name, "source_bytes": raw.hex(), "declared_correct": correct,
            "batches": batches, "start_cursor": start, "end_cursor": end,
            "total_rows": len(dataset.rows),
            "corpus": {
                "manifest_trailer_sha256": manifest[-32:].hex(),
                "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": _hash(p)}
                          for p in sorted(directory.iterdir())],
            },
        })
    payload = {
        "kind": KIND, "profile": PROFILE,
        "tokenizer": {"artifact_sha256": hashlib.sha256(artifact).hexdigest(), "fingerprint": FINGERPRINT, "vocab_size": VOCAB_SIZE},
        "cases": cases,
    }
    validate_manifest(payload)
    (output / "correctness-manifest.json").write_bytes(canonical(payload))
    return payload


def validate_materialized(root: Path, payload: dict[str, object]) -> None:
    validate_manifest(payload)
    if root.is_symlink() or root.resolve() != root.absolute():
        raise ValueError("correctness resource path is a symlink")
    expected_paths = {"tokenizer.etok", "correctness-manifest.json"}
    if (root / "tokenizer.etok").is_symlink() or (root / "correctness-manifest.json").is_symlink():
        raise ValueError("correctness top-level artifact is a symlink")
    for case in payload["cases"]:
        directory = root / case["name"]
        if directory.is_symlink():
            raise ValueError("correctness corpus directory is a symlink")
        expected_paths.add(case["name"])
        for record in case["corpus"]["files"]:
            path = directory / record["path"]
            expected_paths.add(f"{case['name']}/{record['path']}")
            if path.is_symlink() or path.stat().st_size != record["bytes"] or _hash(path) != record["sha256"]:
                raise ValueError("correctness physical file identity mismatch")
        loaded, digest = load_d1_resource(directory, expected_fingerprint=FINGERPRINT, expected_vocab=VOCAB_SIZE, maximum_total_tokens=6)
        if tuple(loaded[0]) != tuple(bytes.fromhex(case["source_bytes"])) or digest.hex() != case["corpus"]["manifest_trailer_sha256"]:
            raise ValueError("correctness physical D1 identity mismatch")
    if {str(p.relative_to(root)) for p in root.rglob("*")} != expected_paths:
        raise ValueError("correctness physical file inventory mismatch")
    if _hash(root / "tokenizer.etok") != payload["tokenizer"]["artifact_sha256"]:
        raise ValueError("correctness tokenizer artifact mismatch")
    if (root / "correctness-manifest.json").read_bytes() != canonical(payload):
        raise ValueError("correctness manifest bytes mismatch")


def build_reference(payload: dict[str, object]) -> dict[str, object]:
    validate_manifest(payload)
    m3.setup()
    parameters, successor = m3.parameters(1729)
    cases = []
    for corpus_case in payload["cases"]:
        synthetic, mathematical, loss = _synthetic_case(corpus_case["batches"], parameters)
        correct = corpus_case["declared_correct"]
        for batch, observation in zip(corpus_case["batches"], synthetic["batches"]):
            if observation["correct_f32"] != f32_bits(float(sum(batch["mask"]) if correct else 0)):
                raise ValueError(f"{corpus_case['name']}: class differs from independent M3 argmax")
        if synthetic["final"]["metrics_f32"][3] != ("3f800000" if correct else "00000000"):
            raise ValueError("correctness final accuracy differs from declared class")
        cases.append({"name": corpus_case["name"], "corpus": corpus_case,
                      "synthetic": synthetic, "mathematical": mathematical,
                      "mathematical_loss_f64": loss})
    parameter_words = b"".join(bytes.fromhex(f32_bits(item)) for path, _ in m3.PARAMETERS
                               for item in parameters[path].to(m3.torch.float32).detach().reshape(-1).tolist())
    source_hash = _hash(Path(__file__))
    result = {
        "kind": "reference-bundle", "profile": m3.PROFILE, "cases": cases,
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


def selection_proof(reference: dict[str, object]) -> dict[str, object]:
    """Re-derive rank from the accepted f64 M3 trace cast to binary32."""
    validate_bundle(reference)
    cases = []
    for case in reference["cases"]:
        declared = case["corpus"]["declared_correct"]
        winners = []
        minimum_gap = float("inf")
        for batch, mathematical in zip(case["corpus"]["batches"], case["mathematical"]):
            final_role = mathematical["roles"][-1]
            if final_role["name"] != "Z" or final_role["shape"] != [1, 2, 256]:
                raise ValueError("correctness selection lacks M3 Z role")
            values = [struct.unpack(">f", struct.pack(">f", struct.unpack(">d", bytes.fromhex(word))[0]))[0]
                      for word in final_role["values_f64"]]
            row_winners = []
            for position, active in enumerate(batch["mask"]):
                row = values[position * 256:(position + 1) * 256]
                ranked = sorted(range(256), key=lambda token: (-row[token], token))
                row_winners.append(ranked[0])
                if active:
                    minimum_gap = min(minimum_gap, row[ranked[0]] - row[ranked[1]])
                    if (ranked[0] == batch["targets"][position]) != declared:
                        raise ValueError("correctness declaration differs from M3 winner")
            winners.append(row_winners)
        if not minimum_gap > 2.0e-4:
            raise ValueError("correctness rank gap is too narrow")
        cases.append({"name": case["name"], "winners": winners,
                      "minimum_rank_gap_f64": minimum_gap,
                      "active_tokens": 5, "declared_correct": declared})
    return {"kind": "e3-correctness-selection-proof-v1", "cases": cases}


def verify_observed(manifest: dict[str, object], case_name: str, observed: dict[str, object]) -> None:
    validate_manifest(manifest)
    matches = [case for case in manifest["cases"] if case["name"] == case_name]
    if len(matches) != 1:
        raise ValueError("correctness observation case mismatch")
    case = matches[0]
    validate_observed_case(observed, tuple(case["batches"]))
    if observed["final"]["counters"] != [5, 3]:
        raise ValueError("correctness observed count mismatch")
    if observed["final"]["metrics_f32"][1] != "40a00000":
        raise ValueError("correctness observed weight mismatch")
    accuracy = "3f800000" if case["declared_correct"] else "00000000"
    if observed["final"]["metrics_f32"][3] != accuracy:
        raise ValueError("correctness observed accuracy mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--observed", type=Path)
    parser.add_argument("--case")
    args = parser.parse_args()
    if args.output is not None:
        if any(value is not None for value in (args.corpus, args.reference, args.proof, args.observed, args.case)):
            parser.error("--output must stand alone")
        materialize(args.output)
        return
    if args.corpus is None or (args.reference is None) == (args.observed is None):
        parser.error("use --corpus with --reference or --observed")
    if args.observed is not None and (args.case is None or args.proof is not None):
        parser.error("observation requires --case and no --proof")
    if args.reference is not None and (args.case is not None or args.observed is not None):
        parser.error("reference does not take --case/--observed")
    payload = json.loads((args.corpus / "correctness-manifest.json").read_bytes())
    validate_materialized(args.corpus, payload)
    if args.reference is not None:
        result = build_reference(payload)
        proof = selection_proof(result)
        _safe_fresh(args.reference)
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        args.reference.write_bytes(encode(result))
        if args.proof is not None:
            _safe_fresh(args.proof)
            args.proof.parent.mkdir(parents=True, exist_ok=True)
            args.proof.write_bytes(canonical(proof))
    else:
        verify_observed(payload, args.case, json.loads(args.observed.read_bytes()))


if __name__ == "__main__":
    main()
