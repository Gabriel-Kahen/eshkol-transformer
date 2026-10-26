"""Private zero-M3, nonzero-correct D1/D2 witness for equivalent controls."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import platform

from tests.d1.reference_format import parse_corpus
from tests.d2.reference import ReferenceConfig, ReferenceDataset, decode_reference_cursor
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_reference.corpus import FINGERPRINT, IDENTITY, PROFILE, VOCAB_SIZE
from tests.e3_reference.transcript import canonical, encode
from tests.e3_reference.uniform_tie import ALIAS, PROVIDER, SHAPES, checkpoints
from tests.t1.reference import encode_artifact, encode_bytes


ROOT = Path(__file__).resolve().parents[2]
NAME = "nonzero-equivalence"
RAW = bytes((0, 0, 1, 1, 0))
CONFIG = ReferenceConfig(1, 2, None, 1, True, maximum_batch_bytes=34)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def batches() -> list[dict[str, object]]:
    dataset = ReferenceDataset((tuple(RAW),), CONFIG,
                               manifest_digest=bytes(32),
                               tokenizer_fingerprint=FINGERPRINT,
                               vocab_size=VOCAB_SIZE)
    return [{"inputs": list(row.inputs), "targets": list(row.targets),
             "mask": list(row.loss_mask)} for row in dataset.rows]


def validate(root: Path, payload: object) -> None:
    if root.is_symlink() or root.resolve() != root.absolute():
        raise ValueError("nonzero resource path is not canonical")
    if not isinstance(payload, dict) or set(payload) != {
        "kind", "profile", "tokenizer", "checkpoint", "case"
    } or payload["kind"] != "e3-nonzero-equivalence-v1" or payload["profile"] != PROFILE:
        raise ValueError("nonzero resource schema differs")
    artifact = encode_artifact(IDENTITY)
    if payload["tokenizer"] != {
        "sha256": hashlib.sha256(artifact).hexdigest(),
        "fingerprint": FINGERPRINT, "vocab_size": VOCAB_SIZE,
    } or (root / "tokenizer.etok").read_bytes() != artifact:
        raise ValueError("nonzero T1 identity differs")
    check = checkpoints()
    expected_checkpoint = {
        "provider": PROVIDER.decode("ascii"), "entry_count": 15,
        "alias_indices": list(ALIAS),
        "files": [{"path": name, "sha256": hashlib.sha256(raw).hexdigest(),
                   "bytes": len(raw)} for name, raw in check.items()],
    }
    if payload["checkpoint"] != expected_checkpoint or any(
        (root / name).is_symlink() or (root / name).read_bytes() != raw
        for name, raw in check.items()
    ):
        raise ValueError("nonzero C1 bytes or contract differ")
    case = payload["case"]
    if not isinstance(case, dict) or set(case) != {
        "name", "source_bytes", "batches", "start_cursor", "end_cursor",
        "total_rows", "corpus"
    } or case["name"] != NAME or case["source_bytes"] != RAW.hex():
        raise ValueError("nonzero case identity differs")
    corpus = root / NAME
    files = [{"path": path.name, "bytes": path.stat().st_size,
              "sha256": digest(path)} for path in sorted(corpus.iterdir())]
    if corpus.is_symlink() or case["corpus"].get("files") != files:
        raise ValueError("nonzero D1 inventory differs")
    loaded, manifest = load_d1_resource(corpus,
        expected_fingerprint=FINGERPRINT, expected_vocab=VOCAB_SIZE,
        maximum_total_tokens=len(RAW))
    if tuple(parse_corpus(corpus)["tokens"]) != tuple(RAW) or \
            case["corpus"].get("manifest_trailer_sha256") != manifest.hex():
        raise ValueError("nonzero D1 bytes or manifest differ")
    dataset = ReferenceDataset(loaded, CONFIG, manifest_digest=manifest,
                               tokenizer_fingerprint=FINGERPRINT,
                               vocab_size=VOCAB_SIZE)
    start = dataset.snapshot().hex()
    rows = []
    while (row := dataset.next_batch()) is not None:
        rows.append({"inputs": list(row.inputs[0]), "targets": list(row.targets[0]),
                     "mask": list(row.loss_mask[0])})
    if rows != batches() or rows != [
        {"inputs": [0, 0], "targets": [0, 1], "mask": [True, True]},
        {"inputs": [1, 1], "targets": [1, 0], "mask": [True, True]},
    ] or case["batches"] != rows or case["total_rows"] != 2 or \
            case["start_cursor"] != start or case["end_cursor"] != dataset.snapshot().hex():
        raise ValueError("nonzero D2 full rows or cursor differ")
    for key, ordinal in (("start_cursor", 0), ("end_cursor", 2)):
        cursor = decode_reference_cursor(bytes.fromhex(case[key]))
        if cursor.next_ordinal != ordinal or cursor.total_rows != 2 or \
                cursor.manifest_digest != manifest or cursor.config != CONFIG or \
                cursor.tokenizer_fingerprint != FINGERPRINT:
            raise ValueError("nonzero D2 cursor identity differs")
    expected = {"tokenizer.etok", *check, NAME, "nonzero-manifest.json",
                *(f"{NAME}/{item['path']}" for item in files)}
    if {str(path.relative_to(root)) for path in root.rglob("*")} != expected or any(
        path.is_symlink() or not (path.is_file() or path.is_dir())
        for path in root.rglob("*")
    ) or (root / "nonzero-manifest.json").read_bytes() != canonical(payload):
        raise ValueError("nonzero resource inventory differs")


def materialize(root: Path) -> dict[str, object]:
    if root.exists() or root.is_symlink() or root.parent.resolve() != root.parent.absolute():
        raise ValueError("nonzero resource output must be fresh and canonical")
    root.mkdir(parents=True)
    artifact = encode_artifact(IDENTITY)
    if encode_bytes(IDENTITY, RAW) != tuple(RAW):
        raise AssertionError("T1 raw byte identity differs")
    (root / "tokenizer.etok").write_bytes(artifact)
    check = checkpoints()
    for name, raw in check.items():
        (root / name).write_bytes(raw)
    corpus = root / NAME
    corpus.mkdir()
    manifest = write_d1_resource(corpus, (tuple(RAW),),
                                 fingerprint=FINGERPRINT, vocab=VOCAB_SIZE)
    loaded, manifest_digest = load_d1_resource(corpus,
        expected_fingerprint=FINGERPRINT, expected_vocab=VOCAB_SIZE,
        maximum_total_tokens=len(RAW))
    if manifest[-32:] != manifest_digest:
        raise AssertionError("D1 trailer differs")
    dataset = ReferenceDataset(loaded, CONFIG, manifest_digest=manifest_digest,
                               tokenizer_fingerprint=FINGERPRINT,
                               vocab_size=VOCAB_SIZE)
    start = dataset.snapshot().hex()
    rows = []
    while (row := dataset.next_batch()) is not None:
        rows.append({"inputs": list(row.inputs[0]), "targets": list(row.targets[0]),
                     "mask": list(row.loss_mask[0])})
    files = [{"path": path.name, "bytes": path.stat().st_size,
              "sha256": digest(path)} for path in sorted(corpus.iterdir())]
    payload = {
        "kind": "e3-nonzero-equivalence-v1", "profile": PROFILE,
        "tokenizer": {"sha256": hashlib.sha256(artifact).hexdigest(),
                      "fingerprint": FINGERPRINT, "vocab_size": VOCAB_SIZE},
        "checkpoint": {"provider": PROVIDER.decode("ascii"),
                       "entry_count": 15, "alias_indices": list(ALIAS),
                       "files": [{"path": name, "sha256": hashlib.sha256(raw).hexdigest(),
                                  "bytes": len(raw)} for name, raw in check.items()]},
        "case": {"name": NAME, "source_bytes": RAW.hex(), "batches": rows,
                 "start_cursor": start, "end_cursor": dataset.snapshot().hex(),
                 "total_rows": len(dataset.rows),
                 "corpus": {"manifest_trailer_sha256": manifest_digest.hex(),
                            "files": files}},
    }
    (root / "nonzero-manifest.json").write_bytes(canonical(payload))
    validate(root, payload)
    return payload


def build_reference(payload: dict[str, object]) -> dict[str, object]:
    from tests.e3_reference.reference import _synthetic_case, validate_bundle
    from tests.m3 import reference as m3

    m3.setup()
    if tuple(m3.PARAMETERS) != SHAPES[:-1]:
        raise ValueError("M3 shape contract differs")
    parameters = {path: m3.torch.zeros(shape, dtype=m3.torch.float64)
                  for path, shape in m3.PARAMETERS}
    case = payload["case"]
    synthetic, mathematical, loss = _synthetic_case(case["batches"], parameters)
    if [row["correct_f32"] for row in synthetic["batches"]] != ["3f800000"] * 2 or \
            synthetic["final"]["metrics_f32"][3] != "3f000000" or \
            any(row["ce_f32"] != ["40b17218"] * 2 for row in synthetic["batches"]) or \
            any(word != "00000000" for row in synthetic["batches"]
                for word in row["logits_f32"]):
        raise ValueError("nonzero correct/CE/zero-logit oracle differs")
    _, successor = m3.parameters(1729)
    total_bytes = sum(math.prod(shape) * 4 for _, shape in SHAPES[:-1])
    result = {"kind": "reference-bundle", "profile": m3.PROFILE,
              "cases": [{"name": NAME, "corpus": case, "synthetic": synthetic,
                         "mathematical": mathematical,
                         "mathematical_loss_f64": loss}],
              "model": {"initializer_seed": 1729,
                        "initializer_successor": list(successor),
                        "parameter_f32_sha256": hashlib.sha256(bytes(total_bytes)).hexdigest(),
                        "reference_manifest_sha256": digest(ROOT / "tests/m3/reference_manifest.json")},
              "provenance": {"corpus_source_sha256": digest(Path(__file__)),
                             "generator_source_sha256": digest(Path(__file__)),
                             "m3_reference_source_sha256": digest(ROOT / "tests/m3/reference.py"),
                             "initializer_source_sha256": digest(ROOT / "tests/n3k/test_initializer_reference.py"),
                             "rational_reference_source_sha256": digest(ROOT / "tests/e3_metrics/reference_probe.py"),
                             "q0_lock_sha256": digest(ROOT / "tests/q0/requirements-oracle.lock"),
                             "python": platform.python_version(), "torch": m3.torch.__version__}}
    validate_bundle(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    args = parser.parse_args()
    if args.reference.exists() or args.reference.is_symlink():
        raise FileExistsError("nonzero reference output must be fresh")
    payload = materialize(args.output)
    args.reference.write_bytes(encode(build_reference(payload)))


if __name__ == "__main__":
    main()
