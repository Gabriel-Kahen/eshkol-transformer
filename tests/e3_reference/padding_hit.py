"""Genuine packed D2 final-10 case with a wrong valid target and padded hit."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import platform

from tests.d2.reference import ReferenceDataset
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_reference.corpus import FINGERPRINT, IDENTITY, PROFILE, VOCAB_SIZE
from tests.e3_reference.transcript import canonical, encode
from tests.e3_reference.uniform_tie import CONFIG, SHAPES, checkpoints, zero_checkpoint
from tests.t1.reference import encode_artifact, encode_bytes


RAW = bytes((0, 0, 0, 0, 0, 1))
NAME = "padding-hit"
ROOT = Path(__file__).resolve().parents[2]


def materialize(output: Path) -> dict[str, object]:
    if output.exists() or output.is_symlink() or output.parent.resolve() != output.parent.absolute():
        raise ValueError("padding-hit fixture output must be fresh and canonical")
    if encode_bytes(IDENTITY, RAW) != tuple(RAW):
        raise AssertionError("raw tokenizer changed the selected IDs")
    output.mkdir(parents=True)
    (output / "tokenizer.etok").write_bytes(encode_artifact(IDENTITY))
    for name, data in checkpoints().items():
        (output / name).write_bytes(data)
    corpus = output / NAME
    corpus.mkdir()
    manifest = write_d1_resource(corpus, (tuple(RAW),),
                                 fingerprint=FINGERPRINT, vocab=VOCAB_SIZE)
    loaded, digest = load_d1_resource(corpus, expected_fingerprint=FINGERPRINT,
                                      expected_vocab=VOCAB_SIZE, maximum_total_tokens=6)
    if loaded != (tuple(RAW),) or digest != manifest[-32:]:
        raise ValueError("padding-hit D1 bytes or checksum differ")
    dataset = ReferenceDataset(loaded, CONFIG, manifest_digest=digest,
                               tokenizer_fingerprint=FINGERPRINT, vocab_size=VOCAB_SIZE)
    start = dataset.snapshot().hex()
    rows = []
    while (row := dataset.next_batch()) is not None:
        rows.append({"inputs": list(row.inputs[0]), "targets": list(row.targets[0]),
                     "mask": list(row.loss_mask[0])})
    if rows != [
        {"inputs": [0, 0], "targets": [0, 0], "mask": [True, True]},
        {"inputs": [0, 0], "targets": [0, 0], "mask": [True, True]},
        {"inputs": [0, 0], "targets": [1, 0], "mask": [True, False]},
    ]:
        raise ValueError("padding-hit D2 emission differs from required 11,11,10")
    return {"name": NAME, "source_bytes": RAW.hex(), "batches": rows,
            "start_cursor": start, "end_cursor": dataset.snapshot().hex(),
            "total_rows": 3,
            "corpus": {"manifest_trailer_sha256": digest.hex(),
                       "files": [{"path": p.name, "bytes": p.stat().st_size,
                                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                 for p in sorted(corpus.iterdir())]}}


def build_reference(case: dict[str, object]) -> dict[str, object]:
    from tests.e3_reference.reference import _synthetic_case, validate_bundle
    from tests.m3 import reference as m3

    m3.setup()
    parameters = {name: m3.torch.zeros(shape, dtype=m3.torch.float64)
                  for name, shape in m3.PARAMETERS}
    synthetic, mathematical, loss = _synthetic_case(case["batches"], parameters)
    if synthetic["final"]["metrics_f32"][3] != "3f4ccccd":
        raise ValueError("normal zero model must score four of five valid tokens")
    if any(any(word != "00000000" for word in batch["logits_f32"])
           for batch in synthetic["batches"]):
        raise ValueError("zero-state model emitted nonzero logits")
    if any(batch["ce_f32"] != ["40b17218", "40b17218"]
           for batch in synthetic["batches"]):
        raise ValueError("uniform CE differs from log(256)")
    _, successor = m3.parameters(1729)
    source = Path(__file__)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    result = {"kind": "reference-bundle", "profile": m3.PROFILE,
              "cases": [{"name": NAME, "corpus": case, "synthetic": synthetic,
                         "mathematical": mathematical,
                         "mathematical_loss_f64": loss}],
              "model": {"initializer_seed": 1729,
                        "initializer_successor": list(successor),
                        "parameter_f32_sha256": hashlib.sha256(bytes(
                            sum(math.prod(shape) * 4 for _, shape in SHAPES[:-1]))).hexdigest(),
                        "reference_manifest_sha256": digest(ROOT / "tests/m3/reference_manifest.json")},
              "provenance": {"corpus_source_sha256": digest(source),
                             "generator_source_sha256": digest(source),
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
        raise FileExistsError("padding-hit reference output must be fresh")
    case = materialize(args.output)
    args.reference.write_bytes(encode(build_reference(case)))
    (args.output / "padding-hit-manifest.json").write_bytes(canonical(case))


if __name__ == "__main__":
    main()
