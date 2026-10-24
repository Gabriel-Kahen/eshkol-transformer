"""Development-only genuine D1/D2 zero-M3 corpus and independent role oracle."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import platform

from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_reference import reference as e3
from tests.e3_reference.transcript import canonical, encode, f32_bits
from tests.m3 import reference as m3
from tests.t1.reference import Identity, fingerprint


ROOT = Path(__file__).resolve().parents[2]
CASES = (("all-correct", (5, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
         ("all-wrong", (5, 7, 7, 7, 7, 7), (7, 7, 7, 7, 7)),
         ("mixed", (5, 0, 7, 0, 7, 0), (0, 7, 0, 7, 0)))
TOKENIZER = Identity("raw", (), (), ())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create(output: Path) -> dict[str, object]:
    if output.exists() or output.is_symlink() or output.parent.is_symlink():
        raise ValueError("zero-control output must be a fresh non-symlink directory")
    output.mkdir(parents=True)
    m3.setup()
    torch = m3.torch
    parameters = {
        path: torch.zeros(shape, dtype=torch.float64).requires_grad_()
        for path, shape in m3.PARAMETERS
    }
    cases = []
    physical = []
    for name, tokens, targets in CASES:
        corpus = output / name
        corpus.mkdir()
        write_d1_resource(corpus, (tokens,), fingerprint=fingerprint(TOKENIZER), vocab=256)
        loaded, manifest_digest = load_d1_resource(
            corpus, expected_fingerprint=fingerprint(TOKENIZER),
            expected_vocab=256, maximum_total_tokens=6,
        )
        dataset = ReferenceDataset(
            loaded, ReferenceConfig(1, 2, None, 1, True),
            manifest_digest=manifest_digest,
            tokenizer_fingerprint=fingerprint(TOKENIZER), vocab_size=256,
        )
        rows = []
        while (batch := dataset.next_batch()) is not None:
            rows.append({"inputs": list(batch.inputs[0]),
                         "targets": list(batch.targets[0]),
                         "mask": list(batch.loss_mask[0])})
        if len(rows) != 3 or [r["mask"] for r in rows] != [[True, True], [True, True], [True, False]]:
            raise AssertionError("genuine packed D2 rows changed")
        observed_targets = tuple(value for row in rows for value, active in
                                 zip(row["targets"], row["mask"]) if active)
        if observed_targets != targets:
            raise AssertionError("controlled active D2 target differs")
        synthetic, mathematical, loss_word = e3._synthetic_case(rows, parameters)
        cases.append({"name": name, "corpus": {"name": name, "batches": rows},
                      "synthetic": synthetic, "mathematical": mathematical,
                      "mathematical_loss_f64": loss_word})
        files = sorted(path for path in corpus.iterdir() if path.is_file())
        physical.append({"name": name, "source_bytes": bytes(tokens).hex(),
                         "files": [{"path": path.name, "bytes": path.stat().st_size,
                                    "sha256": digest(path)} for path in files]})
    zero_bytes = bytes(4 * sum(math.prod(shape) for _, shape in m3.PARAMETERS))
    bundle = {
        "kind": "reference-bundle", "profile": m3.PROFILE, "cases": cases,
        "model": {
            "initializer_seed": 1729, "initializer_successor": [1, 1729, 290, 0],
            "parameter_f32_sha256": hashlib.sha256(zero_bytes).hexdigest(),
            "reference_manifest_sha256": digest(ROOT / "tests/m3/reference_manifest.json"),
        },
        "provenance": {
            "corpus_source_sha256": digest(Path(__file__)),
            "generator_source_sha256": digest(Path(__file__)),
            "initializer_source_sha256": digest(ROOT / "tests/n3k/test_initializer_reference.py"),
            "m3_reference_source_sha256": digest(ROOT / "tests/m3/reference.py"),
            "python": platform.python_version(),
            "q0_lock_sha256": digest(ROOT / "tests/q0/requirements-oracle.lock"),
            "rational_reference_source_sha256": digest(ROOT / "tests/e3_metrics/reference_probe.py"),
            "torch": torch.__version__,
        },
    }
    e3.validate_bundle(bundle)
    for case in cases:
        if any(word != "00000000" for row in case["synthetic"]["batches"]
               for word in row["logits_f32"]):
            raise AssertionError("independent zero M3 forward did not yield zero logits")
        expected_accuracy = sum(value == 0 for value in
                                next(row for row in CASES if row[0] == case["name"])[2]) / 5
        if case["synthetic"]["final"]["metrics_f32"][3] != f32_bits(expected_accuracy):
            raise AssertionError("independent zero M3 tie/correctness outcome changed")
    (output / "reference.json").write_bytes(encode(bundle))
    record = {"kind": "e3-zero-control-corpora-v1", "model_parameter_sha256":
              bundle["model"]["parameter_f32_sha256"], "cases": physical}
    (output / "corpora.json").write_bytes(canonical(record))
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    create(parser.parse_args().output)


if __name__ == "__main__":
    main()
