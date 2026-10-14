"""Development-only seeded M3 target-offset D1/D2 corpus and oracle."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import platform

from tests.d1.reference_format import parse_corpus
from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_reference.transcript import bits_f32, canonical, encode, f32_bits
from tests.t1.reference import Identity, encode_artifact, encode_bytes, fingerprint


ROOT = Path(__file__).resolve().parents[2]
TOKENIZER = Identity("raw", (), (), ())
FINGERPRINT = fingerprint(TOKENIZER)
CONFIG = ReferenceConfig(1, 2, None, 1, True, maximum_batch_bytes=34)
CASES = (("target-zero", (3, 197, 0), (197, 0)),
         ("target-seven", (3, 197, 7), (197, 7)))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(directory: Path, tokens: tuple[int, ...]) -> tuple[list[dict[str, object]], str, str, str]:
    loaded, manifest_digest = load_d1_resource(
        directory, expected_fingerprint=FINGERPRINT,
        expected_vocab=256, maximum_total_tokens=3,
    )
    if tuple(parse_corpus(directory)["tokens"]) != tokens:
        raise ValueError("target-offset D1 bytes differ")
    dataset = ReferenceDataset(
        loaded, CONFIG, manifest_digest=manifest_digest,
        tokenizer_fingerprint=FINGERPRINT, vocab_size=256,
    )
    start_cursor = dataset.snapshot().hex()
    rows = []
    while (batch := dataset.next_batch()) is not None:
        rows.append({"inputs": list(batch.inputs[0]),
                     "targets": list(batch.targets[0]),
                     "mask": list(batch.loss_mask[0])})
    return rows, manifest_digest.hex(), start_cursor, dataset.snapshot().hex()


def materialize(output: Path) -> dict[str, object]:
    if (output.exists() or output.is_symlink() or output.parent.is_symlink()
            or output.parent.resolve() != output.parent.absolute()):
        raise ValueError("target-offset output must be fresh with a real parent")
    output.mkdir(parents=True)
    artifact = encode_artifact(TOKENIZER)
    (output / "tokenizer.etok").write_bytes(artifact)
    cases = []
    for name, tokens, targets in CASES:
        if encode_bytes(TOKENIZER, bytes(tokens)) != tokens:
            raise AssertionError("raw tokenizer changed the byte IDs")
        directory = output / name
        directory.mkdir()
        manifest = write_d1_resource(
            directory, (tokens,), fingerprint=FINGERPRINT, vocab=256,
        )
        rows, digest, start, end = _rows(directory, tokens)
        if rows != [{"inputs": [3, 197], "targets": list(targets),
                     "mask": [True, True]}] or digest != manifest[-32:].hex():
            raise AssertionError("genuine packed D2 target-offset row changed")
        cases.append({
            "name": name, "source_bytes": bytes(tokens).hex(), "batches": rows,
            "start_cursor": start, "end_cursor": end,
            "manifest_trailer_sha256": digest,
            "files": [
                {"path": path.name, "bytes": path.stat().st_size,
                 "sha256": _sha256(path)}
                for path in sorted(directory.iterdir())
            ],
        })
    payload = {
        "kind": "e3-target-offset-corpora-v1",
        "tokenizer": {"artifact_sha256": hashlib.sha256(artifact).hexdigest(),
                      "fingerprint": FINGERPRINT, "vocab_size": 256},
        "cases": cases,
    }
    (output / "corpora.json").write_bytes(canonical(payload))
    return payload


def validate_materialized(output: Path, payload: dict[str, object]) -> None:
    artifact = encode_artifact(TOKENIZER)
    if (output.is_symlink() or output.resolve() != output.absolute()
            or set(payload) != {"kind", "tokenizer", "cases"}
            or payload["kind"] != "e3-target-offset-corpora-v1"
            or payload["tokenizer"] != {
                "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
                "fingerprint": FINGERPRINT, "vocab_size": 256,
            }
            or len(payload.get("cases", ())) != len(CASES)):
        raise ValueError("target-offset corpus identity differs")
    if (output / "tokenizer.etok").read_bytes() != artifact:
        raise ValueError("target-offset tokenizer bytes differ")
    for case, (name, tokens, targets) in zip(payload["cases"], CASES):
        directory = output / name
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("target-offset corpus directory differs")
        rows, digest, start, end = _rows(directory, tokens)
        paths = sorted(directory.iterdir())
        if ([path.name for path in paths] !=
                ["manifest.etm", "shard-0000000000000000.ets"]
                or any(path.is_symlink() or not path.is_file() for path in paths)):
            raise ValueError("target-offset D1 file inventory differs")
        files = [
            {"path": path.name, "bytes": path.stat().st_size,
             "sha256": _sha256(path)}
            for path in paths
        ]
        if case != {"name": name, "source_bytes": bytes(tokens).hex(),
                    "batches": rows, "manifest_trailer_sha256": digest,
                    "start_cursor": start, "end_cursor": end,
                    "files": files} or rows != [
                        {"inputs": [3, 197], "targets": list(targets),
                         "mask": [True, True]}]:
            raise ValueError("target-offset corpus or D2 row differs")
    if (output / "corpora.json").read_bytes() != canonical(payload):
        raise ValueError("target-offset manifest bytes differ")


def build_reference(payload: dict[str, object]) -> dict[str, object]:
    # PyTorch stays in the pinned development oracle, never in the E3 runtime.
    from tests.e3_reference import reference as e3
    from tests.m3 import reference as m3

    m3.setup()
    parameters, successor = m3.parameters(1729)
    cases = []
    for corpus in payload["cases"]:
        synthetic, mathematical, loss = e3._synthetic_case(
            corpus["batches"], parameters,
        )
        cases.append({"name": corpus["name"], "corpus": corpus,
                      "synthetic": synthetic, "mathematical": mathematical,
                      "mathematical_loss_f64": loss})
    first, second = (case["synthetic"]["batches"][0] for case in cases)
    if (first["inputs"] != second["inputs"] or first["mask"] != second["mask"]
            or first["logits_f32"] != second["logits_f32"]
            or first["ce_f32"][0] != second["ce_f32"][0]
            or first["ce_f32"][1] == second["ce_f32"][1]):
        raise AssertionError("seeded target-offset CE witness disappeared")
    # Independently pinned seed-1729 row [3,197] from the five-layout oracle.
    logits = first["logits_f32"]
    ce_zero = bits_f32(first["ce_f32"][1], "target-zero CE")[1]
    ce_seven = bits_f32(second["ce_f32"][1], "target-seven CE")[1]
    if (logits[256] != "3c9e2d20" or logits[263] != "3c4aa83b"
            or first["ce_f32"] != ["40b28797", "40b0d5ab"]
            or not 0.006 < ce_seven - ce_zero < 0.008):
        raise AssertionError("seeded target-offset logits or CE delta drifted")
    words = b"".join(
        bytes.fromhex(f32_bits(value))
        for path, _ in m3.PARAMETERS
        for value in parameters[path].to(m3.torch.float32).detach().reshape(-1).tolist()
    )
    result = {
        "kind": "reference-bundle", "profile": m3.PROFILE, "cases": cases,
        "model": {
            "initializer_seed": 1729, "initializer_successor": list(successor),
            "parameter_f32_sha256": hashlib.sha256(words).hexdigest(),
            "reference_manifest_sha256": _sha256(ROOT / "tests/m3/reference_manifest.json"),
        },
        "provenance": {
            "corpus_source_sha256": _sha256(Path(__file__)),
            "generator_source_sha256": _sha256(Path(__file__)),
            "initializer_source_sha256": _sha256(ROOT / "tests/n3k/test_initializer_reference.py"),
            "m3_reference_source_sha256": _sha256(ROOT / "tests/m3/reference.py"),
            "q0_lock_sha256": _sha256(ROOT / "tests/q0/requirements-oracle.lock"),
            "rational_reference_source_sha256": _sha256(ROOT / "tests/e3_metrics/reference_probe.py"),
            "python": platform.python_version(), "torch": m3.torch.__version__,
        },
    }
    e3.validate_bundle(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = materialize(args.output)
    validate_materialized(args.output, payload)
    (args.output / "reference.json").write_bytes(encode(build_reference(payload)))


if __name__ == "__main__":
    main()
