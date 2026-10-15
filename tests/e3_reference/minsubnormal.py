"""Development-only finite subnormal C1/D2 fixture and independent M3 oracle.

This generates expected data, never an E3 observation or a model parameter setter.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import platform
import struct

from tests.c1.test_checkpoint_format import build_container, parse_golden
from tests.d1.reference_format import parse_corpus
from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_reference.corpus import FINGERPRINT, IDENTITY, PROFILE, VOCAB_SIZE
from tests.e3_reference.transcript import canonical, encode, f32_bits
from tests.e3_reference.uniform_tie import ALIAS, PROVIDER, SHAPES
from tests.t1.reference import encode_artifact, encode_bytes


ROOT = Path(__file__).resolve().parents[2]
NAME = "finite-minsubnormal"
RAW = bytes(6)
MIN_SUBNORMAL = struct.pack("<I", 1)
ONE = struct.pack("<f", 1.0)
CONFIG = ReferenceConfig(1, 2, None, 1, True)


def _payload(path: str, shape: tuple[int, ...], *, clear_bias: bool) -> bytes:
    data = bytearray(4 * math.prod(shape))
    if path == "norm_final/bias" and not clear_bias:
        data[:4] = MIN_SUBNORMAL
    if path in ("head/weight", "token_embedding/weight"):
        data[:4] = ONE
    return bytes(data)


def checkpoint(*, clear_bias: bool = False) -> bytes:
    entries = tuple(
        (tuple(segment.encode("ascii") for segment in path.split("/")),
         1, 3, shape, _payload(path, shape, clear_bias=clear_bias))
        for path, shape in SHAPES
    )
    raw = build_container(entries, (ALIAS,), PROVIDER)
    parsed = parse_golden(raw, PROVIDER)
    if len(parsed.records) != 15 or len(parsed.aliases) != 1 or \
            parsed.aliases[0].members != ALIAS:
        raise AssertionError("C1 path/tie topology changed")
    unique_bytes = 0
    for (path, shape), record in zip(SHAPES, parsed.records, strict=True):
        payload = raw[record.payload_start:record.payload_start + record.payload_bytes]
        if record.path != tuple(part.encode("ascii") for part in path.split("/")) or \
                record.kind != 1 or record.dtype != 3 or record.device != 1 or \
                record.layout != 1 or record.rank != len(shape) or \
                record.elements != math.prod(shape) or \
                record.payload_bytes != 4 * math.prod(shape) or \
                payload != _payload(path, shape, clear_bias=clear_bias):
            raise AssertionError("C1 f32 geometry or payload changed")
        if path != "token_embedding/weight":
            unique_bytes += record.payload_bytes
    if unique_bytes != 4736:
        raise AssertionError("C1 unique parameter byte count changed")
    return raw


def _parameters(*, clear_bias: bool):
    # Independently spell the mathematical state; never decode the C1 fixture
    # or read E3 output to manufacture expected logits.
    from tests.m3 import reference as m3

    if tuple(m3.PARAMETERS) != SHAPES[:-1]:
        raise AssertionError("M3 parameter roster changed")
    parameters = {path: m3.torch.zeros(shape, dtype=m3.torch.float64)
                  for path, shape in m3.PARAMETERS}
    parameters["head/weight"][0, 0] = 1.0
    if not clear_bias:
        parameters["norm_final/bias"][0] = struct.unpack("<f", MIN_SUBNORMAL)[0]
    return parameters


def _reference(case: dict[str, object], *, clear_bias: bool):
    from tests.e3_reference.reference import _synthetic_case, validate_bundle
    from tests.m3 import reference as m3

    m3.setup()
    parameters = _parameters(clear_bias=clear_bias)
    synthetic, mathematical, loss = _synthetic_case(case["batches"], parameters)
    expected = "00000000" if clear_bias else "00000001"
    if any(batch["logits_f32"][position] != expected
           for batch in synthetic["batches"] for position in (0, 256)) or \
            any(any(word != "00000000" for index, word in enumerate(batch["logits_f32"])
                    if index not in (0, 256)) for batch in synthetic["batches"]):
        raise AssertionError("M3 oracle failed finite subnormal logit geometry")
    _, successor = m3.parameters(1729)
    unique = b"".join(bytes.fromhex(f32_bits(item))
                      for path, _ in m3.PARAMETERS
                      for item in parameters[path].reshape(-1).tolist())
    result = {
        "kind": "reference-bundle", "profile": m3.PROFILE,
        "cases": [{"name": NAME, "corpus": case, "synthetic": synthetic,
                   "mathematical": mathematical, "mathematical_loss_f64": loss}],
        "model": {"initializer_seed": 1729, "initializer_successor": list(successor),
                  "parameter_f32_sha256": hashlib.sha256(unique).hexdigest(),
                  "reference_manifest_sha256": hashlib.sha256(
                      (ROOT / "tests/m3/reference_manifest.json").read_bytes()).hexdigest()},
        "provenance": {
            "corpus_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "generator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "m3_reference_source_sha256": hashlib.sha256(
                (ROOT / "tests/m3/reference.py").read_bytes()).hexdigest(),
            "initializer_source_sha256": hashlib.sha256(
                (ROOT / "tests/n3k/test_initializer_reference.py").read_bytes()).hexdigest(),
            "rational_reference_source_sha256": hashlib.sha256(
                (ROOT / "tests/e3_metrics/reference_probe.py").read_bytes()).hexdigest(),
            "q0_lock_sha256": hashlib.sha256(
                (ROOT / "tests/q0/requirements-oracle.lock").read_bytes()).hexdigest(),
            "python": platform.python_version(), "torch": m3.torch.__version__,
        },
    }
    validate_bundle(result)
    return result


def require_first_logit(reference: dict, candidate: dict) -> None:
    for baseline, changed in zip(reference["cases"][0]["synthetic"]["batches"],
                                 candidate["cases"][0]["synthetic"]["batches"], strict=True):
        if baseline["logits_f32"][0] != changed["logits_f32"][0]:
            raise AssertionError("logit0 differs")


def materialize(output: Path) -> tuple[dict, dict, dict]:
    if output.exists() or output.is_symlink() or output.parent.resolve() != output.parent.absolute():
        raise ValueError("fixture output must be fresh and canonical")
    output.mkdir(parents=True)
    artifact = encode_artifact(IDENTITY)
    if encode_bytes(IDENTITY, RAW) != tuple(RAW):
        raise AssertionError("raw tokenizer bytes changed")
    (output / "tokenizer.etok").write_bytes(artifact)
    ordinary, mutant = checkpoint(), checkpoint(clear_bias=True)
    (output / "finite-state.etcp").write_bytes(ordinary)
    (output / "bias-clear.etcp").write_bytes(mutant)
    corpus = output / NAME
    corpus.mkdir()
    manifest = write_d1_resource(corpus, (tuple(RAW),),
                                 fingerprint=FINGERPRINT, vocab=VOCAB_SIZE)
    loaded, digest = load_d1_resource(corpus, expected_fingerprint=FINGERPRINT,
                                      expected_vocab=VOCAB_SIZE, maximum_total_tokens=6)
    if tuple(parse_corpus(corpus)["tokens"]) != tuple(RAW):
        raise AssertionError("D1 token bytes changed")
    dataset = ReferenceDataset(loaded, CONFIG, manifest_digest=digest,
                               tokenizer_fingerprint=FINGERPRINT, vocab_size=VOCAB_SIZE)
    before = dataset.snapshot().hex()
    batches = []
    while (row := dataset.next_batch()) is not None:
        batches.append({"inputs": list(row.inputs[0]), "targets": list(row.targets[0]),
                        "mask": list(row.loss_mask[0])})
    after = dataset.snapshot().hex()
    if [batch["mask"] for batch in batches] != [[True, True], [True, True], [True, False]] or \
            any(batch["inputs"] != [0, 0] or batch["targets"] != [0, 0]
                for batch in batches):
        raise AssertionError("authentic D2 row geometry changed")
    files = sorted((p.name, p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
                   for p in corpus.iterdir())
    payload = {
        "kind": "e3-minsubnormal-fixture-v1", "profile": PROFILE,
        "tokenizer_sha256": hashlib.sha256(artifact).hexdigest(),
        "checkpoints": {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                        for name, raw in (("finite-state.etcp", ordinary),
                                          ("bias-clear.etcp", mutant))},
        "provider": PROVIDER.decode("ascii"), "entry_count": 15,
        "unique_parameter_bytes": 4736, "alias_indices": list(ALIAS),
        "source_bytes": RAW.hex(), "d1_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "d1_trailer_sha256": digest.hex(), "d1_files": files,
        "packing": True, "batch_size": 1, "sequence_length": 2,
        "start_cursor": before, "end_cursor": after, "batches": batches,
    }
    case = {"name": NAME, "batches": batches, "start_cursor": before,
            "end_cursor": after, "source_bytes": RAW.hex(),
            "d1_manifest_sha256": payload["d1_manifest_sha256"], "d1_files": files}
    ordinary_ref = _reference(case, clear_bias=False)
    mutant_ref = _reference(case, clear_bias=True)
    (output / "fixture.json").write_bytes(canonical(payload))
    (output / "reference.json").write_bytes(encode(ordinary_ref))
    (output / "bias-clear-reference.json").write_bytes(encode(mutant_ref))
    return payload, ordinary_ref, mutant_ref
