"""Development-only N3 oracle for an authentic zero-parameter M3 model.

The generated predictions are not observations of a compiled evaluator.  All
corpora pass through the independent D1 writer/reader and D2 row/cursor oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform

from tests.d1.reference_format import parse_corpus
from tests.d2.reference import ReferenceConfig, ReferenceDataset, decode_reference_cursor
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_reference import reference
from tests.e3_reference.corpus import FINGERPRINT, IDENTITY, VOCAB_SIZE
from tests.e3_reference.transcript import (
    FORMAT, MAX_BYTES, VERSION, canonical, encode, validate_mathematical_case,
    validate_observed_case,
)
from tests.m3 import reference as m3
from tests.t1.reference import encode_artifact, encode_bytes


ROOT = Path(__file__).resolve().parents[2]
PROFILE = "e3-n3-zero-model-reference-v1"
# This independent table is deliberately not imported from an installed M3
# descriptor.  It is checked against the separate development M3 reference.
SHAPES = (
    ("blocks/0/attention/key/weight", (4, 4)),
    ("blocks/0/attention/output/weight", (4, 4)),
    ("blocks/0/attention/query/weight", (4, 4)),
    ("blocks/0/attention/value/weight", (4, 4)),
    ("blocks/0/ffn/down/weight", (4, 8)),
    ("blocks/0/ffn/up/weight", (8, 4)),
    ("blocks/0/norm1/bias", (4,)),
    ("blocks/0/norm1/weight", (4,)),
    ("blocks/0/norm2/bias", (4,)),
    ("blocks/0/norm2/weight", (4,)),
    ("head/weight", (256, 4)),
    ("norm_final/bias", (4,)),
    ("norm_final/weight", (4,)),
    ("position_embedding/weight", (2, 4)),
)
# Each sequence has real next-token targets.  The last row is masked 10.
CASES = (
    ("all-correct", bytes((0, 0, 0, 0, 0, 0))),
    ("all-wrong", bytes((7, 7, 7, 7, 7, 7))),
    ("mixed-padded", bytes((3, 0, 4, 0, 5, 0))),
    ("one-valid-padded", bytes((9, 0))),
)
CONFIG = ReferenceConfig(1, 2, None, 1, True)
ROLE_SHAPES = {
    name: ([1, 2, 256] if name == "Z" else [1, 2, 8] if name in ("FU", "FG")
           else [1, 2, 2, 2] if name in ("QH", "KH", "VH", "AH") else [1, 2, 4])
    for name in reference.ROLE_NAMES
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(directory: Path) -> list[dict[str, object]]:
    if any(path.is_symlink() or not path.is_file() for path in directory.iterdir()):
        raise ValueError("zero-model D1 directory contains a nonregular file")
    return [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha(path)}
        for path in sorted(directory.iterdir())
    ]


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate zero-model JSON key: {key}")
        result[key] = value
    return result


def load(output: Path) -> dict[str, object]:
    """Authenticate canonical record and physical T1/D1 resource bytes."""

    path = output / "zero-model.json"
    if path.is_symlink() or output.is_symlink():
        raise ValueError("zero-model input must not be a symlink")
    if not 1 <= path.stat().st_size <= MAX_BYTES:
        raise ValueError("zero-model record exceeds size limit")
    raw = path.read_bytes()
    try:
        document = json.loads(raw, object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("zero-model record is not JSON") from error
    if not isinstance(document, dict) or set(document) != {
        "format", "payload", "payload_sha256", "version"
    } or document["format"] != FORMAT or document["version"] != VERSION:
        raise ValueError("zero-model record envelope differs")
    payload = document["payload"]
    if not isinstance(payload, dict) or not isinstance(document["payload_sha256"], str):
        raise ValueError("zero-model record payload/checksum type differs")
    if hashlib.sha256(canonical(payload)).hexdigest() != document["payload_sha256"] or encode(payload) != raw:
        raise ValueError("zero-model record checksum or canonical bytes differ")
    validate(payload)
    artifact = output / "tokenizer.etok"
    if artifact.is_symlink() or artifact.stat().st_size != payload["tokenizer"]["artifact_bytes"] or _sha(artifact) != payload["tokenizer"]["artifact_sha256"]:
        raise ValueError("zero-model tokenizer file differs")
    for case in payload["cases"]:
        directory = output / case["name"]
        if directory.is_symlink():
            raise ValueError("zero-model D1 directory is a symlink")
        if _files(directory) != case["d1"]["files"]:
            raise ValueError("zero-model D1 files differ")
        loaded, digest = load_d1_resource(
            directory, expected_fingerprint=FINGERPRINT,
            expected_vocab=VOCAB_SIZE, maximum_total_tokens=len(bytes.fromhex(case["source_bytes"])),
        )
        if loaded != (tuple(bytes.fromhex(case["source_bytes"])),) or digest.hex() != case["d1"]["manifest_trailer_sha256"]:
            raise ValueError("zero-model D1 resource differs")
    return payload


def zero_parameters():
    """Create 14 distinct CPU f32 owners, promoted to f64 for M3 equations."""

    if SHAPES != m3.PARAMETERS or m3.TIED != "head/weight":
        raise AssertionError("independent fixed14 shape/tie contract differs from M3")
    m3.setup()
    owners = {
        name: m3.torch.zeros(shape, dtype=m3.torch.float32)
        for name, shape in SHAPES
    }
    if len({id(value) for value in owners.values()}) != 14:
        raise AssertionError("zero parameter owners are not distinct")
    logical = dict(owners)
    logical["token_embedding/weight"] = owners["head/weight"]
    if len(logical) != 15 or logical["token_embedding/weight"] is not logical["head/weight"]:
        raise AssertionError("tied head identity differs")
    count = sum(math.prod(shape) for _, shape in SHAPES)
    if count != 1184:
        raise AssertionError("fixed14 element count differs")
    payload = b"\x00" * (count * 4)
    if len(payload) != 4736:
        raise AssertionError("fixed14 payload byte count differs")
    return {name: value.to(m3.torch.float64) for name, value in owners.items()}, payload


def _batch(batch) -> dict[str, object]:
    return {
        "inputs": list(batch.inputs[0]),
        "targets": list(batch.targets[0]),
        "mask": list(batch.loss_mask[0]),
    }


def _case(output: Path, name: str, source: bytes, parameters) -> dict[str, object]:
    directory = output / name
    directory.mkdir()
    shards = (tuple(source),)
    manifest = write_d1_resource(directory, shards, fingerprint=FINGERPRINT, vocab=VOCAB_SIZE)
    loaded, digest = load_d1_resource(
        directory, expected_fingerprint=FINGERPRINT, expected_vocab=VOCAB_SIZE,
        maximum_total_tokens=len(source),
    )
    if loaded != shards or tuple(parse_corpus(directory)["tokens"]) != tuple(source):
        raise AssertionError("D1 round trip differs from source bytes")
    dataset = ReferenceDataset(
        loaded, CONFIG, manifest_digest=digest,
        tokenizer_fingerprint=FINGERPRINT, vocab_size=VOCAB_SIZE,
    )
    start = dataset.snapshot()
    batches = []
    while (batch := dataset.next_batch()) is not None:
        batches.append(_batch(batch))
    end = dataset.snapshot()
    if len(batches) != len(dataset.rows):
        raise AssertionError("D2 row count differs")
    replay = ReferenceDataset(
        loaded, CONFIG, manifest_digest=digest,
        tokenizer_fingerprint=FINGERPRINT, vocab_size=VOCAB_SIZE,
    )
    replay.seek(start)
    replay_rows = []
    while (batch := replay.next_batch()) is not None:
        replay_rows.append(_batch(batch))
    if replay_rows != batches or replay.snapshot() != end:
        raise AssertionError("D2 cursor replay differs")
    prediction, roles, loss = reference._synthetic_case(batches, parameters)
    validate_observed_case(prediction, tuple(batches))
    validate_mathematical_case(prediction, roles, tuple(batches), loss)
    return {
        "name": name,
        "source_bytes": source.hex(),
        "d2": {
            "config": {"batch_size": 1, "sequence_length": 2, "packing": True,
                       "shuffle_seed": None, "shuffle_window": 1},
            "start_cursor": start.hex(), "end_cursor": end.hex(),
            "expected_batch_count": len(batches), "batches": batches,
        },
        "d1": {"manifest_trailer_sha256": manifest[-32:].hex(), "files": _files(directory)},
        "oracle": {"mathematical": roles, "mathematical_loss_f64": loss,
                   "predicted_f32": prediction},
    }


def build(output: Path) -> dict[str, object]:
    """Materialize a fresh tiny resource tree and return its strict oracle record."""

    if output.exists() or output.is_symlink():
        raise FileExistsError("zero-model output must be a fresh path")
    if output.parent.is_symlink() or output.parent.resolve() != output.parent.absolute():
        raise ValueError("zero-model output parent must not be a symlink")
    output.mkdir(parents=True)
    parameters, zero_bytes = zero_parameters()
    artifact = encode_artifact(IDENTITY)
    (output / "tokenizer.etok").write_bytes(artifact)
    for _, source in CASES:
        if encode_bytes(IDENTITY, source) != tuple(source):
            raise AssertionError("raw tokenizer changed byte IDs")
    cases = [_case(output, name, source, parameters) for name, source in CASES]
    result = {
        "kind": "e3-n3-zero-model-oracle", "profile": PROFILE,
        "model": {
            "parameter_shapes": {name: list(shape) for name, shape in SHAPES},
            "parameter_count": 14, "element_count": 1184, "byte_count": 4736,
            "parameter_f32_sha256": hashlib.sha256(zero_bytes).hexdigest(),
            "logical_tie": ["head/weight", "token_embedding/weight"],
            "control": "idle-real-model-state-load-proposed",
        },
        "tokenizer": {"artifact_bytes": len(artifact),
                      "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
                      "fingerprint": FINGERPRINT, "vocab_size": VOCAB_SIZE},
        "provenance": {
            "python": platform.python_version(), "torch": m3.torch.__version__,
            "q0_lock_sha256": _sha(ROOT / "tests/q0/requirements-oracle.lock"),
            "generator_sha256": _sha(Path(__file__)),
            "d2_reference_sha256": _sha(ROOT / "tests/d2/reference.py"),
            "m3_reference_sha256": _sha(ROOT / "tests/m3/reference.py"),
            "role_reference_sha256": _sha(ROOT / "tests/e3_reference/reference.py"),
            "rational_reference_sha256": _sha(ROOT / "tests/e3_metrics/reference_probe.py"),
        },
        "cases": cases,
    }
    validate(result)
    raw = encode(result)
    if len(raw) > MAX_BYTES:
        raise ValueError("zero-model record exceeds the transcript byte limit")
    (output / "zero-model.json").write_bytes(raw)
    return result


def validate(payload: dict[str, object]) -> None:
    """Reject altered identity, rows, roles, metrics, or cursor metadata."""

    if not isinstance(payload, dict) or set(payload) != {
        "kind", "profile", "model", "tokenizer", "provenance", "cases"
    } or payload["kind"] != "e3-n3-zero-model-oracle" or payload["profile"] != PROFILE:
        raise ValueError("zero-model top-level schema differs")
    model = payload["model"]
    if not isinstance(model, dict) or set(model) != {
        "parameter_shapes", "parameter_count", "element_count", "byte_count",
        "parameter_f32_sha256", "logical_tie", "control"
    } or model != {
        "parameter_shapes": {name: list(shape) for name, shape in SHAPES},
        "parameter_count": 14, "element_count": 1184, "byte_count": 4736,
        "parameter_f32_sha256": hashlib.sha256(b"\x00" * 4736).hexdigest(),
        "logical_tie": ["head/weight", "token_embedding/weight"],
        "control": "idle-real-model-state-load-proposed",
    }:
        raise ValueError("zero-model parameter schema differs")
    tokenizer = payload["tokenizer"]
    artifact = encode_artifact(IDENTITY)
    if tokenizer != {"artifact_bytes": len(artifact),
                     "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
                     "fingerprint": FINGERPRINT, "vocab_size": VOCAB_SIZE}:
        raise ValueError("zero-model tokenizer differs")
    provenance = payload["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {
        "python", "torch", "q0_lock_sha256", "generator_sha256",
        "d2_reference_sha256", "m3_reference_sha256",
        "role_reference_sha256", "rational_reference_sha256"
    } or provenance != {
        "python": "3.14.6", "torch": "2.13.0+cpu",
        "q0_lock_sha256": _sha(ROOT / "tests/q0/requirements-oracle.lock"),
        "generator_sha256": _sha(Path(__file__)),
        "d2_reference_sha256": _sha(ROOT / "tests/d2/reference.py"),
        "m3_reference_sha256": _sha(ROOT / "tests/m3/reference.py"),
        "role_reference_sha256": _sha(ROOT / "tests/e3_reference/reference.py"),
        "rational_reference_sha256": _sha(ROOT / "tests/e3_metrics/reference_probe.py"),
    }:
        raise ValueError("zero-model pinned provenance differs")
    cases = payload["cases"]
    if not isinstance(cases, list) or len(cases) != len(CASES):
        raise ValueError("zero-model case count differs")
    for case, (name, source) in zip(cases, CASES):
        if not isinstance(case, dict) or set(case) != {
            "name", "source_bytes", "d1", "d2", "oracle"
        } or case["name"] != name or case["source_bytes"] != source.hex():
            raise ValueError("zero-model case identity differs")
        d1, d2, oracle = case["d1"], case["d2"], case["oracle"]
        if not isinstance(d1, dict) or set(d1) != {"files", "manifest_trailer_sha256"}:
            raise ValueError("zero-model D1 schema differs")
        if not isinstance(d1["files"], list) or len(d1["files"]) != 2 or any(
            not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}
            or type(item["bytes"]) is not int or item["bytes"] <= 0
            or not isinstance(item["sha256"], str) or len(item["sha256"]) != 64
            for item in d1["files"]
        ) or [item["path"] for item in d1["files"]] != ["manifest.etm", "shard-0000000000000000.ets"]:
            raise ValueError("zero-model D1 file manifest differs")
        if not isinstance(d2, dict) or set(d2) != {
            "config", "start_cursor", "end_cursor", "expected_batch_count", "batches"
        } or d2["config"] != {"batch_size": 1, "sequence_length": 2,
                              "packing": True, "shuffle_seed": None, "shuffle_window": 1}:
            raise ValueError("zero-model D2 schema differs")
        expected = ReferenceDataset(
            (tuple(source),), CONFIG, manifest_digest=bytes.fromhex(d1["manifest_trailer_sha256"]),
            tokenizer_fingerprint=FINGERPRINT, vocab_size=VOCAB_SIZE,
        )
        expected_rows = [
            {"inputs": list(row.inputs), "targets": list(row.targets),
             "mask": list(row.loss_mask)} for row in expected.rows
        ]
        if d2["batches"] != expected_rows or type(d2["expected_batch_count"]) is not int or d2["expected_batch_count"] != len(expected_rows):
            raise ValueError("zero-model D2 emitted rows differ")
        expected_start = expected.snapshot().hex()
        for _ in expected_rows:
            if expected.next_batch() is None:
                raise ValueError("zero-model D2 traversal ended early")
        expected_end = expected.snapshot().hex()
        if d2["start_cursor"] != expected_start or d2["end_cursor"] != expected_end:
            raise ValueError("zero-model D2 exact cursor differs")
        for cursor, ordinal in ((d2["start_cursor"], 0), (d2["end_cursor"], len(expected_rows))):
            fields = decode_reference_cursor(bytes.fromhex(cursor))
            if (fields.next_ordinal != ordinal or fields.total_rows != len(expected_rows)
                or fields.manifest_digest.hex() != d1["manifest_trailer_sha256"]
                or fields.tokenizer_fingerprint != FINGERPRINT or fields.vocab_size != VOCAB_SIZE
                or fields.config != CONFIG):
                raise ValueError("zero-model D2 cursor differs")
        if not isinstance(oracle, dict) or set(oracle) != {
            "mathematical", "mathematical_loss_f64", "predicted_f32"
        }:
            raise ValueError("zero-model oracle schema differs")
        prediction = oracle["predicted_f32"]
        if not isinstance(prediction, dict) or prediction.get("kind") != "synthetic-observation":
            raise ValueError("zero-model prediction kind differs")
        validate_observed_case(prediction, tuple(expected_rows))
        if not isinstance(oracle["mathematical"], list):
            raise ValueError("zero-model mathematical rows differ")
        for batch in oracle["mathematical"]:
            if not isinstance(batch, dict) or set(batch) != {
                "ce_f64", "inputs", "mask", "ordinal", "roles", "targets"
            } or not isinstance(batch["roles"], list):
                raise ValueError("zero-model mathematical row schema differs")
            for role in batch["roles"]:
                if not isinstance(role, dict) or set(role) != {
                    "name", "shape", "values_f64"
                }:
                    raise ValueError("zero-model role schema differs")
        validate_mathematical_case(
            prediction, oracle["mathematical"], tuple(expected_rows),
            oracle["mathematical_loss_f64"],
        )
        for batch in oracle["mathematical"]:
            roles = batch["roles"]
            if tuple(role["name"] for role in roles) != reference.ROLE_NAMES:
                raise ValueError("zero-model role schedule differs")
            for role in roles:
                if role["shape"] != ROLE_SHAPES[role["name"]] or len(role["values_f64"]) != math.prod(role["shape"]):
                    raise ValueError("zero-model role shape differs")
                if any(word != "0000000000000000" for word in role["values_f64"]):
                    raise ValueError("zero-model role must be positive zero")
        for batch in prediction["batches"]:
            if any(word != "00000000" for word in batch["logits_f32"]):
                raise ValueError("zero-model logit must be positive zero")
            if batch["ce_f32"][0] != batch["ce_f32"][1]:
                raise ValueError("zero-model cross-entropy must be uniform")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    build(arguments.output)


if __name__ == "__main__":
    main()
