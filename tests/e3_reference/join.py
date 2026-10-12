"""Join the five frozen E3 development cases to their physical D1/D2 sources.

This checks reference data only. It neither reads nor certifies private AOT output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource
from tests.e3_reference.corpus import (
    CASES, FINGERPRINT, PROFILE, SOURCE_BYTES, VOCAB_SIZE, validate_payload,
)
from tests.e3_reference.transcript import (
    TranscriptError, bits_f64, canonical, encode, validate_mathematical_case,
    validate_observed_case,
)


ROLE_NAMES = (
    "ET", "EP", "X", "N1", "QT", "KT", "VT", "QH", "KH", "VH", "AH",
    "AT", "AO", "R", "N2", "FU", "FG", "FD", "Y", "NF", "Z",
)
ROOT = Path(__file__).resolve().parents[2]
# Canonical envelopes from the pinned five-case Q0 output. A changed oracle
# needs a separately reviewed re-pin; source hashes alone cannot authenticate
# intermediate role values or the initialized parameter image.
FROZEN_CORPUS_SHA256 = "97f4652b1cf5eed34c54cdbabcb33451b8e43e274a4478143dfe2b3483e9bacb"
FROZEN_BUNDLE_SHA256 = "2bc7b75821bc757830b2733d9147abeb166dbb7f25f028bcd46aed0503f820ee"
MODEL_FIELDS = {
    "initializer_seed", "initializer_successor", "parameter_f32_sha256",
    "reference_manifest_sha256",
}
PROVENANCE_FIELDS = {
    "corpus_source_sha256", "generator_source_sha256",
    "initializer_source_sha256", "m3_reference_source_sha256", "python",
    "q0_lock_sha256", "rational_reference_source_sha256", "torch",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TranscriptError(message)


def _digest(value: object, where: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value),
        f"{where}: expected lowercase SHA-256",
    )
    return value


def _row(batch) -> dict[str, object]:
    return {
        "inputs": list(batch.inputs[0]),
        "targets": list(batch.targets[0]),
        "mask": list(batch.loss_mask[0]),
    }


def _roles(rows: object, expected_count: int, where: str) -> None:
    _require(isinstance(rows, list) and len(rows) == expected_count,
             f"{where}: mathematical row count mismatch")
    for index, row in enumerate(rows):
        label = f"{where}.mathematical[{index}]"
        _require(isinstance(row, dict) and set(row) == {
            "ce_f64", "inputs", "mask", "ordinal", "roles", "targets"
        }, f"{label}: fields differ from version 1")
        _require(type(row["ordinal"]) is int and row["ordinal"] == index,
                 f"{label}: ordinal mismatch")
        roles = row["roles"]
        _require(isinstance(roles, list) and len(roles) == len(ROLE_NAMES),
                 f"{label}: expected 21 roles")
        for name, role in zip(ROLE_NAMES, roles):
            _require(isinstance(role, dict) and set(role) == {
                "name", "shape", "values_f64"
            } and role["name"] == name, f"{label}: role name or fields mismatch")
            shape = ([1, 2, 256] if name == "Z" else
                     [1, 2, 8] if name in {"FU", "FG"} else
                     [1, 2, 2, 2] if name in {"QH", "KH", "VH", "AH"} else
                     [1, 2, 4])
            values = role["values_f64"]
            count = 1
            for axis in shape:
                count *= axis
            _require(type(role["shape"]) is list and
                     canonical(role["shape"]) == canonical(shape) and
                     isinstance(values, list)
                     and len(values) == count, f"{label}.{name}: shape mismatch")
            for word in values:
                bits_f64(word, f"{label}.{name}")


def validate_five_case_join(
    corpus_root: Path, corpus_payload: dict[str, object],
    bundle: dict[str, object],
) -> tuple[dict[str, object], ...]:
    """Return checked case IDs, D1 digests, D2 endpoints and oracle records.

    Inputs must be decoded with ``transcript.decode`` to reject duplicate JSON
    keys and noncanonical envelopes before this semantic join.
    """

    validate_payload(corpus_payload)
    _require(hashlib.sha256(encode(corpus_payload)).hexdigest() == FROZEN_CORPUS_SHA256,
             "frozen corpus reference digest mismatch")
    _require(isinstance(bundle, dict) and set(bundle) == {
        "cases", "kind", "model", "profile", "provenance"
    } and bundle["kind"] == "reference-bundle", "bundle fields/kind mismatch")
    manifest_path = ROOT / "tests/m3/reference_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    _require(canonical(bundle["profile"]) == canonical(manifest["profile"]),
             "bundle model profile mismatch")
    model = bundle["model"]
    _require(isinstance(model, dict) and set(model) == MODEL_FIELDS
             and type(model["initializer_seed"]) is int
             and model["initializer_seed"] == 1729
             and canonical(model["initializer_successor"]) ==
             canonical([1, 1729, 290, 0]),
             "bundle model fields/identity mismatch")
    _digest(model["parameter_f32_sha256"], "model.parameter_f32_sha256")
    _digest(model["reference_manifest_sha256"], "model.reference_manifest_sha256")
    _require(model["reference_manifest_sha256"] ==
             hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
             "model reference manifest digest mismatch")
    provenance = bundle["provenance"]
    _require(isinstance(provenance, dict) and set(provenance) == PROVENANCE_FIELDS,
             "bundle provenance fields mismatch")
    for field in PROVENANCE_FIELDS:
        if field.endswith("sha256"):
            _digest(provenance[field], f"provenance.{field}")
        else:
            _require(isinstance(provenance[field], str) and bool(provenance[field]),
                     f"provenance.{field}: expected nonempty string")
    _require(provenance["python"] == "3.14.6" and
             provenance["torch"] == "2.13.0+cpu",
             "bundle Q0 runtime provenance mismatch")
    _require(hashlib.sha256(encode(bundle)).hexdigest() == FROZEN_BUNDLE_SHA256,
             "frozen reference bundle digest mismatch")
    for field, path in {
        "corpus_source_sha256": ROOT / "tests/e3_reference/corpus.py",
        "generator_source_sha256": ROOT / "tests/e3_reference/reference.py",
        "initializer_source_sha256": ROOT / "tests/n3k/test_initializer_reference.py",
        "m3_reference_source_sha256": ROOT / "tests/m3/reference.py",
        "q0_lock_sha256": ROOT / "tests/q0/requirements-oracle.lock",
        "rational_reference_source_sha256": ROOT / "tests/e3_metrics/reference_probe.py",
    }.items():
        _require(provenance[field] == hashlib.sha256(path.read_bytes()).hexdigest(),
                 f"provenance.{field}: source digest mismatch")
    cases = bundle["cases"]
    _require(isinstance(cases, list) and len(cases) == len(CASES),
             "bundle case count mismatch")
    _require(corpus_root.is_dir() and not corpus_root.is_symlink()
             and corpus_root.resolve() == corpus_root.absolute(),
             "corpus root must be a real directory")
    artifact = corpus_root / "tokenizer.etok"
    _require(artifact.is_file() and not artifact.is_symlink()
             and hashlib.sha256(artifact.read_bytes()).hexdigest() ==
             corpus_payload["tokenizer"]["artifact_sha256"],
             "physical T1 artifact digest mismatch")

    joined = []
    for index, (name, shards, packed, start_ordinal) in enumerate(CASES):
        source = corpus_payload["cases"][index]
        case = cases[index]
        _require(isinstance(case, dict) and set(case) == {
            "corpus", "mathematical", "mathematical_loss_f64", "name", "synthetic"
        } and case["name"] == name and
                 canonical(case["corpus"]) == canonical(source),
                 f"{name}: case ID or corpus join mismatch")
        directory = corpus_root / name
        files = source["corpus"]["files"]
        _require(directory.is_dir() and not directory.is_symlink()
                 and sorted(path.name for path in directory.iterdir()) ==
                 sorted(item["path"] for item in files),
                 f"{name}: physical D1 file set mismatch")
        for item in files:
            path = directory / item["path"]
            _require(path.is_file() and not path.is_symlink()
                     and path.stat().st_size == item["bytes"]
                     and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"],
                     f"{name}: physical D1 byte/digest mismatch")
        loaded, digest = load_d1_resource(
            directory, expected_fingerprint=FINGERPRINT,
            expected_vocab=VOCAB_SIZE, maximum_total_tokens=len(SOURCE_BYTES),
        )
        _require(loaded == shards and digest.hex() ==
                 source["corpus"]["manifest_trailer_sha256"],
                 f"{name}: D1 shard/manifest mismatch")
        dataset = ReferenceDataset(
            loaded, ReferenceConfig(1, 2, None, 1, packed),
            manifest_digest=digest, tokenizer_fingerprint=FINGERPRINT,
            vocab_size=VOCAB_SIZE,
        )
        for _ in range(start_ordinal):
            _require(dataset.next_batch() is not None, f"{name}: invalid suffix")
        _require(dataset.snapshot().hex() == source["start_cursor"],
                 f"{name}: exact D2 start cursor mismatch")
        rows = []
        while (batch := dataset.next_batch()) is not None:
            rows.append(_row(batch))
        _require(rows == source["batches"], f"{name}: D2 emitted rows mismatch")
        synthetic = case["synthetic"]
        _require(isinstance(synthetic, dict) and
                 synthetic.get("kind") == "synthetic-observation" and
                 synthetic.get("profile") == PROFILE,
                 f"{name}: synthetic label/profile mismatch")
        _roles(case["mathematical"], len(rows), name)
        validate_observed_case(synthetic, tuple(rows))
        validate_mathematical_case(
            synthetic, case["mathematical"], tuple(rows),
            case["mathematical_loss_f64"],
        )
        joined.append({
            "name": name,
            "manifest_trailer_sha256": digest.hex(),
            "start_cursor": source["start_cursor"],
            "end_cursor": dataset.snapshot().hex(),
            "rows": rows,
            "mathematical": case["mathematical"],
            "synthetic": synthetic,
        })

    single, sharded, suffix, three, pairs = joined
    _require(single["rows"] == sharded["rows"] and
             single["synthetic"] == sharded["synthetic"] and
             single["mathematical"] == sharded["mathematical"] and
             single["manifest_trailer_sha256"] != sharded["manifest_trailer_sha256"] and
             single["end_cursor"] != sharded["end_cursor"],
             "packed repartition equality mismatch")
    _require(suffix["rows"] == single["rows"][1:] and
             suffix["manifest_trailer_sha256"] == sharded["manifest_trailer_sha256"] and
             suffix["start_cursor"] != sharded["start_cursor"] and
             suffix["end_cursor"] == sharded["end_cursor"],
             "packed suffix/cursor mismatch")
    for offset, row in enumerate(suffix["synthetic"]["batches"]):
        prior = single["synthetic"]["batches"][offset + 1]
        _require(all(row[field] == prior[field] for field in (
            "inputs", "targets", "mask", "logits_f32", "ce_f32",
            "correct_f32", "reduction_f32"
        )), "packed suffix per-row numerical mismatch")
        _require(suffix["mathematical"][offset]["roles"] ==
                 single["mathematical"][offset + 1]["roles"],
                 "packed suffix role mismatch")
        _require(all(suffix["mathematical"][offset][field] ==
                     single["mathematical"][offset + 1][field]
                     for field in ("inputs", "targets", "mask", "ce_f64")),
                 "packed suffix mathematical row mismatch")
    _require(three["rows"] != pairs["rows"] and
             three["rows"] != single["rows"] and
             pairs["rows"] != single["rows"] and
             [sum(row["mask"]) for row in three["rows"]] == [2, 2] and
             [sum(row["mask"]) for row in pairs["rows"]] == [1, 1, 1],
             "unpacked boundary difference mismatch")
    _require(pairs["manifest_trailer_sha256"] == sharded["manifest_trailer_sha256"]
             and three["manifest_trailer_sha256"] != sharded["manifest_trailer_sha256"]
             and len(single["rows"]) == len(pairs["rows"]) == 3
             and sum(sum(row["mask"]) for row in single["rows"]) == 5
             and sum(sum(row["mask"]) for row in three["rows"]) == 4
             and sum(sum(row["mask"]) for row in pairs["rows"]) == 3,
             "unpacked D1 digest or active-token difference mismatch")
    return tuple(joined)
