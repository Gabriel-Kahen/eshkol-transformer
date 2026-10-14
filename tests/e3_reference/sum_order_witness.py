"""Bounded D1/D2 M2 witness selected by actual compiled CE observations.

The frozen probe records the first compiled distinction in a SHA-256 domain.
PyTorch supplies an independent mathematical reference, not the selection rule.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_metrics.reference_probe import add, reduce_bool, rounded, value
from tests.e3_reference.corpus import FINGERPRINT, IDENTITY
from tests.e3_reference.reference import _roles, role_trace
from tests.e3_reference.transcript import canonical, f32_bits
from tests.m3 import reference as m3
from tests.t1.reference import encode_artifact, encode_bytes


DOMAIN = b"e3-section9-sum-order-search-v1"
MAX_CANDIDATES = 64
SEED = 1729
FIXTURE = Path(__file__).with_name("sum_order_witness_v2.json")
PROBE = Path(__file__).with_name("sum_order_compiled_probe_v1.json")
PROBE_SHA256 = "f590f9912bfd02b507972076cb51bb57433342bf91cc744d86eac9d47baf6f1f"


def candidate(index: int) -> bytes:
    if type(index) is not int or not 0 <= index < MAX_CANDIDATES:
        raise ValueError("candidate index outside declared finite search")
    return hashlib.sha256(DOMAIN + index.to_bytes(4, "little")).digest()[:6]


def rows(raw: bytes, manifest_digest: bytes = bytes(32)) -> list[dict[str, object]]:
    if type(raw) is not bytes or len(raw) != 6:
        raise ValueError("expected exactly six raw V256 source bytes")
    if encode_bytes(IDENTITY, raw) != tuple(raw):
        raise AssertionError("raw T1 identity changed")
    dataset = ReferenceDataset(
        (tuple(raw),), ReferenceConfig(1, 2, None, 1, True),
        manifest_digest=manifest_digest, tokenizer_fingerprint=FINGERPRINT,
        vocab_size=256,
    )
    emitted = []
    while (batch := dataset.next_batch()) is not None:
        emitted.append({
            "inputs": list(batch.inputs[0]),
            "targets": list(batch.targets[0]),
            "mask": [int(bit) for bit in batch.loss_mask[0]],
        })
    if len(emitted) != 3:
        raise AssertionError("six-byte packed D2 corpus did not yield three rows")
    return emitted


def ce_numerators(parameters, emitted: list[dict[str, object]]):
    """Use registered seed-1729 M3 f64 forward, then accepted f32 CE boundary."""
    import torch

    result = []
    for row in emitted:
        logits = m3.forward(parameters, tuple(row["inputs"])).to(torch.float32)
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(2, 256),
            torch.tensor(row["targets"], dtype=torch.int64), reduction="none",
        )
        ce_words = tuple(int(f32_bits(item), 16) for item in ce.tolist())
        numerator, weight = reduce_bool(ce_words, tuple(row["mask"]))
        result.append((ce_words, numerator, weight))
    return result


def order_words(numerators: tuple[int, int, int]) -> dict[str, str]:
    a, b, c = numerators
    return {
        "serial": f"{add(add(a, b), c):08x}",
        "grouped": f"{add(a, add(b, c)):08x}",
        "wider": f"{rounded(value(a) + value(b) + value(c)):08x}",
    }


def batch_words_sha256(batches: list[dict[str, object]]) -> str:
    return hashlib.sha256(canonical([
        {"ce_f32": batch["ce_f32"], "numerator_f32": batch["numerator_f32"],
         "weight_f32": batch["weight_f32"]}
        for batch in batches
    ])).hexdigest()


def compiled_probe() -> tuple[dict[str, object], dict[str, object]]:
    """Authenticate and independently check the finite compiled search record."""
    raw_probe = PROBE.read_bytes()
    if hashlib.sha256(raw_probe).hexdigest() != PROBE_SHA256:
        raise ValueError("compiled probe hash differs")
    probe = json.loads(raw_probe)
    if (canonical(probe) != raw_probe or probe["version"] != 1
            or probe["source_head"] != "df9546cc6d5f39e53609e0fe41bb7190952508ad"
            or probe["source_tree"] != "365da582ed1fe7fa8fce32857bdc15dddf72cd39"
            or len(probe["rows"]) not in range(1, MAX_CANDIDATES + 1)):
        raise ValueError("compiled probe identity differs")
    for index, record in enumerate(probe["rows"]):
        raw = candidate(index)
        if record["index"] != index or record["raw"] != raw.hex():
            raise ValueError("compiled probe candidate order differs")
        emitted = rows(raw)
        ce = record["compiled_ce"]
        if len(ce) != 3 or any(len(pair) != 2 for pair in ce):
            raise ValueError("compiled probe CE shape differs")
        try:
            numerators = [
                reduce_bool(tuple(int(word, 16) for word in pair),
                            tuple(row["mask"]))[0]
                for pair, row in zip(ce, emitted)
            ]
        except (TypeError, ValueError) as error:
            raise ValueError("compiled probe CE words malformed") from error
        if ([f"{word:08x}" for word in numerators] != record["compiled_numerators"]
                or order_words(tuple(numerators)) != record["order"]):
            raise ValueError("compiled CE reduction or sum oracle differs")
        if (index < len(probe["rows"]) - 1
                and record["order"]["serial"] != record["order"]["grouped"]):
            raise ValueError("compiled search skipped earlier witness")
    witness = probe["rows"][-1]
    if witness["order"]["serial"] == witness["order"]["grouped"]:
        raise ValueError("compiled search has no distinguishing witness")
    return probe, witness


def validate_oracle(payload: dict[str, object]) -> None:
    """Independently recompute RN-even sums from frozen CE numerator words."""
    if not isinstance(payload, dict) or set(payload) != {
        "kind", "version", "search", "model", "corpus", "batches", "order",
        "batch_words_sha256", "compiled", "mathematical_order",
    }:
        raise ValueError("witness fields differ from version 2")
    if payload["kind"] != "genuine-m3-ce-sum-order" or payload["version"] != 2:
        raise ValueError("witness kind/version mismatch")
    batches = payload["batches"]
    if not isinstance(batches, list) or len(batches) != 3:
        raise ValueError("witness requires exactly three D2 batches")
    if payload["batch_words_sha256"] != batch_words_sha256(batches):
        raise ValueError("CE/numerator batch word checksum mismatch")
    try:
        numerators = tuple(int(batch["numerator_f32"], 16) for batch in batches)
        if any(len(batch["numerator_f32"]) != 8 for batch in batches):
            raise ValueError("noncanonical numerator word")
        expected = order_words(numerators)
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("invalid numerator words") from error
    if payload["mathematical_order"] != expected:
        raise ValueError("mathematical sum-order comparator disagrees with rational RN-even oracle")
    probe, witness = compiled_probe()
    compiled = payload["compiled"]
    if (payload["search"]["first_witness_index"] != witness["index"]
            or payload["corpus"]["source_hex"] != witness["raw"]
            or compiled != {"probe_sha256": PROBE_SHA256,
                            "executable_sha256": probe["executable_sha256"],
                            "openblas_sha256": probe["openblas_sha256"],
                            "ce_f32": witness["compiled_ce"],
                            "numerator_f32": witness["compiled_numerators"],
                            "stdout_sha256": witness["stdout_sha256"],
                            "roles_sha256": witness["roles_sha256"]}
            or payload["order"] != witness["order"]):
        raise ValueError("compiled witness differs from checked native probe")


def build() -> dict[str, object]:
    """Search finite domain and freeze only a genuinely distinguishing case."""
    import torch

    m3.setup()  # single CPU thread and deterministic algorithms
    parameters, successor = m3.parameters(SEED)
    probe, witness = compiled_probe()
    index, raw = witness["index"], bytes.fromhex(witness["raw"])
    with tempfile.TemporaryDirectory(prefix="e3-sum-order-") as location:
        directory = Path(location)
        manifest = write_d1_resource(
            directory, (tuple(raw),), fingerprint=FINGERPRINT, vocab=256,
        )
        loaded, digest = load_d1_resource(
            directory, expected_fingerprint=FINGERPRINT, expected_vocab=256,
            maximum_total_tokens=6,
        )
        if loaded != (tuple(raw),):
            raise AssertionError("physical D1 resource altered raw source")
        emitted = rows(raw, digest)
        reductions = ce_numerators(parameters, emitted)
        files = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.iterdir()
        }
    role_records = [
        _roles(role_trace(parameters, tuple(row["inputs"]))) for row in emitted
    ]
    batches = [
        {**row, "ce_f32": [f"{word:08x}" for word in ce],
         "numerator_f32": f"{numerator:08x}", "weight_f32": f"{weight:08x}"}
        for row, (ce, numerator, weight) in zip(emitted, reductions)
    ]
    parameter_words = b"".join(
        bytes.fromhex(f32_bits(item)) for path, _ in m3.PARAMETERS
        for item in parameters[path].to(torch.float32).detach().reshape(-1).tolist()
    )
    payload = {
        "kind": "genuine-m3-ce-sum-order", "version": 2,
        "search": {"domain": DOMAIN.decode("ascii"), "maximum_candidates": MAX_CANDIDATES,
                   "first_witness_index": index, "compiled_candidates_tested": len(probe["rows"])},
        "model": {"seed": SEED, "initializer_successor": list(successor),
                  "parameter_f32_sha256": hashlib.sha256(parameter_words).hexdigest(),
                  "reference_manifest_sha256": hashlib.sha256(
                      (Path(__file__).parents[1] / "m3/reference_manifest.json").read_bytes()
                  ).hexdigest(),
                  "role_f64_sha256": hashlib.sha256(canonical(role_records)).hexdigest(),
                  "role_count_per_batch": len(role_records[0]),
                  "q0_lock_sha256": hashlib.sha256(
                      (Path(__file__).parents[1] / "q0/requirements-oracle.lock").read_bytes()
                  ).hexdigest()},
        "corpus": {"source_hex": raw.hex(), "source_sha256": hashlib.sha256(raw).hexdigest(),
                   "tokenizer_sha256": hashlib.sha256(encode_artifact(IDENTITY)).hexdigest(),
                   "d1_manifest_trailer_sha256": manifest[-32:].hex(), "d1_files_sha256": files,
                   "d2_rows_sha256": hashlib.sha256(canonical(emitted)).hexdigest()},
        "batches": batches,
        "batch_words_sha256": batch_words_sha256(batches),
        "mathematical_order": order_words(tuple(item[1] for item in reductions)),
        "compiled": {"probe_sha256": PROBE_SHA256,
                     "executable_sha256": probe["executable_sha256"],
                     "openblas_sha256": probe["openblas_sha256"],
                     "ce_f32": witness["compiled_ce"],
                     "numerator_f32": witness["compiled_numerators"],
                     "stdout_sha256": witness["stdout_sha256"],
                     "roles_sha256": witness["roles_sha256"]},
        "order": witness["order"],
    }
    validate_oracle(payload)
    return payload
