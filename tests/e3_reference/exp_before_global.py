"""Frozen host witness for exponentiating per batch before E3's global loss.

This consumes only the independent D2/M3 development reference. It does not
observe or modify a compiled evaluator and cannot by itself kill a private mutant.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from tests.e3_metrics.reference_probe import (
    add, accumulate, divide, exp_bits, finalize, multiply,
)
from tests.e3_reference.corpus import IDENTITY, SOURCE_BYTES
from tests.e3_reference.reference import validate_bundle
from tests.e3_reference.transcript import canonical
from tests.t1.reference import encode_artifact


FIXTURE = Path(__file__).with_name("exp_before_global_v1.json")
CASE = "packed-single"


def derive(reference: dict[str, object]) -> dict[str, object]:
    validate_bundle(reference)
    matches = [case for case in reference["cases"] if case["name"] == CASE]
    if len(matches) != 1:
        raise ValueError("expected exactly one packed-single reference case")
    case = matches[0]
    corpus = case["corpus"]
    observed = case["synthetic"]
    if [row["mask"] for row in corpus["batches"]] != [
        [True, True], [True, True], [True, False]
    ]:
        raise ValueError("witness requires authentic unequal active counts")

    state = (0, 0, 0, 0, 0)
    weighted_exp = 0
    rows = []
    for batch in observed["batches"]:
        numerator, weight, loss = (int(word, 16) for word in batch["reduction_f32"])
        if divide(numerator, weight) != loss:
            raise ValueError("batch loss is inconsistent with independent f32 division")
        per_batch_exp = exp_bits(loss)
        weighted_exp = add(weighted_exp, multiply(per_batch_exp, weight))
        state = accumulate(state, (numerator, weight, 0, sum(batch["mask"])))
        rows.append({
            "numerator_f32": f"{numerator:08x}", "weight_f32": f"{weight:08x}",
            "loss_f32": f"{loss:08x}", "exp_loss_f32": f"{per_batch_exp:08x}",
        })
    global_exp = finalize(state)[2]
    mutant_exp = divide(weighted_exp, state[1])
    if len({row["loss_f32"] for row in rows}) < 2 or mutant_exp <= global_exp + 2:
        raise ValueError("case does not distinguish exp-before-global beyond tolerance")
    if f"{global_exp:08x}" != observed["final"]["metrics_f32"][2]:
        raise ValueError("global oracle differs from frozen synthetic finalization")
    return {
        "kind": "e3-exp-before-global-candidate-v1", "case": CASE,
        "model_parameter_f32_sha256": reference["model"]["parameter_f32_sha256"],
        "source_bytes": SOURCE_BYTES.hex(),
        "tokenizer_artifact_sha256": hashlib.sha256(encode_artifact(IDENTITY)).hexdigest(),
        "d1_files": corpus["corpus"]["files"],
        "d1_manifest_trailer_sha256": corpus["corpus"]["manifest_trailer_sha256"],
        "d2_start_cursor": corpus["start_cursor"],
        "provenance": dict(reference["provenance"],
                           witness_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        "active_masks": [row["mask"] for row in corpus["batches"]],
        "batches": rows,
        "global": {"numerator_f32": f"{state[0]:08x}",
                   "weight_f32": f"{state[1]:08x}",
                   "loss_f32": f"{divide(state[0], state[1]):08x}",
                   "perplexity_f32": f"{global_exp:08x}"},
        "mutant_weighted_batch_exp_f32": f"{mutant_exp:08x}",
        "difference_ulps": mutant_exp - global_exp,
    }


def validate_frozen(reference: dict[str, object], raw: bytes) -> None:
    if raw != canonical(derive(reference)):
        raise ValueError("frozen exp-before-global candidate differs from reference")
