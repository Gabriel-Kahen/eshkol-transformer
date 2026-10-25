"""Verify the genuine zero-state padding-hit mutation without publishing it."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from tests.e3_native_parity.verify_observation import _roles
from tests.e3_reference.transcript import (
    TranscriptError, decode, encode, validate_mathematical_case,
)


def check_normal(observed: dict, roles: bytes, reference: dict) -> None:
    case = reference["cases"][0]
    if case["name"] != "padding-hit" or len(observed.get("batches", [])) != 3:
        raise ValueError("padding-hit case or compiled batch count differs")
    validate_mathematical_case(observed, case["mathematical"],
                               tuple(case["corpus"]["batches"]),
                               case["mathematical_loss_f64"])
    parsed_roles = _roles_bytes(roles)
    if len(parsed_roles) != 63:
        raise ValueError("padding-hit model roles are incomplete")
    for batch_index, batch in enumerate(case["mathematical"]):
        for role_index, expected in enumerate(batch["roles"]):
            ordinal, name, shape, words = parsed_roles[21 * batch_index + role_index]
            if (ordinal, name, list(shape)) != (batch_index, expected["name"], expected["shape"]):
                raise ValueError("padding-hit model role identity or shape differs")
            if any(word != "00000000" for word in words):
                raise ValueError("padding-hit zero-state role is not zero")


def _roles_bytes(raw: bytes):
    import tempfile
    with tempfile.TemporaryDirectory(prefix="e3-padding-hit-roles-") as directory:
        path = Path(directory) / "roles.tsv"
        path.write_bytes(raw)
        return _roles(path)


def check_mutant(normal: dict, mutant: dict, normal_roles: bytes,
                 mutant_roles: bytes, reference: dict) -> None:
    check_normal(normal, normal_roles, reference)
    if normal_roles != mutant_roles:
        raise ValueError("padding-hit mutation changed M3 roles")
    if normal["batches"][:2] != mutant.get("batches", [])[:2]:
        raise ValueError("padding-hit mutant diverged before the final D2 row")
    before, after = normal["batches"][2], mutant["batches"][2]
    if before["mask"] != [True, False] or before["targets"] != [1, 0]:
        raise ValueError("padding-hit corpus does not distinguish valid and padded targets")
    projected = copy.deepcopy(normal)
    projected["batches"][2]["correct_f32"] = "3f800000"
    projected["batches"][2]["state_f32"][2] = "40a00000"
    projected["final"]["metrics_f32"][3] = "3f800000"
    if mutant != projected:
        raise ValueError("padding-hit mutation changed more than final C and accuracy")
    if (before["correct_f32"], after["correct_f32"]) != ("00000000", "3f800000"):
        raise ValueError("first divergent word is not third-row correct_f32")
    try:
        check_normal(mutant, mutant_roles, reference)
    except TranscriptError as error:
        if "batches[2].correct_f32" not in str(error):
            raise
        raise
    else:
        raise ValueError("padding-hit mutant survived the independent oracle")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--observed", required=True, type=Path)
    parser.add_argument("--roles", required=True, type=Path)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--baseline-observed", type=Path)
    parser.add_argument("--baseline-roles", type=Path)
    args = parser.parse_args()
    if args.transcript.exists():
        raise FileExistsError("padding-hit transcript output must be fresh")
    reference = decode(args.reference.read_bytes())
    # Observed records are canonical JSON, not checksummed reference records.
    from tests.e3_reference.transcript import canonical
    import json
    observed = json.loads(args.observed.read_bytes())
    if canonical(observed) != args.observed.read_bytes():
        raise ValueError("padding-hit observation is not canonical")
    roles = args.roles.read_bytes()
    if args.baseline_observed is None:
        if args.baseline_roles is not None:
            raise ValueError("baseline roles require a baseline observation")
        check_normal(observed, roles, reference)
        args.transcript.write_bytes(encode(observed))
        print("E3-PADDING-HIT-NORMAL-PASS")
    else:
        if args.baseline_roles is None:
            raise ValueError("padding-hit mutant requires baseline roles")
        baseline = json.loads(args.baseline_observed.read_bytes())
        if canonical(baseline) != args.baseline_observed.read_bytes():
            raise ValueError("padding-hit baseline is not canonical")
        check_mutant(baseline, observed, args.baseline_roles.read_bytes(),
                     roles, reference)


if __name__ == "__main__":
    main()
