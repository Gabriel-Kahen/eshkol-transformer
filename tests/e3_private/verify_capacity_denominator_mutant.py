"""Verify the genuine final-row physical-capacity denominator mutation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.e3_metrics.reference_probe import accumulate, divide, finalize, rounded
from tests.e3_private.verify_sum_order_mutant import check_normal, frozen
from tests.e3_reference.transcript import TranscriptError, encode, validate_observed_case


def check_mutant(normal: dict[str, object], mutant: dict[str, object],
                 normal_roles: bytes, mutant_roles: bytes) -> None:
    if normal_roles != mutant_roles:
        raise ValueError("capacity mutant changed model roles")
    if set(normal) != set(mutant) or len(mutant.get("batches", [])) != 3:
        raise ValueError("capacity mutant omitted a genuine D2 batch")
    if mutant["batches"][:2] != normal["batches"][:2]:
        raise ValueError("capacity mutant diverged before the final row")
    before, after = normal["batches"][2], mutant["batches"][2]
    if before["mask"] != [True, False]:
        raise ValueError("capacity corpus lacks an authentic padded final row")
    for key in ("ordinal", "inputs", "targets", "mask", "logits_f32", "ce_f32",
                "correct_f32"):
        if after[key] != before[key]:
            raise ValueError(f"capacity mutant changed final {key}")
    numerator = before["reduction_f32"][0]
    weight = f"{rounded(2):08x}"  # physical width 2, fixed by D2 shape
    expected_reduction = [numerator, weight,
                          f"{divide(int(numerator, 16), int(weight, 16)):08x}"]
    if before["reduction_f32"][1] == weight:
        raise ValueError("capacity corpus does not distinguish the mutant")
    if after["reduction_f32"] != expected_reduction:
        raise ValueError("capacity L3S numerator, weight or mean differs from oracle")
    previous = normal["batches"][1]
    state = accumulate(tuple(int(word, 16) for word in previous["state_f32"])
                       + tuple(previous["counters"]),
                       (int(numerator, 16), int(weight, 16),
                        int(after["correct_f32"], 16), 2))
    if (after["state_f32"] != [f"{word:08x}" for word in state[:3]]
            or after["counters"] != list(state[3:])):
        raise ValueError("capacity accumulated state differs from oracle")
    if (mutant["final"]["metrics_f32"] !=
            [f"{word:08x}" for word in finalize(state)]
            or mutant["final"]["counters"] != list(state[3:])):
        raise ValueError("capacity final publication differs from oracle")
    rows = tuple({key: batch[key] for key in ("inputs", "targets", "mask")}
                 for batch in normal["batches"])
    try:
        validate_observed_case(mutant, rows)
    except TranscriptError as error:
        if "batches[2].reduction_f32: exact rational reduction mismatch" not in str(error):
            raise
        raise
    raise ValueError("capacity mutant survived the genuine weight assertion")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", required=True, type=Path)
    parser.add_argument("--roles", required=True, type=Path)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--baseline-observed", type=Path)
    parser.add_argument("--baseline-roles", type=Path)
    args = parser.parse_args()
    if args.transcript.exists():
        raise FileExistsError("capacity transcript output must be fresh")
    reference = frozen()
    observed = json.loads(args.observed.read_bytes())
    if args.baseline_observed is None:
        if args.baseline_roles is not None:
            raise ValueError("baseline roles require baseline observation")
        check_normal(observed, args.roles, reference)
        args.transcript.write_bytes(encode(observed))
        print("E3-CAPACITY-DENOMINATOR-NORMAL-PASS")
    else:
        if args.baseline_roles is None:
            raise ValueError("mutant comparison requires baseline roles")
        normal = json.loads(args.baseline_observed.read_bytes())
        check_normal(normal, args.baseline_roles, reference)
        check_mutant(normal, observed, args.baseline_roles.read_bytes(),
                     args.roles.read_bytes())


if __name__ == "__main__":
    main()
