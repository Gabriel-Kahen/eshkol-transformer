"""Check one lost K1 accumulation after a genuine third D2/M3 row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.e3_metrics.reference_probe import finalize
from tests.e3_private.verify_sum_order_mutant import check_normal, frozen
from tests.e3_reference.transcript import TranscriptError, encode, validate_observed_case


def check_mutant(normal: dict[str, object], mutant: dict[str, object],
                 normal_roles: bytes, mutant_roles: bytes) -> None:
    if normal_roles != mutant_roles:
        raise ValueError("missing-batch mutant changed M3 roles")
    if set(normal) != set(mutant) or len(mutant.get("batches", [])) != 3:
        raise ValueError("missing-batch mutant omitted a genuine D2 row")
    if mutant["batches"][:2] != normal["batches"][:2]:
        raise ValueError("missing-batch mutant diverged before third accumulation")
    before, after = normal["batches"][2], mutant["batches"][2]
    if ({key: value for key, value in before.items()
         if key not in ("state_f32", "counters")}
            != {key: value for key, value in after.items()
                if key not in ("state_f32", "counters")}):
        raise ValueError("missing-batch mutant changed a third-row input or reduction")
    prior = normal["batches"][1]
    if (after["state_f32"] != prior["state_f32"]
            or after["counters"] != prior["counters"]
            or before["counters"] != [5, 3]
            or after["counters"] != [4, 2]
            or before["state_f32"] == after["state_f32"]):
        raise ValueError("missing-batch state/counters are not the prior serial state")
    prior_state = tuple(int(word, 16) for word in prior["state_f32"])
    expected = finalize(prior_state + tuple(prior["counters"]))
    actual = [int(word, 16) for word in mutant["final"]["metrics_f32"]]
    if (mutant["final"]["counters"] != [4, 2]
            or actual[:2] != list(expected[:2]) or actual[3] != expected[3]
            or abs(actual[2] - expected[2]) > 2):
        raise ValueError("missing-batch final is not derived from the prior state")
    rows = tuple({key: batch[key] for key in ("inputs", "targets", "mask")}
                 for batch in normal["batches"])
    try:
        validate_observed_case(mutant, rows)
    except TranscriptError as error:
        if "batches[2].state_f32: exact accumulation mismatch" not in str(error):
            raise
        raise
    raise ValueError("missing-batch mutant survived the serial-state assertion")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", required=True, type=Path)
    parser.add_argument("--roles", required=True, type=Path)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--baseline-observed", type=Path)
    parser.add_argument("--baseline-roles", type=Path)
    args = parser.parse_args()
    if args.transcript.exists():
        raise FileExistsError("missing-batch transcript output must be fresh")
    reference = frozen()
    observed = json.loads(args.observed.read_bytes())
    if args.baseline_observed is None:
        if args.baseline_roles is not None:
            raise ValueError("baseline roles require baseline observation")
        check_normal(observed, args.roles, reference)
        args.transcript.write_bytes(encode(observed))
        print("E3-MISSING-BATCH-NORMAL-PASS")
    else:
        if args.baseline_roles is None:
            raise ValueError("mutant comparison requires baseline roles")
        normal = json.loads(args.baseline_observed.read_bytes())
        check_normal(normal, args.baseline_roles, reference)
        check_mutant(normal, observed, args.baseline_roles.read_bytes(),
                     args.roles.read_bytes())


if __name__ == "__main__":
    main()
