"""Check the genuine three-row private M2 padded-position mutation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from tests.e3_metrics.reference_probe import reduce_bool
from tests.e3_private.verify_sum_order_mutant import check_normal, frozen
from tests.e3_reference.transcript import TranscriptError, encode, validate_observed_case


def check_mutant(normal: dict[str, object], mutant: dict[str, object],
                 normal_roles: bytes, mutant_roles: bytes) -> None:
    if normal_roles != mutant_roles:
        raise ValueError("padded-position mutant changed model roles")
    if set(normal) != set(mutant) or len(mutant.get("batches", [])) != 3:
        raise ValueError("padded-position mutant omitted a genuine D2 batch")
    if mutant["batches"][:2] != normal["batches"][:2]:
        raise ValueError("padded-position mutant diverged before the final row")
    before, after = normal["batches"][2], mutant["batches"][2]
    if before["mask"] != [True, False] or after["mask"] != before["mask"]:
        raise ValueError("padded-position mutant changed the emitted D2 mask")
    for key in ("ordinal", "inputs", "targets", "mask", "logits_f32", "ce_f32"):
        if after[key] != before[key]:
            raise ValueError(f"padded-position mutant changed final {key}")
    ce = tuple(int(word, 16) for word in after["ce_f32"])
    expected_n, expected_w = reduce_bool(ce, (True, True))
    if (before["reduction_f32"][0] == f"{expected_n:08x}"
            or before["reduction_f32"][1] == f"{expected_w:08x}"):
        raise ValueError("padded-position corpus does not distinguish the mutant")
    if (after["reduction_f32"][0] != f"{expected_n:08x}"
            or after["reduction_f32"][1] != f"{expected_w:08x}"):
        raise ValueError("padded-position L3S numerator or weight differs from oracle")
    # Reinterpret only the test oracle's final mask. The compiled row above must
    # remain 10, while both numerical calls must behave as though it were 11.
    projected = copy.deepcopy(mutant)
    projected["batches"][2]["mask"] = [True, True]
    expected_rows = [dict((key, batch[key]) for key in ("inputs", "targets", "mask"))
                     for batch in projected["batches"]]
    validate_observed_case(projected, tuple(expected_rows))
    normal_rows = [dict((key, batch[key]) for key in ("inputs", "targets", "mask"))
                   for batch in normal["batches"]]
    try:
        validate_observed_case(mutant, tuple(normal_rows))
    except TranscriptError as error:
        if "batches[2].reduction_f32: exact rational reduction mismatch" not in str(error):
            raise
        raise
    raise ValueError("padded-position mutant survived the genuine mask assertion")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", required=True, type=Path)
    parser.add_argument("--roles", required=True, type=Path)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--baseline-observed", type=Path)
    parser.add_argument("--baseline-roles", type=Path)
    args = parser.parse_args()
    if args.transcript.exists():
        raise FileExistsError("padded-position transcript output must be fresh")
    reference = frozen()
    observed = json.loads(args.observed.read_bytes())
    if args.baseline_observed is None:
        if args.baseline_roles is not None:
            raise ValueError("baseline roles require baseline observation")
        check_normal(observed, args.roles, reference)
        args.transcript.write_bytes(encode(observed))
        print("E3-PADDED-POSITION-NORMAL-PASS")
    else:
        if args.baseline_roles is None:
            raise ValueError("mutant comparison requires baseline roles")
        normal = json.loads(args.baseline_observed.read_bytes())
        check_normal(normal, args.baseline_roles, reference)
        check_mutant(normal, observed, args.baseline_roles.read_bytes(),
                     args.roles.read_bytes())


if __name__ == "__main__":
    main()
