"""Check weighted per-batch exp against genuine global E3 perplexity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.e3_metrics.reference_probe import add, divide, exp_bits, multiply
from tests.e3_private.verify_sum_order_mutant import check_normal, frozen
from tests.e3_reference.transcript import TranscriptError, encode, validate_observed_case


def expected_perplexities(batches: list[dict[str, object]]) -> tuple[str, str]:
    numerator = weight = weighted_ppl = 0
    for batch in batches:
        batch_n, batch_w = (int(word, 16) for word in batch["reduction_f32"][:2])
        numerator = add(numerator, batch_n)
        weight = add(weight, batch_w)
        weighted_ppl = add(weighted_ppl,
                           multiply(exp_bits(divide(batch_n, batch_w)), batch_w))
    return (f"{exp_bits(divide(numerator, weight)):08x}",
            f"{divide(weighted_ppl, weight):08x}")


def check_mutant(normal: dict[str, object], mutant: dict[str, object],
                 normal_roles: bytes, mutant_roles: bytes) -> None:
    if normal_roles != mutant_roles or normal["batches"] != mutant["batches"]:
        raise ValueError("exp-before-global mutant changed model roles or batch state")
    if len(mutant["batches"]) != 3 or [b["mask"] for b in mutant["batches"]] != [
            [True, True], [True, True], [True, False]]:
        raise ValueError("exp-before-global corpus lacks the genuine three-row witness")
    ordinary, early = expected_perplexities(normal["batches"])
    if ordinary == early:
        raise ValueError("genuine batch losses do not distinguish exp order")
    if abs(int(normal["final"]["metrics_f32"][2], 16) - int(ordinary, 16)) > 2:
        raise ValueError("normal global perplexity differs from oracle")
    if (abs(int(mutant["final"]["metrics_f32"][2], 16) - int(early, 16)) > 4
            or mutant["final"]["metrics_f32"][2] == normal["final"]["metrics_f32"][2]
            or mutant["final"]["metrics_f32"][:2] != normal["final"]["metrics_f32"][:2]
            or mutant["final"]["metrics_f32"][3] != normal["final"]["metrics_f32"][3]
            or mutant["final"]["counters"] != normal["final"]["counters"]):
        raise ValueError("mutant publication is not weighted per-batch perplexity")
    rows = tuple({key: batch[key] for key in ("inputs", "targets", "mask")}
                 for batch in normal["batches"])
    try:
        validate_observed_case(mutant, rows)
    except TranscriptError as error:
        if "final.metrics_f32: exponential exceeds two binary32 ULP" not in str(error):
            raise
        raise
    raise ValueError("exp-before-global mutant survived the global assertion")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", required=True, type=Path)
    parser.add_argument("--roles", required=True, type=Path)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--baseline-observed", type=Path)
    parser.add_argument("--baseline-roles", type=Path)
    args = parser.parse_args()
    if args.transcript.exists():
        raise FileExistsError("exp-before-global transcript output must be fresh")
    reference = frozen()
    observed = json.loads(args.observed.read_bytes())
    if args.baseline_observed is None:
        if args.baseline_roles is not None:
            raise ValueError("baseline roles require baseline observation")
        check_normal(observed, args.roles, reference)
        args.transcript.write_bytes(encode(observed))
        print("E3-EXP-BEFORE-GLOBAL-NORMAL-PASS")
    else:
        if args.baseline_roles is None:
            raise ValueError("mutant comparison requires baseline roles")
        normal = json.loads(args.baseline_observed.read_bytes())
        check_normal(normal, args.baseline_roles, reference)
        check_mutant(normal, observed, args.baseline_roles.read_bytes(),
                     args.roles.read_bytes())


if __name__ == "__main__":
    main()
