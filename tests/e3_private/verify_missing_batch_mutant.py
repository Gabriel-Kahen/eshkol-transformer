"""Check a third-row compiled E3 accumulate omission against the frozen oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.e3_metrics.reference_probe import accumulate, finalize
from tests.e3_private.verify_sum_order_mutant import check_normal, frozen
from tests.e3_reference.transcript import encode


def check_mutant(normal: dict[str, object], mutant: dict[str, object],
                 normal_roles: bytes, mutant_roles: bytes) -> None:
    if normal_roles != mutant_roles:
        raise ValueError("missing-batch mutant changed model roles")
    if set(normal) != set(mutant) or len(mutant.get("batches", [])) != 3:
        raise ValueError("missing-batch mutant omitted an emitted D2 row")
    if mutant["batches"][:2] != normal["batches"][:2]:
        raise ValueError("missing-batch mutant diverged before the third row")
    before, after = normal["batches"][2], mutant["batches"][2]
    for key in ("ordinal", "inputs", "targets", "mask", "logits_f32",
                "ce_f32", "reduction_f32", "correct_f32"):
        if after[key] != before[key]:
            raise ValueError(f"missing-batch mutant changed third-row {key}")
    if [batch["mask"] for batch in mutant["batches"]] != [
            [True, True], [True, True], [True, False]]:
        raise ValueError("missing-batch mutant changed genuine D2 masks")

    previous = normal["batches"][1]
    prior = tuple(int(word, 16) for word in previous["state_f32"]) + tuple(previous["counters"])
    reduction = before["reduction_f32"]
    expected = accumulate(prior, (int(reduction[0], 16), int(reduction[1], 16),
                                  int(before["correct_f32"], 16), sum(before["mask"])))
    if normal["batches"][2]["counters"] != list(expected[3:]) or [
            int(word, 16) for word in before["state_f32"]] != list(expected[:3]):
        raise ValueError("frozen normal third accumulation differs from serial oracle")
    if after["counters"] != list(prior[3:]) or after["counters"] == before["counters"]:
        raise ValueError("batches[2].counters is not the missing-third-batch witness")
    if after["state_f32"] != previous["state_f32"]:
        raise ValueError("batches[2].state_f32 is not the omitted serial N/W/C")
    if (expected[0] == prior[0] or expected[1] == prior[1]
            or expected[2] != prior[2]):
        raise ValueError("genuine third row does not witness N/W and exact C")

    final = mutant["final"]
    if final["counters"] != list(prior[3:]):
        raise ValueError("missing-batch final counters differ from prior state")
    expected_final = finalize(prior)
    actual_final = [int(word, 16) for word in final["metrics_f32"]]
    if (actual_final[:2] != list(expected_final[:2])
            or actual_final[3] != expected_final[3]
            or abs(actual_final[2] - expected_final[2]) > 2):
        raise ValueError("missing-batch final metrics differ from omitted-state oracle")
    raise ValueError("batches[2].counters: missing third accumulated batch")


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
