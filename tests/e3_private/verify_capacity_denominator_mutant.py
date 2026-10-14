"""Kill a physical-capacity denominator on the genuine final packed 10 row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.e3_metrics.reference_probe import add, divide, exp_bits, rounded
from tests.e3_reference.transcript import encode, validate_observed_case


def check_mutant(normal: dict[str, object], mutant: dict[str, object],
                 normal_roles: bytes, mutant_roles: bytes) -> None:
    if normal_roles != mutant_roles:
        raise ValueError("capacity mutant changed model roles")
    if set(normal) != set(mutant) or len(mutant.get("batches", [])) != 3:
        raise ValueError("capacity mutant omitted a genuine D2 batch")
    if mutant["batches"][:2] != normal["batches"][:2]:
        raise ValueError("capacity mutant diverged before the final row")
    before, after = normal["batches"][2], mutant["batches"][2]
    if set(before) != set(after):
        raise ValueError("capacity mutant changed final-row schema")
    if before["mask"] != [True, False]:
        raise ValueError("capacity witness lacks the authentic 10 mask")
    for key in ("ordinal", "inputs", "targets", "mask", "logits_f32",
                "ce_f32", "correct_f32", "counters"):
        if before[key] != after[key]:
            raise ValueError(f"capacity mutant changed final {key}")
    numerator = int(before["reduction_f32"][0], 16)
    weight = rounded(2)
    if (before["reduction_f32"][1] != f"{rounded(1):08x}"
            or after["reduction_f32"] !=
            [f"{numerator:08x}", f"{weight:08x}", f"{divide(numerator, weight):08x}"]):
        raise ValueError("first changed reduction is not third-batch weight")
    previous = normal["batches"][1]
    state = [before["state_f32"][0],
             f"{add(int(previous['state_f32'][1], 16), weight):08x}",
             before["state_f32"][2]]
    if after["state_f32"] != state or after["counters"] != [5, 3]:
        raise ValueError("capacity mutant serial state or active count differs")
    n, w, c = (int(word, 16) for word in state)
    loss = divide(n, w)
    expected_final = {
        "metrics_f32": [f"{word:08x}" for word in
                        (loss, w, exp_bits(loss), divide(c, w))],
        "counters": [5, 3],
    }
    if mutant["final"] != expected_final or normal["final"] == expected_final:
        raise ValueError("capacity mutant final publication is not weight-derived")
    rows = tuple({key: batch[key] for key in ("inputs", "targets", "mask")}
                 for batch in normal["batches"])
    validate_observed_case(mutant, rows)
    raise ValueError("capacity mutant survived the genuine reduction assertion")


def main() -> None:
    from tests.e3_private.verify_sum_order_mutant import check_normal, frozen

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
