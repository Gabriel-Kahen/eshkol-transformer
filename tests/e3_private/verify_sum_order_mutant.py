"""Compare compiled E3 sum-order observations to the frozen genuine witness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tests.e3_metrics.reference_probe import add, divide, reduce_bool
from tests.e3_native_parity.verify_observation import _f32, _f64, _roles
from tests.e3_reference.reference import (
    ROLE_NAMES, _synthetic_case, role_trace, _roles as mathematical_roles,
)
from tests.e3_reference.sum_order_witness import FIXTURE, build
from tests.e3_reference.transcript import (
    canonical, encode, validate_mathematical_case, validate_observed_case,
)
from tests.m3 import reference as m3


def frozen() -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], str]:
    fixture = json.loads(FIXTURE.read_bytes())
    if canonical(fixture) != FIXTURE.read_bytes() or fixture != build():
        raise ValueError("sum-order source fixture is not the frozen reference")
    rows = [{"inputs": batch["inputs"], "targets": batch["targets"],
             "mask": [bool(bit) for bit in batch["mask"]]}
            for batch in fixture["batches"]]
    m3.setup()
    parameters, _ = m3.parameters(1729)
    synthetic, mathematical, loss_word = _synthetic_case(rows, parameters)
    if (hashlib.sha256(canonical([batch["roles"] for batch in mathematical])).hexdigest()
            != fixture["model"]["role_f64_sha256"]):
        raise ValueError("frozen M3 role digest differs")
    for actual, expected in zip(synthetic["batches"], fixture["batches"]):
        if (actual["ce_f32"] != expected["ce_f32"]
                or actual["reduction_f32"][:2]
                != [expected["numerator_f32"], expected["weight_f32"]]):
            raise ValueError("frozen M3/CE words differ")
    if fixture["order"]["serial"] == fixture["order"]["grouped"]:
        raise ValueError("frozen compiled CE words do not distinguish grouping")
    return fixture, rows, mathematical, loss_word


def check_normal(observed: dict[str, object], role_path: Path,
                 frozen_reference=None) -> None:
    fixture, rows, mathematical, loss_word = frozen_reference or frozen()
    if len(observed.get("batches", [])) != 3:
        raise ValueError("actual compiled batch count differs from witness")
    for actual, ce, numerator in zip(observed["batches"],
                                     fixture["compiled"]["ce_f32"],
                                     fixture["compiled"]["numerator_f32"]):
        if (actual["ce_f32"] != ce or actual["reduction_f32"][0] != numerator):
            raise ValueError("actual compiled CE words differ from frozen witness")
    if observed["batches"][2]["state_f32"][0] != fixture["order"]["serial"]:
        raise ValueError("actual compiled serial state differs from frozen witness")
    validate_mathematical_case(observed, mathematical, tuple(rows), loss_word)
    roles = _roles(role_path)
    if len(roles) != 3 * len(ROLE_NAMES):
        raise ValueError("compiled role observation is incomplete")
    for batch_index, reference_batch in enumerate(mathematical):
        for role_index, expected in enumerate(reference_batch["roles"]):
            ordinal, name, shape, words = roles[batch_index * len(ROLE_NAMES) + role_index]
            if (ordinal != batch_index or name != expected["name"]
                    or list(shape) != expected["shape"]):
                raise ValueError("compiled M3 role identity or shape differs")
            reference = [_f64(word) for word in expected["values_f64"]]
            actual = [_f32(word) for word in words]
            if (len(actual) != len(reference)
                    or max(abs(a - b) for a, b in zip(actual, reference))
                    > 1.0e-10 + 3.0e-4 * max(map(abs, reference))):
                raise ValueError(f"compiled M3 role {name} exceeds tolerance")
            if name == "Z" and list(words) != observed["batches"][batch_index]["logits_f32"]:
                raise ValueError("compiled Z role differs from observed logits")


def check_mutant(normal: dict[str, object], mutant: dict[str, object],
                 normal_roles: bytes, mutant_roles: bytes) -> None:
    if normal_roles != mutant_roles:
        raise ValueError("sum-order mutant changed model roles before accumulate")
    if (set(normal) != set(mutant) or normal["kind"] != "observed-f32"
            or mutant["kind"] != "observed-f32"):
        raise ValueError("sum-order mutant observation schema differs")
    if len(mutant["batches"]) != 3:
        raise ValueError("sum-order mutant omitted a genuine D2 batch")
    if mutant["batches"][:2] != normal["batches"][:2]:
        raise ValueError("sum-order mutant diverged before the third state")
    before, after = normal["batches"][2], mutant["batches"][2]
    if ({key: value for key, value in before.items() if key != "state_f32"}
            != {key: value for key, value in after.items() if key != "state_f32"}):
        raise ValueError("sum-order mutant changed a third-batch input or reduction")
    numerators = [int(batch["reduction_f32"][0], 16) for batch in normal["batches"]]
    serial = add(add(numerators[0], numerators[1]), numerators[2])
    grouped = add(numerators[0], add(numerators[1], numerators[2]))
    if serial == grouped:
        raise ValueError("compiled CE numerators do not distinguish grouping")
    if (before["state_f32"][0] != f"{serial:08x}"
            or after["state_f32"][0] != f"{grouped:08x}"
            or before["state_f32"][1:] != after["state_f32"][1:]
            or before["counters"] != after["counters"]):
        raise ValueError("first divergent actual state_f32 is not the third numerator")
    normal_final, mutant_final = normal["final"], mutant["final"]
    if (normal_final["counters"] != mutant_final["counters"]
            or normal_final["metrics_f32"][1] != mutant_final["metrics_f32"][1]
            or normal_final["metrics_f32"][3] != mutant_final["metrics_f32"][3]
            or mutant_final["metrics_f32"][0]
            != f"{divide(grouped, int(after['state_f32'][1], 16)):08x}"):
        raise ValueError("sum-order mutant final publication is not grouped-state derived")
    validate_observed_case(mutant, tuple({key: batch[key] for key in
                                         ("inputs", "targets", "mask")}
                                        for batch in normal["batches"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", required=True, type=Path)
    parser.add_argument("--roles", required=True, type=Path)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--baseline-observed", type=Path)
    parser.add_argument("--baseline-roles", type=Path)
    args = parser.parse_args()
    if args.transcript.exists():
        raise FileExistsError("sum-order transcript output must be fresh")
    reference = frozen()
    observed = json.loads(args.observed.read_bytes())
    if args.baseline_observed is None:
        if args.baseline_roles is not None:
            raise ValueError("baseline roles require baseline observation")
        check_normal(observed, args.roles, reference)
        args.transcript.write_bytes(encode(observed))
        print("E3-SUM-ORDER-NORMAL-PASS")
    else:
        if args.baseline_roles is None:
            raise ValueError("mutant comparison requires baseline roles")
        normal = json.loads(args.baseline_observed.read_bytes())
        check_normal(normal, args.baseline_roles, reference)
        check_mutant(normal, observed, args.baseline_roles.read_bytes(),
                     args.roles.read_bytes())


if __name__ == "__main__":
    main()
