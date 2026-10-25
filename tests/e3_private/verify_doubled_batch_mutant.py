"""Reject a replayed final genuine D2 batch after the private E3 traversal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.e3_metrics.reference_probe import accumulate, finalize
from tests.e3_reference.transcript import TranscriptError, validate_observed_case


def check_mutant(normal: dict[str, object], mutant: dict[str, object],
                 normal_roles: bytes, mutant_roles: bytes) -> None:
    if normal_roles != mutant_roles:
        raise ValueError("doubled-batch mutant changed genuine M3 roles")
    if set(normal) != set(mutant) or len(mutant.get("batches", [])) != 3:
        raise ValueError("doubled-batch mutant omitted a genuine D2 row")
    if mutant["batches"][:2] != normal["batches"][:2]:
        raise ValueError("doubled-batch mutant diverged before the third row")
    before, after = normal["batches"][2], mutant["batches"][2]
    for key in ("ordinal", "inputs", "targets", "mask", "logits_f32",
                "ce_f32", "reduction_f32", "correct_f32"):
        if after[key] != before[key]:
            raise ValueError(f"doubled-batch mutant changed genuine third {key}")
    if before["mask"] != [True, False] or before["correct_f32"] != "3f800000":
        raise ValueError("third D2 row lacks an active correct prediction")
    previous = normal["batches"][1]
    state = tuple(int(word, 16) for word in previous["state_f32"]) + tuple(previous["counters"])
    batch = (int(before["reduction_f32"][0], 16),
             int(before["reduction_f32"][1], 16),
             int(before["correct_f32"], 16), sum(before["mask"]))
    once = accumulate(state, batch)
    twice = accumulate(once, batch)
    if (before["state_f32"] != [f"{word:08x}" for word in once[:3]]
            or before["counters"] != list(once[3:])
            or after["state_f32"] != [f"{word:08x}" for word in twice[:3]]
            or after["counters"] != list(twice[3:])
            or before["counters"] != [5, 3]
            or after["counters"] != [6, 4]
            or before["state_f32"][2] == after["state_f32"][2]):
        raise ValueError("third serial N/W/C and batch counter are not exactly doubled")
    expected_final = finalize(twice)
    actual_final = mutant["final"]
    actual_metrics = tuple(int(word, 16) for word in actual_final["metrics_f32"])
    if (actual_final["counters"] != [6, 4]
            or actual_metrics[:2] != expected_final[:2]
            or actual_metrics[3] != expected_final[3]
            or abs(actual_metrics[2] - expected_final[2]) > 2):
        raise ValueError("doubled-batch final publication differs from serial state")
    rows = tuple({key: batch[key] for key in ("inputs", "targets", "mask")}
                 for batch in normal["batches"])
    try:
        validate_observed_case(mutant, rows)
    except TranscriptError as error:
        if "batches[2].state_f32: exact accumulation mismatch" not in str(error):
            raise
        raise
    raise ValueError("doubled-batch mutant survived the genuine serial assertion")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", required=True, type=Path)
    parser.add_argument("--roles", required=True, type=Path)
    parser.add_argument("--baseline-observed", required=True, type=Path)
    parser.add_argument("--baseline-roles", required=True, type=Path)
    parser.add_argument("--transcript", required=True, type=Path)
    args = parser.parse_args()
    if args.transcript.exists():
        raise FileExistsError("doubled-batch rejected transcript path must be fresh")
    check_mutant(json.loads(args.baseline_observed.read_bytes()),
                 json.loads(args.observed.read_bytes()),
                 args.baseline_roles.read_bytes(), args.roles.read_bytes())


if __name__ == "__main__":
    main()
