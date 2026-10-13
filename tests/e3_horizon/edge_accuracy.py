"""Compare a private edge observation to frozen independent D2/M3 selection."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import struct


CASES = ("empty", "singleton", "entry-eos", "one-token-tail")
SELECTION = Path(__file__).with_name("edge_accuracy_v1.json")


def expected_accuracy_word(rows: object) -> int:
    proof = json.loads(SELECTION.read_text())
    if (set(proof) != {"kind", "model_seed", "parameter_f32_sha256",
                       "reference_sha256", "selection_sha256", "source_case", "rows"}
            or proof["kind"] != "e3-edge-accuracy-selection-v1"
            or proof["model_seed"] != 1729 or proof["source_case"] != "all-correct"
            or len(proof["rows"]) != 2):
        raise ValueError("edge selection proof schema differs")
    frozen = tuple(
        (tuple(row["inputs"]), tuple(row["targets"]), tuple(row["mask"]))
        for row in proof["rows"]
    )
    if rows != frozen:
        raise ValueError("edge D2 rows differ from independent M3 selection proof")
    correct = active = 0
    for row in proof["rows"]:
        if set(row) != {"inputs", "targets", "mask", "winners"}:
            raise ValueError("edge selection row schema differs")
        for winner, target, mask in zip(row["winners"], row["targets"], row["mask"]):
            if type(mask) is not bool or type(winner) is not int or type(target) is not int:
                raise ValueError("edge selection row type differs")
            active += int(mask)
            correct += int(mask and winner == target)
    if (active, correct) != (3, 3):
        raise ValueError("edge selection no longer proves three correct targets")
    return struct.unpack("<I", struct.pack("<f", correct / active))[0]


def verify_observation(oracle_text: str, observed_text: str) -> None:
    payload: dict[str, tuple[str, object]] = {}
    for line in oracle_text.splitlines():
        case, cursor, rows_text = line.split("\t")
        if case in payload or case not in CASES or len(cursor) % 2:
            raise ValueError("edge D2 oracle case or cursor differs")
        bytes.fromhex(cursor)
        payload[case] = cursor, ast.literal_eval(rows_text)
    if set(payload) != set(CASES):
        raise ValueError("edge D2 oracle case set differs")
    accuracy = expected_accuracy_word(payload["one-token-tail"][1])
    expected = [f"E3-EDGE-CURSOR {case} {payload[case][0]}" for case in CASES]
    expected += [f"E3-EDGE-ACCURACY one-token-tail {accuracy}",
                 "E3-EDGE-PRIVATE-PASS cases=4 zero-weight=3 one-token-tail=1"]
    if observed_text.splitlines() != expected:
        raise ValueError("edge compiled cursor or fourth I2 accuracy differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    args = parser.parse_args()
    verify_observation(args.oracle.read_text(), args.stdout.read_text())


if __name__ == "__main__":
    main()
