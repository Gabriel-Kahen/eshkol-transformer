"""Dependency-light causal checks for the physical-capacity verifier."""

from __future__ import annotations

import copy
import math
import struct
import unittest
from pathlib import Path

from tests.e3_metrics.reference_probe import accumulate, divide, finalize, reduce_bool, rounded
from tests.e3_private.verify_capacity_denominator_mutant import check_mutant
from tests.e3_reference.transcript import TranscriptError, validate_observed_case


def _word(value: float) -> str:
    return struct.pack(">f", value).hex()


def _ordinary() -> dict[str, object]:
    ce = _word(math.log(256))
    rows = []
    state = (0, 0, 0, 0, 0)
    for ordinal, mask in enumerate(([True, True], [True, True], [True, False])):
        n, w = reduce_bool((int(ce, 16), int(ce, 16)), mask)
        correct = rounded(sum(mask))
        state = accumulate(state, (n, w, correct, sum(mask)))
        rows.append({
            "ordinal": ordinal, "inputs": [ordinal, ordinal],
            "targets": [0, 0], "mask": mask,
            "logits_f32": ["00000000"] * 512, "ce_f32": [ce, ce],
            "reduction_f32": [f"{n:08x}", f"{w:08x}", f"{divide(n, w):08x}"],
            "correct_f32": f"{correct:08x}",
            "state_f32": [f"{word:08x}" for word in state[:3]],
            "counters": list(state[3:]),
        })
    return {
        "kind": "observed-f32", "profile": "e3-private-n1-t2-v256-d4-v1",
        "batches": rows,
        "final": {"metrics_f32": [f"{word:08x}" for word in finalize(state)],
                  "counters": list(state[3:])},
    }


def _mutant(normal: dict[str, object]) -> dict[str, object]:
    changed = copy.deepcopy(normal)
    row = changed["batches"][2]
    n, w = int(row["reduction_f32"][0], 16), rounded(2)
    row["reduction_f32"] = [f"{n:08x}", f"{w:08x}", f"{divide(n, w):08x}"]
    previous = changed["batches"][1]
    state = accumulate(tuple(int(word, 16) for word in previous["state_f32"])
                       + tuple(previous["counters"]),
                       (n, w, int(row["correct_f32"], 16), 1))
    row["state_f32"] = [f"{word:08x}" for word in state[:3]]
    row["counters"] = list(state[3:])
    changed["final"] = {
        "metrics_f32": [f"{word:08x}" for word in finalize(state)],
        "counters": list(state[3:]),
    }
    return changed


class CapacityDenominatorHostTests(unittest.TestCase):
    def test_gate_accepts_verifiers_normal_marker(self) -> None:
        root = Path(__file__).resolve().parents[2]
        gate = (root / "scripts/test-e3-capacity-denominator-mutant.sh").read_text()
        verifier = (root / "tests/e3_private/verify_capacity_denominator_mutant.py").read_text()
        marker = "E3-CAPACITY-DENOMINATOR-NORMAL-PASS"
        self.assertIn(f"print(\"{marker}\")", verifier)
        self.assertIn(f"grep -Fx '{marker}'", gate)

    def test_weight_is_first_change_and_oracle_kills(self) -> None:
        normal = _ordinary()
        rows = tuple({key: row[key] for key in ("inputs", "targets", "mask")}
                     for row in normal["batches"])
        validate_observed_case(normal, rows)
        mutant = _mutant(normal)
        self.assertEqual(normal["batches"][2]["reduction_f32"][0],
                         mutant["batches"][2]["reduction_f32"][0])
        with self.assertRaisesRegex(TranscriptError,
                                    r"batches\[2\]\.reduction_f32: exact rational reduction mismatch"):
            check_mutant(normal, mutant, b"roles", b"roles")

    def test_false_kills_are_rejected_before_oracle(self) -> None:
        normal = _ordinary()
        for field, value, reason in (
            ("mask", [True, True], "final mask"),
            ("correct_f32", "00000000", "final correct_f32"),
            ("ce_f32", ["00000000"] * 2, "final ce_f32"),
            ("reduction_f32", ["00000000", "40000000", "00000000"],
             "first changed reduction"),
        ):
            mutant = _mutant(normal)
            mutant["batches"][2][field] = value
            with self.assertRaisesRegex(ValueError, reason):
                check_mutant(normal, mutant, b"roles", b"roles")
        mutant = _mutant(normal)
        mutant["final"]["metrics_f32"][1] = "40a00000"
        with self.assertRaisesRegex(ValueError, "final publication"):
            check_mutant(normal, mutant, b"roles", b"roles")


if __name__ == "__main__":
    unittest.main()
