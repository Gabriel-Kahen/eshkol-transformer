"""Negative controls for the causal private finalization mutation check."""

from __future__ import annotations

import copy
import math
import unittest

from tests.e3_metrics.reference_probe import accumulate, divide, exp_bits, finalize, reduce_bool
from tests.e3_private.verify_second_normalization_mutant import check
from tests.e3_reference.transcript import f32_bits


def fixture() -> tuple[dict, dict, dict]:
    """Build synthetic comparator inputs; these are not compiled E3 evidence."""
    ce = f32_bits(math.log(256))
    rows = [
        {"inputs": [3, 197], "targets": [0, 0], "mask": [True, True]},
        {"inputs": [0, 255], "targets": [0, 0], "mask": [True, True]},
        {"inputs": [7, 7], "targets": [0, 0], "mask": [True, False]},
    ]
    state = (0, 0, 0, 0, 0)
    batches = []
    for ordinal, row in enumerate(rows):
        numerator, weight = reduce_bool((int(ce, 16), int(ce, 16)), row["mask"])
        correct = f32_bits(sum(row["mask"]))
        state = accumulate(state, (numerator, weight, int(correct, 16), sum(row["mask"])))
        batches.append({
            "ce_f32": [ce, ce], "correct_f32": correct,
            "counters": list(state[3:]), "inputs": row["inputs"],
            "logits_f32": ["00000000"] * 512, "mask": row["mask"],
            "ordinal": ordinal,
            "reduction_f32": [f"{numerator:08x}", f"{weight:08x}",
                              f"{divide(numerator, weight):08x}"],
            "state_f32": [f"{word:08x}" for word in state[:3]],
            "targets": row["targets"],
        })
    normal = {
        "kind": "observed-f32", "profile": "e3-private-n1-t2-v256-d4-v1",
        "batches": batches,
        "final": {"counters": list(state[3:]),
                  "metrics_f32": [f"{word:08x}" for word in finalize(state)]},
    }
    mutant = copy.deepcopy(normal)
    loss = divide(divide(state[0], state[1]), state[1])
    mutant["final"]["metrics_f32"][0] = f"{loss:08x}"
    mutant["final"]["metrics_f32"][2] = f"{exp_bits(loss):08x}"
    reference = {"cases": [{"name": "packed-single", "corpus": {"batches": rows}}]}
    return reference, normal, mutant


class SecondNormalizationMutantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference, self.normal, self.mutant = fixture()

    def check(self) -> None:
        check(self.reference, self.normal, self.mutant, b"roles", b"roles")

    def test_exact_final_mutation_is_distinguished(self) -> None:
        self.check()
        with self.assertRaisesRegex(ValueError, "role words"):
            check(self.reference, self.normal, self.mutant, b"roles", b"other")

    def test_earlier_state_and_publication_mutations_fail(self) -> None:
        for edit, message in (
            (lambda m: m["batches"][1]["ce_f32"].__setitem__(0, "00000000"),
             "before final publication"),
            (lambda m: m["final"]["metrics_f32"].__setitem__(0, "00000000"),
             "second exact division"),
            (lambda m: m["final"]["metrics_f32"].__setitem__(0,
             self.normal["final"]["metrics_f32"][0]), "second exact division"),
            (lambda m: m["final"]["metrics_f32"].__setitem__(1, "00000000"),
             "weight, accuracy or counters"),
            (lambda m: m["final"]["metrics_f32"].__setitem__(3, "00000000"),
             "weight, accuracy or counters"),
            (lambda m: m["final"]["metrics_f32"].__setitem__(2, "00000000"),
             "perplexity"),
            (lambda m: m["final"]["counters"].__setitem__(0, 4),
             "weight, accuracy or counters"),
            (lambda m: m["batches"].pop(), "before final publication"),
        ):
            with self.subTest(message=message):
                mutant = copy.deepcopy(self.mutant)
                edit(mutant)
                with self.assertRaisesRegex(ValueError, message):
                    check(self.reference, self.normal, mutant, b"roles", b"roles")

    def test_unvalidated_baseline_cannot_authorize_mutant(self) -> None:
        self.normal["batches"][0]["state_f32"][0] = "00000000"
        with self.assertRaisesRegex(ValueError, "state_f32"):
            self.check()

    def test_final_schema_and_two_ulp_bound(self) -> None:
        mutant = copy.deepcopy(self.mutant)
        exp = int(mutant["final"]["metrics_f32"][2], 16)
        mutant["final"]["metrics_f32"][2] = f"{exp + 2:08x}"
        check(self.reference, self.normal, mutant, b"roles", b"roles")
        mutant["final"]["metrics_f32"][2] = f"{exp + 3:08x}"
        with self.assertRaisesRegex(ValueError, "perplexity"):
            check(self.reference, self.normal, mutant, b"roles", b"roles")
        mutant["final"]["metrics_f32"][2] = "invalid"
        with self.assertRaisesRegex(ValueError, "malformed"):
            check(self.reference, self.normal, mutant, b"roles", b"roles")
        mutant = copy.deepcopy(self.mutant)
        mutant["final"]["extra"] = 1
        with self.assertRaisesRegex(ValueError, "schema"):
            check(self.reference, self.normal, mutant, b"roles", b"roles")
        mutant = copy.deepcopy(self.mutant)
        mutant["final"]["counters"] = [5.0, 3.0]
        with self.assertRaisesRegex(ValueError, "counters are malformed"):
            check(self.reference, self.normal, mutant, b"roles", b"roles")
        mutant = copy.deepcopy(self.mutant)
        mutant["batches"][0]["ordinal"] = 0.0
        with self.assertRaisesRegex(ValueError, "ordinal"):
            check(self.reference, self.normal, mutant, b"roles", b"roles")

    def test_gate_requires_causal_classification_after_strict_rejection(self) -> None:
        from pathlib import Path

        gate = (Path(__file__).resolve().parents[2] /
                "scripts/test-e3-numerical-transcript.sh").read_text()
        self.assertLess(gate.index('verify-second-normalization.stderr'),
                        gate.index('verify_second_normalization_mutant'))
        for name in ("--reference", "--normal", "--mutant", "--normal-roles",
                     "--mutant-roles"):
            self.assertIn(name, gate)


if __name__ == "__main__":
    unittest.main()
