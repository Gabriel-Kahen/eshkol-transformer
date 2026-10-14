"""Host-only controls for the private packed-final-row capacity mutant."""

from __future__ import annotations

import copy
import subprocess
import unittest
from pathlib import Path

from tests.e3_metrics.reference_probe import accumulate, divide, finalize, reduce_bool, rounded
from tests.e3_private.verify_capacity_denominator_mutant import check_mutant
from tests.e3_private.verify_sum_order_mutant import frozen
from tests.e3_reference.reference import _synthetic_case
from tests.e3_reference.transcript import TranscriptError
from tests.m3 import reference as m3

ROOT = Path(__file__).resolve().parents[2]


class CapacityDenominatorSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture, rows, _, _ = frozen()
        m3.setup()
        parameters, _ = m3.parameters(1729)
        cls.normal, _, _ = _synthetic_case(rows, parameters)
        cls.normal["kind"] = "observed-f32"
        state = (0, 0, 0, 0, 0)
        for batch, ce in zip(cls.normal["batches"], fixture["compiled"]["ce_f32"]):
            batch["ce_f32"] = ce
            n, w = reduce_bool(tuple(int(word, 16) for word in ce), batch["mask"])
            batch["reduction_f32"] = [f"{n:08x}", f"{w:08x}", f"{divide(n, w):08x}"]
            state = accumulate(state, (n, w, int(batch["correct_f32"], 16),
                                       sum(batch["mask"])))
            batch["state_f32"] = [f"{word:08x}" for word in state[:3]]
            batch["counters"] = list(state[3:])
        cls.normal["final"] = {
            "metrics_f32": [f"{word:08x}" for word in finalize(state)],
            "counters": list(state[3:]),
        }

    def mutant(self) -> dict[str, object]:
        result = copy.deepcopy(self.normal)
        final = result["batches"][2]
        n, w = int(final["reduction_f32"][0], 16), rounded(2)
        final["reduction_f32"] = [f"{n:08x}", f"{w:08x}", f"{divide(n, w):08x}"]
        previous = result["batches"][1]
        state = accumulate(tuple(int(word, 16) for word in previous["state_f32"])
                           + tuple(previous["counters"]),
                           (n, w, int(final["correct_f32"], 16), 1))
        final["state_f32"] = [f"{word:08x}" for word in state[:3]]
        final["counters"] = list(state[3:])
        result["final"] = {
            "metrics_f32": [f"{word:08x}" for word in finalize(state)],
            "counters": list(state[3:]),
        }
        return result

    def test_first_difference_and_downstream_oracle(self) -> None:
        mutant = self.mutant()
        self.assertEqual([batch["mask"] for batch in mutant["batches"]],
                         [[True, True], [True, True], [True, False]])
        self.assertEqual(mutant["batches"][2]["reduction_f32"][:1],
                         self.normal["batches"][2]["reduction_f32"][:1])
        self.assertEqual(mutant["batches"][2]["reduction_f32"][1], "40000000")
        with self.assertRaisesRegex(TranscriptError,
                                    r"batches\[2\]\.reduction_f32: exact rational reduction mismatch"):
            check_mutant(self.normal, mutant, b"same roles", b"same roles")

    def test_preflight_and_false_kills(self) -> None:
        mutant = self.mutant()
        for field, value, message in (
            ("mask", [True, True], "final mask"),
            ("targets", [0, 0], "final targets"),
            ("ce_f32", ["00000000", "00000000"], "final ce_f32"),
            ("correct_f32", "40000000", "final correct_f32"),
        ):
            changed = copy.deepcopy(mutant)
            changed["batches"][2][field] = value
            with self.assertRaisesRegex(ValueError, message):
                check_mutant(self.normal, changed, b"same roles", b"same roles")
        changed = copy.deepcopy(mutant)
        changed["batches"][2]["reduction_f32"][0] = "00000000"
        with self.assertRaisesRegex(ValueError, "first changed reduction"):
            check_mutant(self.normal, changed, b"same roles", b"same roles")
        changed = copy.deepcopy(mutant)
        changed["final"]["metrics_f32"][1] = "40a00000"
        with self.assertRaisesRegex(ValueError, "final publication"):
            check_mutant(self.normal, changed, b"same roles", b"same roles")
        with self.assertRaisesRegex(ValueError, "changed model roles"):
            check_mutant(self.normal, mutant, b"before", b"after")

    def test_selector_is_private_and_closed(self) -> None:
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        selector = builder.split("e3_numerical_mutant=${E3_NUMERICAL_MUTANT:-normal}", 1)[1]
        selector = "e3_numerical_mutant=${E3_NUMERICAL_MUTANT:-normal}" + selector.split(
            "require_command awk", 1)[0]
        shell = ("set -euo pipefail\npackage_policy=$1\ne3_tuple_kind=$2\n"
                 "die() { printf '%s\\n' \"$*\" >&2; exit 77; }\n"
                 + selector + "printf '%s\\n' \"$e3_numerical_mutant\"\n")
        def select(policy: str, kind: str, value: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(["bash", "-c", shell, "bash", policy, kind],
                                  env={"E3_NUMERICAL_MUTANT": value},
                                  capture_output=True, text=True, check=False)
        self.assertEqual(select("e3-private-aggregate", "numerical",
                                "capacity-denominator").stdout, "capacity-denominator\n")
        for policy, kind in (("installed", "numerical"),
                             ("e3-private-aggregate", "retention")):
            self.assertEqual(select(policy, kind, "capacity-denominator").returncode, 77)
        self.assertEqual(select("e3-private-aggregate", "numerical",
                                "capacity-denominator-extra").returncode, 77)
        l3s = (ROOT / "native/l3s_masked_objective_provider.c").read_text()
        metrics = (ROOT / "native/e3_evaluation_metrics_provider.c").read_text()
        self.assertIn("!defined(ET_E3_NUMERICAL_TESTING)", l3s)
        self.assertIn("!defined(ET_E3_NUMERICAL_TESTING)", metrics)
        self.assertIn("m[0] == 1.0f && m[1] == 0.0f", l3s)
        self.assertIn("tokens == 4 && batches == 2", metrics)


if __name__ == "__main__":
    unittest.main()
