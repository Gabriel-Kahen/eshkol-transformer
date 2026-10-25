"""Source controls for weighted per-batch exp on the genuine M2 corpus."""

from __future__ import annotations

import copy
import shutil
import subprocess
import unittest
from pathlib import Path

from tests.e3_metrics.reference_probe import accumulate, divide, finalize, reduce_bool
from tests.e3_private.verify_exp_before_global_mutant import (
    check_mutant, expected_perplexities,
)
from tests.e3_private.verify_sum_order_mutant import frozen
from tests.e3_reference.reference import _synthetic_case
from tests.e3_reference.transcript import TranscriptError
from tests.m3 import reference as m3

ROOT = Path(__file__).resolve().parents[2]


class ExpBeforeGlobalMutantTests(unittest.TestCase):
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

    def test_real_compiled_ce_witness_and_first_difference(self) -> None:
        ordinary, early = expected_perplexities(self.normal["batches"])
        self.assertEqual((ordinary, early), ("43815d49", "438160e2"))
        mutant = copy.deepcopy(self.normal)
        mutant["final"]["metrics_f32"][2] = early
        self.assertEqual(mutant["batches"], self.normal["batches"])
        with self.assertRaisesRegex(TranscriptError,
                                    "final.metrics_f32: exponential exceeds two binary32 ULP"):
            check_mutant(self.normal, mutant, b"same roles", b"same roles")
        mutant["batches"][2]["mask"] = [True, True]
        with self.assertRaisesRegex(ValueError, "changed model roles or batch state"):
            check_mutant(self.normal, mutant, b"same roles", b"same roles")

    def test_mutant_is_private_and_compiler_guarded(self) -> None:
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        gate = (ROOT / "scripts/test-e3-exp-before-global-mutant.sh").read_text()
        frame = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        self.assertIn("exp-before-global) package_native_cflags+=(-DET_E3_NUMERICAL_EXP_BEFORE_GLOBAL)", builder)
        self.assertIn("frame->exp_before_sum + contribution", frame)
        self.assertIn("frame->exp_before_sum / frame->exp_before_weight", frame)
        self.assertIn('[[ ! -e "${out}/transcript-${mutant}.json" ]]', gate)
        self.assertIn("final.metrics_f32: exponential exceeds two binary32 ULP", gate)
        compiler = shutil.which("clang")
        if compiler is None:
            self.skipTest("Clang unavailable")
        base = [compiler, "-std=c11", "-fsyntax-only", "-Werror",
                "-I", str(ROOT / "include"), "-I", str(ROOT / "src"),
                "-I", str(ROOT / "native")]
        source = str(ROOT / "src/eshkol_transformer/e3_frame.c")
        denied = subprocess.run(base + ["-DET_E3_NUMERICAL_EXP_BEFORE_GLOBAL", source],
                                capture_output=True, text=True, check=False)
        allowed = subprocess.run(base + ["-DET_E3_NUMERICAL_TESTING",
                                        "-DET_E3_NUMERICAL_EXP_BEFORE_GLOBAL", source],
                                 capture_output=True, text=True, check=False)
        self.assertNotEqual(denied.returncode, 0)
        self.assertIn("requires the private numerical test tuple", denied.stderr)
        self.assertEqual(allowed.returncode, 0, allowed.stderr)


if __name__ == "__main__":
    unittest.main()
