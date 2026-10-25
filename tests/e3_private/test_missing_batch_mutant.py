"""Source and independent-oracle controls for the real third-row loss."""

from __future__ import annotations

import copy
import shutil
import subprocess
import unittest
from pathlib import Path

from tests.e3_metrics.reference_probe import finalize
from tests.e3_private import test_sum_order_mutant
from tests.e3_private.verify_missing_batch_mutant import check_mutant
from tests.e3_reference.transcript import TranscriptError


ROOT = Path(__file__).resolve().parents[2]


class MissingBatchMutantSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        test_sum_order_mutant.SumOrderMutantSourceTests.setUpClass()
        cls.normal = test_sum_order_mutant.SumOrderMutantSourceTests.normal

    def mutant(self) -> dict[str, object]:
        result = copy.deepcopy(self.normal)
        prior = result["batches"][1]
        last = result["batches"][2]
        last["state_f32"] = prior["state_f32"].copy()
        last["counters"] = prior["counters"].copy()
        state = tuple(int(word, 16) for word in prior["state_f32"])
        metrics = finalize(state + tuple(prior["counters"]))
        result["final"] = {"metrics_f32": [f"{word:08x}" for word in metrics],
                           "counters": prior["counters"].copy()}
        return result

    def test_real_third_row_reaches_exact_serial_state_kill(self) -> None:
        mutant = self.mutant()
        self.assertEqual([batch["mask"] for batch in mutant["batches"]],
                         [[True, True], [True, True], [True, False]])
        self.assertEqual(mutant["batches"][2]["counters"], [4, 2])
        with self.assertRaisesRegex(TranscriptError,
                                    r"batches\[2\]\.state_f32: exact accumulation mismatch"):
            check_mutant(self.normal, mutant, b"same roles", b"same roles")

    def test_wrong_mutations_cannot_count_as_kills(self) -> None:
        mutant = self.mutant()
        with self.assertRaisesRegex(ValueError, "omitted a genuine D2 row"):
            check_mutant(self.normal, {**mutant, "batches": mutant["batches"][:2]},
                         b"same roles", b"same roles")
        changed = copy.deepcopy(mutant)
        changed["batches"][2]["counters"] = [5, 3]
        with self.assertRaisesRegex(ValueError, "prior serial state"):
            check_mutant(self.normal, changed, b"same roles", b"same roles")
        changed = copy.deepcopy(mutant)
        changed["batches"][2]["mask"] = [True, True]
        with self.assertRaisesRegex(ValueError, "third-row input or reduction"):
            check_mutant(self.normal, changed, b"same roles", b"same roles")
        with self.assertRaisesRegex(ValueError, "changed M3 roles"):
            check_mutant(self.normal, mutant, b"normal", b"mutant")

    def test_test_only_source_and_gate(self) -> None:
        provider = (ROOT / "native/e3_evaluation_metrics_provider.c").read_text()
        frame = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        gate = (ROOT / "scripts/test-e3-missing-batch-mutant.sh").read_text()
        self.assertIn("if (batches == 2 && active == 1)", provider)
        self.assertIn("const int64_t ordinal = frame->missing_batch_emitted++;", frame)
        self.assertIn("missing-batch) package_native_cflags+=(-DET_E3_NUMERICAL_MISSING_BATCH)", builder)
        self.assertIn('case_name=sum-order-missing', gate)
        self.assertIn('batches[2].state_f32: exact accumulation mismatch', gate)
        self.assertIn('[[ ! -e "${out}/transcript-${mutant}.json" ]]', gate)
        compiler = shutil.which("clang")
        if compiler is None:
            self.skipTest("Clang unavailable for source guard")
        base = [compiler, "-std=c11", "-fsyntax-only", "-Werror",
                "-I", str(ROOT / "include"), "-I", str(ROOT / "src"),
                "-I", str(ROOT / "native")]
        for source in (ROOT / "native/e3_evaluation_metrics_provider.c",
                       ROOT / "src/eshkol_transformer/e3_frame.c"):
            denied = subprocess.run(base + ["-DET_E3_NUMERICAL_MISSING_BATCH",
                                            str(source)], capture_output=True, text=True)
            allowed = subprocess.run(base + ["-DET_E3_NUMERICAL_TESTING",
                                             "-DET_E3_NUMERICAL_MISSING_BATCH",
                                             str(source)], capture_output=True, text=True)
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("requires the private numerical test tuple", denied.stderr)
            self.assertEqual(allowed.returncode, 0, allowed.stderr)


if __name__ == "__main__":
    unittest.main()
