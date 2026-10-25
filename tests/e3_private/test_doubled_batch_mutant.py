"""Source controls for the private all-correct doubled-batch counterexample."""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.e3_metrics.reference_probe import accumulate, finalize
from tests.e3_private.verify_doubled_batch_mutant import check_mutant
from tests.e3_reference.correctness import build_reference, materialize
from tests.e3_reference.transcript import TranscriptError


ROOT = Path(__file__).resolve().parents[2]


class DoubledBatchMutantSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="e3-m2-double-")
        manifest = materialize(Path(cls.temporary.name) / "corpus")
        cls.normal = build_reference(manifest)["cases"][0]["synthetic"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def mutant(self) -> dict[str, object]:
        mutant = copy.deepcopy(self.normal)
        third = mutant["batches"][2]
        normal_third = self.normal["batches"][2]
        state = tuple(int(word, 16) for word in normal_third["state_f32"]) + tuple(normal_third["counters"])
        batch = (int(third["reduction_f32"][0], 16), int(third["reduction_f32"][1], 16),
                 int(third["correct_f32"], 16), sum(third["mask"]))
        doubled = accumulate(state, batch)
        third["state_f32"] = [f"{word:08x}" for word in doubled[:3]]
        third["counters"] = list(doubled[3:])
        final = finalize(doubled)
        mutant["final"] = {"metrics_f32": [f"{word:08x}" for word in final],
                           "counters": list(doubled[3:])}
        return mutant

    def test_all_correct_counterexample_changes_serial_n_w_c_and_count(self) -> None:
        mutant = self.mutant()
        before, after = self.normal["batches"][2], mutant["batches"][2]
        self.assertEqual(before["mask"], [True, False])
        self.assertEqual(before["correct_f32"], "3f800000")
        self.assertTrue(all(left != right for left, right in zip(
            before["state_f32"], after["state_f32"])))
        self.assertEqual((before["counters"], after["counters"]), ([5, 3], [6, 4]))
        with self.assertRaisesRegex(TranscriptError,
                                    r"batches\[2\]\.state_f32: exact accumulation mismatch"):
            check_mutant(self.normal, mutant, b"same roles", b"same roles")

    def test_missing_row_or_wrong_replay_cannot_count_as_kill(self) -> None:
        mutant = self.mutant()
        with self.assertRaisesRegex(ValueError, "omitted a genuine D2 row"):
            check_mutant(self.normal, {**mutant, "batches": mutant["batches"][:2]},
                         b"same", b"same")
        changed = copy.deepcopy(mutant)
        changed["batches"][2]["mask"] = [True, True]
        with self.assertRaisesRegex(ValueError, "changed genuine third mask"):
            check_mutant(self.normal, changed, b"same", b"same")
        changed = copy.deepcopy(mutant)
        changed["batches"][2]["state_f32"][2] = "00000000"
        with self.assertRaisesRegex(ValueError, "not exactly doubled"):
            check_mutant(self.normal, changed, b"same", b"same")
        with self.assertRaisesRegex(ValueError, "changed genuine M3 roles"):
            check_mutant(self.normal, mutant, b"before", b"after")

    def test_private_tuple_and_gate_bindings(self) -> None:
        frame = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        selector = (ROOT / "scripts/build-e3-numerical-test.sh").read_text()
        gate = (ROOT / "scripts/test-e3-doubled-batch-mutant.sh").read_text()
        runtime = (ROOT / "tests/e3_private/doubled_batch_runtime.esk").read_text()
        self.assertIn("if (frame->counts[frame->sum_bank][1] == 3)", frame)
        self.assertIn("rc = e3_accumulate_dispatch(frame);", frame)
        self.assertIn("doubled-batch) package_native_cflags+=(-DET_E3_NUMERICAL_DOUBLED_BATCH)", builder)
        self.assertIn("doubled-batch-sanitize", selector)
        self.assertIn("--case all-correct", gate)
        self.assertIn("parse_correctness_observation", gate)
        self.assertIn("batches[2].state_f32: exact accumulation mismatch", gate)
        self.assertIn('[[ ! -e "${out}/transcript-${mutant}.json" ]]', gate)
        self.assertIn("(= (vector-ref result 6) 4)", runtime)
        cc = os.environ.get("E3_SOURCE_CC") or shutil.which("clang")
        if cc is None:
            self.skipTest("Clang unavailable for source admission guard")
        base = [cc, "-std=c11", "-fsyntax-only", "-Werror",
                "-I", str(ROOT / "include"), "-I", str(ROOT / "src"),
                "-I", str(ROOT / "native")]
        source = str(ROOT / "src/eshkol_transformer/e3_frame.c")
        denied = subprocess.run(base + ["-DET_E3_NUMERICAL_DOUBLED_BATCH", source],
                                capture_output=True, text=True, check=False)
        allowed = subprocess.run(base + ["-DET_E3_NUMERICAL_TESTING",
                                         "-DET_E3_NUMERICAL_DOUBLED_BATCH", source],
                                 capture_output=True, text=True, check=False)
        self.assertNotEqual(denied.returncode, 0)
        self.assertIn("requires the private numerical test tuple", denied.stderr)
        self.assertEqual(allowed.returncode, 0, allowed.stderr)


if __name__ == "__main__":
    unittest.main()
