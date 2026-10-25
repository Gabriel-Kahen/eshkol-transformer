"""Host-only controls for the authentic padded-hit M2 candidate."""

from pathlib import Path
import copy
import shutil
import subprocess
import tempfile
import unittest

from tests.e3_reference.padding_hit import materialize
from tests.e3_metrics.reference_probe import accumulate, divide, finalize, reduce_bool
from tests.e3_reference.transcript import TranscriptError, validate_observed_case


ROOT = Path(__file__).resolve().parents[2]


class PaddingHitSourceTest(unittest.TestCase):
    def test_genuine_d2_and_distinct_counting_boundary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-padding-hit-") as temporary:
            first = materialize(Path(temporary) / "first")
            second = materialize(Path(temporary) / "second")
            self.assertEqual(first, second)
            self.assertEqual(first["total_rows"], 3)
            self.assertEqual([row["mask"] for row in first["batches"]],
                             [[True, True], [True, True], [True, False]])
            self.assertEqual(first["batches"][2]["targets"], [1, 0])
            # Every zero-state logit ties, so the actual winner is ID zero.
            # Valid target 1 misses; padded target 0 hits if the mask is ignored.
            normal_hits = sum(target == 0 and active for row in first["batches"]
                              for target, active in zip(row["targets"], row["mask"]))
            mutant_hits = sum(target == 0 for row in first["batches"]
                              for target in row["targets"])
            self.assertEqual((normal_hits, mutant_hits), (4, 5))
            self.assertEqual(sum(sum(row["mask"]) for row in first["batches"]), 5)
            self.assertEqual(first["batches"][2]["mask"].count(True), 1)

    def test_private_macro_and_gate_contract(self) -> None:
        source = (ROOT / "native/e3_evaluation_metrics_provider.c").read_text()
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        gate = (ROOT / "scripts/test-e3-padding-hit-mutant.sh").read_text()
        self.assertIn("ET_E3_PADDING_HIT_TEST", source)
        self.assertIn("result->correct.active = (int64_t)mask[0] + (int64_t)mask[1]", source)
        self.assertIn('"${e3_uniform_mutant}" == padding-hit', builder)
        self.assertIn("package_native_cflags+=(-DET_E3_PADDING_HIT_TEST)", builder)
        self.assertIn("mutant-sanitized:sanitized", gate)
        self.assertIn("batches[2].correct_f32", gate)
        self.assertIn('[[ ! -e "${out}/transcript-${mutant}.json" ]]', gate)
        compiler = shutil.which("clang")
        if compiler is None:
            self.skipTest("Clang unavailable for private macro check")
        base = [compiler, "-std=c11", "-fsyntax-only", "-Werror",
                "-I", str(ROOT / "include"), "-I", str(ROOT / "src"),
                "-I", str(ROOT / "native"),
                str(ROOT / "native/e3_evaluation_metrics_provider.c")]
        denied = subprocess.run(base + ["-DET_E3_PADDING_HIT_TEST"],
                                capture_output=True, text=True)
        allowed = subprocess.run(base + ["-DET_E3_NUMERICAL_TESTING",
                                         "-DET_E3_PADDING_HIT_TEST"],
                                 capture_output=True, text=True)
        self.assertNotEqual(denied.returncode, 0)
        self.assertIn("requires the private numerical test tuple", denied.stderr)
        self.assertEqual(allowed.returncode, 0, allowed.stderr)

    def test_independent_oracle_first_divergence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-padding-oracle-") as temporary:
            rows = materialize(Path(temporary) / "fixture")["batches"]
        state = (0, 0, 0, 0, 0)
        batches = []
        for ordinal, row in enumerate(rows):
            mask = row["mask"]
            numerator, weight = reduce_bool((0x40B17218, 0x40B17218), mask)
            correct = (0x40000000, 0x40000000, 0)[ordinal]
            state = accumulate(state, (numerator, weight, correct, sum(mask)))
            batches.append({"ordinal": ordinal, **row,
                            "logits_f32": ["00000000"] * 512,
                            "ce_f32": ["40b17218"] * 2,
                            "reduction_f32": [f"{word:08x}" for word in
                                              (numerator, weight, divide(numerator, weight))],
                            "correct_f32": f"{correct:08x}",
                            "state_f32": [f"{word:08x}" for word in state[:3]],
                            "counters": list(state[3:])})
        normal = {"kind": "observed-f32", "profile": "e3-private-n1-t2-v256-d4-v1",
                  "batches": batches,
                  "final": {"metrics_f32": [f"{word:08x}" for word in finalize(state)],
                            "counters": list(state[3:])}}
        validate_observed_case(normal, tuple(rows))
        mutant = copy.deepcopy(normal)
        mutant["batches"][2]["correct_f32"] = "3f800000"
        mutant["batches"][2]["state_f32"][2] = "40a00000"
        mutant["final"]["metrics_f32"][3] = "3f800000"
        with self.assertRaisesRegex(TranscriptError,
                                    r"batches\[2\]\.correct_f32: target/argmax wiring mismatch"):
            validate_observed_case(mutant, tuple(rows))


if __name__ == "__main__":
    unittest.main()
