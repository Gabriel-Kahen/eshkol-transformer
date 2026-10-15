"""Source controls for the compiled-private third-accumulation omission."""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.e3_metrics.reference_probe import accumulate, divide, exp_bits, finalize, reduce_bool
from tests.e3_private.parse_numerical_observation import parse
from tests.e3_private.test_numerical_tuple import frame_bytes, frame_lines
from tests.e3_private.verify_missing_batch_mutant import check_mutant
from tests.e3_private.verify_sum_order_mutant import frozen
from tests.e3_reference.reference import _synthetic_case
from tests.m3 import reference as m3

ROOT = Path(__file__).resolve().parents[2]


class MissingBatchMutantSourceTests(unittest.TestCase):
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
        loss = divide(state[0], state[1])
        cls.normal["final"] = {
            "metrics_f32": [f"{loss:08x}", f"{state[1]:08x}",
                            f"{exp_bits(loss):08x}", f"{divide(state[2], state[1]):08x}"],
            "counters": list(state[3:]),
        }

    def mutant(self) -> dict[str, object]:
        result = copy.deepcopy(self.normal)
        previous = result["batches"][1]
        result["batches"][2]["state_f32"] = previous["state_f32"].copy()
        result["batches"][2]["counters"] = previous["counters"].copy()
        prior = tuple(int(word, 16) for word in previous["state_f32"]) + tuple(previous["counters"])
        result["final"] = {
            "metrics_f32": [f"{word:08x}" for word in finalize(prior)],
            "counters": list(prior[3:]),
        }
        return result

    def test_causal_third_row_and_first_named_counter(self) -> None:
        mutant = self.mutant()
        self.assertEqual([row["mask"] for row in mutant["batches"]],
                         [[True, True], [True, True], [True, False]])
        self.assertEqual(mutant["batches"][2]["counters"], [4, 2])
        self.assertEqual(self.normal["batches"][2]["counters"], [5, 3])
        with self.assertRaisesRegex(ValueError,
                                    r"batches\[2\]\.counters: missing third accumulated batch"):
            check_mutant(self.normal, mutant, b"same 63 roles", b"same 63 roles")

    def test_alternative_differences_cannot_count_as_kill(self) -> None:
        mutant = self.mutant()
        for change, message in (
            (lambda item: item["batches"].pop(), "omitted an emitted D2 row"),
            (lambda item: item["batches"][2]["mask"].reverse(), "third-row mask"),
            (lambda item: item["batches"][2]["reduction_f32"].__setitem__(0, "00000000"),
             "third-row reduction_f32"),
            (lambda item: item["batches"][2]["counters"].__setitem__(0, 3),
             "not the missing-third-batch witness"),
            (lambda item: item["final"]["counters"].__setitem__(0, 3),
             "final counters"),
        ):
            changed = copy.deepcopy(mutant)
            change(changed)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                check_mutant(self.normal, changed, b"same", b"same")
        with self.assertRaisesRegex(ValueError, "changed model roles"):
            check_mutant(self.normal, mutant, b"normal", b"mutant")

    def test_parser_requires_three_complete_rows_and_sixty_three_roles(self) -> None:
        records = frame_lines(3)
        payload, roles = parse(frame_bytes(records), 3)
        self.assertEqual(len(payload["batches"]), 3)
        self.assertEqual(len(roles.splitlines()), 63)
        final_row = next(index for index, line in enumerate(records)
                         if line.startswith("E3ROW\t2\t"))
        with self.assertRaisesRegex(ValueError, "omitted or duplicated"):
            parse(frame_bytes(records[:final_row] + records[final_row + 1:]), 3)

    def test_private_macro_and_gate_source(self) -> None:
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        selector = (ROOT / "scripts/build-e3-numerical-test.sh").read_text()
        gate = (ROOT / "scripts/test-e3-missing-batch-mutant.sh").read_text()
        runtime = (ROOT / "tests/e3_private/numerical_runtime.esk").read_text()
        self.assertIn('missing-batch) package_native_cflags+=(-DET_E3_NUMERICAL_MISSING_BATCH)', builder)
        self.assertIn('"${e3_tuple_kind}" == numerical', builder)
        self.assertIn('missing-batch-sanitize', selector)
        self.assertIn('case_name=sum-order-missing-batch', gate)
        self.assertIn('batches[2].counters: missing third accumulated batch', gate)
        self.assertIn('[[ ! -e "${out}/transcript-${mutant}.json" ]]', gate)
        self.assertIn('(string=? name "sum-order-missing-batch") (vector #t 0 4 2)', runtime)
        self.assertIn('(bytes=? (vector-ref result 7) (vector-ref result 8))', runtime)
        cc = os.environ.get("E3_SOURCE_CC") or shutil.which("clang")
        if cc is None:
            self.skipTest("Clang unavailable for macro admission")
        base = [cc, "-std=c11", "-fsyntax-only", "-Werror", "-I", str(ROOT / "include"),
                "-I", str(ROOT / "src"), "-I", str(ROOT / "native")]
        source = str(ROOT / "native/e3_evaluation_metrics_provider.c")
        denied = subprocess.run(base + ["-DET_E3_NUMERICAL_MISSING_BATCH", source],
                                capture_output=True, text=True, check=False)
        allowed = subprocess.run(base + ["-DET_E3_NUMERICAL_TESTING",
                                         "-DET_E3_NUMERICAL_MISSING_BATCH", source],
                                 capture_output=True, text=True, check=False)
        canonical = subprocess.run(base + [source], capture_output=True, text=True, check=False)
        self.assertNotEqual(denied.returncode, 0)
        self.assertIn("requires the private numerical test tuple", denied.stderr)
        self.assertEqual(allowed.returncode, 0, allowed.stderr)
        self.assertEqual(canonical.returncode, 0, canonical.stderr)


if __name__ == "__main__":
    unittest.main()
