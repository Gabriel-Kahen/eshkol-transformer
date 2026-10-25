"""Source controls for the genuine final-row capacity denominator mutant."""

from __future__ import annotations

import copy
import shutil
import subprocess
import unittest
from pathlib import Path

from tests.e3_metrics.reference_probe import accumulate, divide, finalize, reduce_bool
from tests.e3_private.parse_numerical_observation import parse
from tests.e3_private.test_numerical_tuple import frame_bytes, frame_lines
from tests.e3_private.verify_capacity_denominator_mutant import check_mutant
from tests.e3_private.verify_sum_order_mutant import frozen
from tests.e3_reference.reference import _synthetic_case
from tests.e3_reference.transcript import TranscriptError
from tests.m3 import reference as m3

ROOT = Path(__file__).resolve().parents[2]


class CapacityDenominatorMutantTests(unittest.TestCase):
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
        ce = tuple(int(word, 16) for word in final["ce_f32"])
        n, _ = reduce_bool(ce, (True, False))
        w = 0x40000000
        final["reduction_f32"] = [f"{n:08x}", f"{w:08x}", f"{divide(n, w):08x}"]
        previous = result["batches"][1]
        state = accumulate(tuple(int(word, 16) for word in previous["state_f32"])
                           + tuple(previous["counters"]),
                           (n, w, int(final["correct_f32"], 16), 2))
        final["state_f32"] = [f"{word:08x}" for word in state[:3]]
        final["counters"] = list(state[3:])
        result["final"] = {"metrics_f32": [f"{word:08x}" for word in finalize(state)],
                           "counters": list(state[3:])}
        return result

    def test_first_difference_and_full_mutant_oracle(self) -> None:
        mutant = self.mutant()
        self.assertEqual(mutant["batches"][2]["reduction_f32"][0],
                         self.normal["batches"][2]["reduction_f32"][0])
        self.assertNotEqual(mutant["batches"][2]["reduction_f32"][1],
                            self.normal["batches"][2]["reduction_f32"][1])
        with self.assertRaisesRegex(TranscriptError,
                                    r"batches\[2\]\.reduction_f32: exact rational reduction mismatch"):
            check_mutant(self.normal, mutant, b"same roles", b"same roles")
        changed = copy.deepcopy(mutant)
        changed["batches"][2]["correct_f32"] = (
            "3f800000" if mutant["batches"][2]["correct_f32"] == "00000000"
            else "00000000")
        with self.assertRaisesRegex(ValueError, "changed final correct_f32"):
            check_mutant(self.normal, changed, b"same roles", b"same roles")

    def test_parser_admits_only_genuine_last_row_capacity(self) -> None:
        records = frame_lines(3)
        row = next(index for index, line in enumerate(records)
                   if line.startswith("E3ROW\t2\t"))
        records[row] = records[row].replace("\t1,1", "\t1,0")
        with self.assertRaisesRegex(ValueError, "active count differs"):
            parse(frame_bytes(records), 3)
        payload, _ = parse(frame_bytes(records), 3, capacity_mutant=True)
        self.assertEqual(payload["batches"][2]["mask"], [True, False])
        records[row] = records[row].replace("\t1,0", "\t0,1")
        with self.assertRaisesRegex(ValueError, "active count differs"):
            parse(frame_bytes(records), 3, capacity_mutant=True)

    def test_tuple_guards_and_gate(self) -> None:
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        gate = (ROOT / "scripts/test-e3-capacity-denominator-mutant.sh").read_text()
        self.assertIn("capacity-denominator) package_native_cflags+=(-DET_E3_NUMERICAL_CAPACITY_DENOMINATOR)", builder)
        self.assertIn("--capacity-mutant", gate)
        self.assertIn("sum-order-capacity", gate)
        self.assertIn("batches[2].reduction_f32: exact rational reduction mismatch", gate)
        self.assertIn('[[ ! -e "${out}/transcript-${mutant}.json" ]]', gate)
        compiler = shutil.which("clang")
        if compiler is None:
            self.skipTest("Clang unavailable")
        base = [compiler, "-std=c11", "-fsyntax-only", "-Werror",
                "-I", str(ROOT / "include"), "-I", str(ROOT / "src")]
        for path in ("native/l3s_masked_objective_provider.c",
                     "native/e3_evaluation_metrics_provider.c"):
            source = str(ROOT / path)
            denied = subprocess.run(base + ["-DET_E3_NUMERICAL_CAPACITY_DENOMINATOR", source],
                                    capture_output=True, text=True, check=False)
            allowed = subprocess.run(base + ["-DET_E3_NUMERICAL_TESTING",
                                            "-DET_E3_NUMERICAL_CAPACITY_DENOMINATOR", source],
                                     capture_output=True, text=True, check=False)
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("requires the private numerical test tuple", denied.stderr)
            self.assertEqual(allowed.returncode, 0, allowed.stderr)


if __name__ == "__main__":
    unittest.main()
