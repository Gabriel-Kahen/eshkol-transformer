"""Source-only controls for the genuine packed-final-row M2 mutation."""

from __future__ import annotations

import copy
import shutil
import struct
import subprocess
import unittest
from pathlib import Path

from tests.e3_metrics.reference_probe import accumulate, divide, exp_bits, reduce_bool, rounded
from tests.e3_private.verify_padded_position_mutant import check_mutant
from tests.e3_private.verify_sum_order_mutant import frozen
from tests.e3_private.parse_numerical_observation import parse
from tests.e3_private.test_numerical_tuple import frame_bytes, frame_lines
from tests.e3_reference.reference import _synthetic_case
from tests.e3_reference.transcript import TranscriptError
from tests.m3 import reference as m3

ROOT = Path(__file__).resolve().parents[2]


class PaddedPositionMutantSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture, rows, mathematical, _ = frozen()
        cls.mathematical = mathematical
        m3.setup()
        parameters, _ = m3.parameters(1729)
        cls.normal, _, _ = _synthetic_case(rows, parameters)
        cls.normal["kind"] = "observed-f32"
        state = (0, 0, 0, 0, 0)
        for batch, ce in zip(cls.normal["batches"], fixture["compiled"]["ce_f32"]):
            batch["ce_f32"] = ce
            n, w = reduce_bool(tuple(int(word, 16) for word in ce), batch["mask"])
            batch["reduction_f32"] = [f"{n:08x}", f"{w:08x}", f"{divide(n, w):08x}"]
            correct = int(batch["correct_f32"], 16)
            state = accumulate(state, (n, w, correct, sum(batch["mask"])))
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
        final = result["batches"][2]
        ce = tuple(int(word, 16) for word in final["ce_f32"])
        n, w = reduce_bool(ce, (True, True))
        final["reduction_f32"] = [f"{n:08x}", f"{w:08x}", f"{divide(n, w):08x}"]
        logits = [struct.unpack(">f", bytes.fromhex(word))[0]
                  for word in final["logits_f32"]]
        correct = sum(max(range(256), key=lambda token: (logits[i*256 + token], -token))
                      == final["targets"][i] for i in range(2))
        final["correct_f32"] = f"{rounded(correct):08x}"
        previous = result["batches"][1]
        state = accumulate(tuple(int(word, 16) for word in previous["state_f32"])
                           + tuple(previous["counters"]),
                           (n, w, rounded(correct), 2))
        final["state_f32"] = [f"{word:08x}" for word in state[:3]]
        final["counters"] = list(state[3:])
        loss = divide(state[0], state[1])
        result["final"] = {
            "metrics_f32": [f"{loss:08x}", f"{state[1]:08x}",
                            f"{exp_bits(loss):08x}", f"{divide(state[2], state[1]):08x}"],
            "counters": list(state[3:]),
        }
        return result

    def test_genuine_row_and_oracle_first_difference(self) -> None:
        mutant = self.mutant()
        self.assertEqual([batch["mask"] for batch in mutant["batches"]],
                         [[True, True], [True, True], [True, False]])
        self.assertNotEqual(mutant["batches"][2]["reduction_f32"][0],
                            self.normal["batches"][2]["reduction_f32"][0])
        self.assertEqual(mutant["batches"][2]["reduction_f32"][1], "40000000")
        with self.assertRaisesRegex(TranscriptError,
                                    r"batches\[2\]\.reduction_f32: exact rational reduction mismatch"):
            check_mutant(self.normal, mutant, b"same roles", b"same roles")

    def test_preflight_and_wrong_mutations_cannot_count_as_kills(self) -> None:
        mutant = self.mutant()
        with self.assertRaisesRegex(ValueError, "omitted a genuine D2 batch"):
            check_mutant(self.normal, {**mutant, "batches": mutant["batches"][:2]},
                         b"same roles", b"same roles")
        with self.assertRaisesRegex(ValueError, "changed the emitted D2 mask"):
            changed = copy.deepcopy(mutant)
            changed["batches"][2]["mask"] = [True, True]
            check_mutant(self.normal, changed, b"same roles", b"same roles")
        with self.assertRaisesRegex(ValueError, "L3S numerator or weight"):
            changed = copy.deepcopy(mutant)
            changed["batches"][2]["reduction_f32"][1] = "3f800000"
            check_mutant(self.normal, changed, b"same roles", b"same roles")
        with self.assertRaisesRegex(ValueError, "changed model roles"):
            check_mutant(self.normal, mutant, b"normal", b"mutant")

    def test_parser_admits_only_the_real_final_row_seam(self) -> None:
        records = frame_lines(3)
        final_row = next(index for index, line in enumerate(records)
                         if line.startswith("E3ROW\t2\t"))
        records[final_row] = records[final_row].replace("\t1,1", "\t1,0")
        with self.assertRaisesRegex(ValueError, "active count differs"):
            parse(frame_bytes(records), 3)
        payload, roles = parse(frame_bytes(records), 3, padded_mutant=True)
        self.assertEqual(payload["batches"][2]["mask"], [True, False])
        self.assertEqual(len(roles.splitlines()), 63)
        with self.assertRaisesRegex(ValueError, "omitted"):
            parse(frame_bytes(records[:final_row] + records[final_row + 25:]),
                  3, padded_mutant=True)
        records[final_row] = records[final_row].replace("\t1,0", "\t0,1")
        with self.assertRaisesRegex(ValueError, "active count differs"):
            parse(frame_bytes(records), 3, padded_mutant=True)

    def test_private_macro_and_source_gate(self) -> None:
        frame = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        gate = (ROOT / "scripts/test-e3-padded-position-mutant.sh").read_text()
        self.assertEqual(frame.count("mask.data = padded_mask;"), 2)
        self.assertIn("padded-position) package_native_cflags+=(-DET_E3_NUMERICAL_PADDED_POSITION)", builder)
        self.assertIn("--padded-mutant", gate)
        self.assertIn("batches[2].reduction_f32: exact rational reduction mismatch", gate)
        self.assertIn('[[ ! -e "${out}/transcript-${mutant}.json" ]]', gate)
        if shutil.which("clang") is None:
            self.skipTest("Clang unavailable for source admission guard")
        base = ["clang", "-std=c11", "-fsyntax-only", "-I", str(ROOT / "include"),
                "-I", str(ROOT / "src"), "-I", str(ROOT / "native")]
        source = str(ROOT / "src/eshkol_transformer/e3_frame.c")
        denied = subprocess.run(base + ["-DET_E3_NUMERICAL_PADDED_POSITION", source],
                                capture_output=True, text=True, check=False)
        allowed = subprocess.run(base + ["-DET_E3_NUMERICAL_TESTING",
                                         "-DET_E3_NUMERICAL_PADDED_POSITION", source],
                                 capture_output=True, text=True, check=False)
        self.assertNotEqual(denied.returncode, 0)
        self.assertIn("requires the private numerical test tuple", denied.stderr)
        self.assertEqual(allowed.returncode, 0, allowed.stderr)


if __name__ == "__main__":
    unittest.main()
