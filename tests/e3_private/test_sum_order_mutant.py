"""Source-only M2 mutant seam, genuine corpus and first-difference controls."""

from __future__ import annotations

import copy
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.e3_metrics.reference_probe import add, divide, exp_bits
from tests.e3_private.sum_order_corpus import materialize
from tests.e3_private.verify_sum_order_mutant import check_mutant, check_normal, frozen
from tests.e3_reference.reference import _synthetic_case
from tests.e3_reference.transcript import TranscriptError, f32_bits
from tests.m3 import reference as m3

ROOT = Path(__file__).resolve().parents[2]


class SumOrderMutantSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = frozen()
        cls.fixture, cls.rows, mathematical, _ = cls.reference
        m3.setup()
        parameters, _ = m3.parameters(1729)
        cls.normal, _, _ = _synthetic_case(cls.rows, parameters)
        cls.normal["kind"] = "observed-f32"
        numerator = 0
        for batch, ce, word in zip(cls.normal["batches"],
                                   cls.fixture["compiled"]["ce_f32"],
                                   cls.fixture["compiled"]["numerator_f32"]):
            batch["ce_f32"] = ce
            batch["reduction_f32"][0] = word
            batch["reduction_f32"][2] = f"{divide(int(word, 16), int(batch['reduction_f32'][1], 16)):08x}"
            numerator = add(numerator, int(word, 16))
            batch["state_f32"][0] = f"{numerator:08x}"
        weight = int(cls.normal["batches"][-1]["state_f32"][1], 16)
        loss = divide(numerator, weight)
        cls.normal["final"]["metrics_f32"][0] = f"{loss:08x}"
        cls.normal["final"]["metrics_f32"][2] = f"{exp_bits(loss):08x}"

    def test_normal_comparator_checks_all_named_m3_roles(self) -> None:
        lines = []
        for ordinal, batch in enumerate(self.reference[2]):
            for role in batch["roles"]:
                words = [f32_bits(struct.unpack(">d", bytes.fromhex(word))[0])
                         for word in role["values_f64"]]
                lines.append("\t".join((str(ordinal), role["name"],
                                         ",".join(map(str, role["shape"])),
                                         ",".join(words))))
        with tempfile.TemporaryDirectory(prefix="e3-m2-roles-") as location:
            path = Path(location) / "roles.tsv"
            path.write_text("\n".join(lines) + "\n")
            check_normal(self.normal, path, self.reference)
            lines[0] = lines[0].replace("\tET\t", "\tBAD\t", 1)
            path.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "role identity or shape"):
                check_normal(self.normal, path, self.reference)

    def test_frozen_physical_d1_and_packed_d2_materialize(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-m2-corpus-") as location:
            directory = Path(location) / "corpus"
            self.assertEqual(materialize(directory), self.fixture)
            self.assertEqual(sorted(path.name for path in directory.iterdir()),
                             ["manifest.etm", "shard-0000000000000000.ets"])
            with self.assertRaises(FileExistsError):
                materialize(directory)

    def test_first_mutant_difference_is_third_actual_state_word(self) -> None:
        normal = self.normal
        numerators = [int(batch["reduction_f32"][0], 16)
                      for batch in normal["batches"]]
        grouped = add(numerators[0], add(numerators[1], numerators[2]))
        mutant = copy.deepcopy(normal)
        mutant["batches"][2]["state_f32"][0] = f"{grouped:08x}"
        weight = int(mutant["batches"][2]["state_f32"][1], 16)
        mutant["final"]["metrics_f32"][0] = f"{divide(grouped, weight):08x}"
        with self.assertRaisesRegex(TranscriptError, r"batches\[2\]\.state_f32"):
            check_mutant(normal, mutant, b"unchanged roles", b"unchanged roles")
        mutant["batches"][2]["state_f32"][0] = "00000000"
        with self.assertRaisesRegex(ValueError, "first divergent actual state_f32"):
            check_mutant(normal, mutant, b"unchanged roles", b"unchanged roles")
        with self.assertRaisesRegex(ValueError, "changed model roles"):
            check_mutant(normal, mutant, b"roles a", b"roles b")

    def test_unwitnessed_compiled_ce_is_rejected(self) -> None:
        observed = copy.deepcopy(self.normal)
        observed["batches"][0]["ce_f32"][0] = "40b1902d"
        with tempfile.TemporaryDirectory(prefix="e3-m2-ce-") as location:
            role_path = Path(location) / "roles.tsv"
            role_path.write_text("")
            with self.assertRaisesRegex(ValueError, "compiled CE words differ"):
                check_normal(observed, role_path, self.reference)

    def test_mutation_is_confined_to_the_exact_noninstalled_tuple(self) -> None:
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        frame = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        gate = (ROOT / "scripts/test-e3-sum-order-mutant.sh").read_text()
        self.assertIn('"${e3_tuple_kind}" == numerical', builder)
        self.assertIn('sum-order) package_native_cflags+=(-DET_E3_NUMERICAL_SUM_ORDER)',
                      builder)
        self.assertIn("sum-order-sanitize", (ROOT / "scripts/build-e3-numerical-test.sh").read_text())
        self.assertIn("inputs[0].data = &frame->sum_order_first", frame)
        self.assertIn("inputs[5].data = &grouped_tail", frame)
        self.assertIn("transcript-${mutant}.json", gate)
        self.assertIn("batches[2].state_f32: exact accumulation mismatch", gate)
        if shutil.which("clang") is None:
            self.skipTest("Clang unavailable for source admission guard")
        base = ["clang", "-std=c11", "-fsyntax-only", "-I", str(ROOT / "include"),
                "-I", str(ROOT / "src"), "-I", str(ROOT / "native")]
        source = str(ROOT / "src/eshkol_transformer/e3_frame.c")
        denied = subprocess.run(base + ["-DET_E3_NUMERICAL_SUM_ORDER", source],
                                capture_output=True, text=True, check=False)
        allowed = subprocess.run(base + ["-DET_E3_NUMERICAL_TESTING",
                                         "-DET_E3_NUMERICAL_SUM_ORDER", source],
                                 capture_output=True, text=True, check=False)
        self.assertNotEqual(denied.returncode, 0)
        self.assertIn("requires the private numerical test tuple", denied.stderr)
        self.assertEqual(allowed.returncode, 0, allowed.stderr)


if __name__ == "__main__":
    unittest.main()
