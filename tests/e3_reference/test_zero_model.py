from __future__ import annotations

import copy
import math
import struct
import tempfile
import unittest
from pathlib import Path

from tests.e3_reference.transcript import TranscriptError
from tests.e3_reference.zero_model import CASES, build, load, validate, zero_parameters


class ZeroModelReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="e3-n3-zero-")
        root = Path(cls.temporary.name)
        cls.first = build(root / "first")
        cls.second = build(root / "second")
        cls.first_root = root / "first"
        cls.second_root = root / "second"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_two_independent_generations_are_byte_identical(self) -> None:
        first_paths = sorted(path.relative_to(self.first_root) for path in self.first_root.rglob("*") if path.is_file())
        second_paths = sorted(path.relative_to(self.second_root) for path in self.second_root.rglob("*") if path.is_file())
        self.assertEqual(first_paths, second_paths)
        for path in first_paths:
            with self.subTest(path=path):
                self.assertEqual((self.first_root / path).read_bytes(), (self.second_root / path).read_bytes())
        self.assertEqual(self.first, self.second)
        self.assertEqual(load(self.first_root), self.first)
        self.assertEqual(load(self.second_root), self.second)

    def test_distinct_owners_and_real_forward_zero(self) -> None:
        parameters, payload = zero_parameters()
        self.assertEqual(len(parameters), 14)
        self.assertEqual(len({id(value) for value in parameters.values()}), 14)
        self.assertEqual(len(payload), 4736)
        self.assertEqual(set(payload), {0})

    def test_genuine_d2_cases_ties_counts_and_padded_row(self) -> None:
        self.assertEqual(tuple(case["name"] for case in self.first["cases"]), tuple(name for name, _ in CASES))
        expected_accuracy = {"all-correct": "3f800000", "all-wrong": "00000000",
                             "mixed-padded": "3f19999a", "one-valid-padded": "3f800000"}
        for case in self.first["cases"]:
            with self.subTest(case=case["name"]):
                validate(self.first)
                batches = case["d2"]["batches"]
                self.assertEqual(case["d2"]["expected_batch_count"], len(batches))
                self.assertEqual(batches[-1]["mask"], [True, False])
                active = sum(sum(row["mask"]) for row in batches)
                final = case["oracle"]["predicted_f32"]["final"]
                self.assertEqual(final["counters"], [active, len(batches)])
                self.assertEqual(final["metrics_f32"][3], expected_accuracy[case["name"]])
                loss = struct.unpack(">f", bytes.fromhex(final["metrics_f32"][0]))[0]
                perplexity = struct.unpack(">f", bytes.fromhex(final["metrics_f32"][2]))[0]
                self.assertAlmostEqual(loss, math.log(256), delta=2e-6)
                self.assertAlmostEqual(perplexity, 256.0, delta=0.002)
                for row in case["oracle"]["mathematical"]:
                    self.assertEqual(len(row["roles"]), 21)
                    self.assertEqual(row["roles"][-1]["shape"], [1, 2, 256])
                for row in case["oracle"]["predicted_f32"]["batches"]:
                    self.assertEqual(len(row["logits_f32"]), 512)
                    self.assertEqual(set(row["logits_f32"]), {"00000000"})

    def test_negative_schema_rows_roles_metrics_and_cursor(self) -> None:
        mutations = []
        wrong = copy.deepcopy(self.first)
        wrong["model"]["parameter_count"] = 15
        mutations.append((wrong, ValueError))
        wrong = copy.deepcopy(self.first)
        wrong["cases"][0]["d2"]["batches"][0]["targets"][0] = 1
        mutations.append((wrong, ValueError))
        wrong = copy.deepcopy(self.first)
        wrong["cases"][0]["oracle"]["mathematical"][0]["roles"][0]["name"] = "forged"
        mutations.append((wrong, ValueError))
        wrong = copy.deepcopy(self.first)
        wrong["cases"][0]["oracle"]["mathematical"][0]["roles"][0]["shape"] = [1, 1, 8]
        mutations.append((wrong, ValueError))
        wrong = copy.deepcopy(self.first)
        wrong["cases"][0]["oracle"]["predicted_f32"]["batches"][0]["correct_f32"] = "00000000"
        mutations.append((wrong, TranscriptError))
        wrong = copy.deepcopy(self.first)
        wrong["cases"][0]["d2"]["end_cursor"] = wrong["cases"][0]["d2"]["start_cursor"]
        mutations.append((wrong, ValueError))
        for index, (payload, error) in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaises(error):
                validate(payload)

    def test_fresh_output_required(self) -> None:
        with self.assertRaises(FileExistsError):
            build(self.first_root)

    def test_physical_d1_and_record_corruption_reject(self) -> None:
        shard = self.first_root / "all-correct" / "shard-0000000000000000.ets"
        raw = shard.read_bytes()
        try:
            shard.write_bytes(raw[:-1] + bytes((raw[-1] ^ 1,)))
            with self.assertRaisesRegex(ValueError, "D1 files differ"):
                load(self.first_root)
        finally:
            shard.write_bytes(raw)
        record = self.first_root / "zero-model.json"
        raw = record.read_bytes()
        try:
            record.write_bytes(raw.replace(b'"version":1', b'"version":1,"version":1', 1))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load(self.first_root)
        finally:
            record.write_bytes(raw)


if __name__ == "__main__":
    unittest.main()
