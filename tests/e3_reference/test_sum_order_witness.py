"""Development-only genuine E3 three-batch M2 sum-order witness."""

from __future__ import annotations

import copy
import json
import unittest

from tests.e3_reference.sum_order_witness import (
    FIXTURE, MAX_CANDIDATES, build, candidate, validate_oracle,
)
from tests.e3_reference.transcript import canonical


class SumOrderWitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = json.loads(FIXTURE.read_bytes())
        cls.generated = build()

    def test_first_genuine_witness_is_frozen_exactly(self) -> None:
        self.assertEqual(FIXTURE.read_bytes(), canonical(self.generated))
        self.assertEqual(self.frozen["search"]["first_witness_index"], 2)
        self.assertEqual(self.frozen["search"]["maximum_candidates"], MAX_CANDIDATES)
        self.assertEqual(candidate(2).hex(), self.frozen["corpus"]["source_hex"])
        self.assertEqual([batch["mask"] for batch in self.frozen["batches"]],
                         [[1, 1], [1, 1], [1, 0]])
        self.assertEqual(self.frozen["model"]["role_count_per_batch"], 21)
        self.assertEqual(self.frozen["order"], {
            "serial": "41dcde81", "grouped": "41dcde80", "wider": "41dcde81",
        })
        validate_oracle(self.frozen)

    def test_grouped_comparator_mutation_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.frozen)
        mutated["order"]["grouped"] = mutated["order"]["serial"]
        with self.assertRaisesRegex(ValueError, "rational RN-even oracle"):
            validate_oracle(mutated)

    def test_numerator_mutation_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.frozen)
        mutated["batches"][0]["numerator_f32"] = "00000000"
        with self.assertRaisesRegex(ValueError, "batch word checksum"):
            validate_oracle(mutated)


if __name__ == "__main__":
    unittest.main()
