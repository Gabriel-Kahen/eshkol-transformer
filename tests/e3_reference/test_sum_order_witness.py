"""Development-only genuine E3 three-batch M2 sum-order witness."""

from __future__ import annotations

import copy
import json
import unittest

from tests.e3_reference.sum_order_witness import (
    FIXTURE, MAX_CANDIDATES, build, candidate, compiled_probe, validate_oracle,
)
from tests.e3_reference.transcript import canonical


class SumOrderWitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = json.loads(FIXTURE.read_bytes())
        cls.generated = build()

    def test_first_genuine_witness_is_frozen_exactly(self) -> None:
        self.assertEqual(FIXTURE.read_bytes(), canonical(self.generated))
        self.assertEqual(self.frozen["search"]["first_witness_index"], 3)
        self.assertEqual(self.frozen["search"]["compiled_candidates_tested"], 4)
        self.assertEqual(self.frozen["search"]["maximum_candidates"], MAX_CANDIDATES)
        self.assertEqual(candidate(3).hex(), self.frozen["corpus"]["source_hex"])
        self.assertEqual([batch["mask"] for batch in self.frozen["batches"]],
                         [[1, 1], [1, 1], [1, 0]])
        self.assertEqual(self.frozen["model"]["role_count_per_batch"], 21)
        self.assertEqual(self.frozen["order"], {
            "serial": "41de3b31", "grouped": "41de3b30", "wider": "41de3b31",
        })
        validate_oracle(self.frozen)

    def test_grouped_comparator_mutation_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.frozen)
        mutated["order"]["grouped"] = mutated["order"]["serial"]
        with self.assertRaisesRegex(ValueError, "checked native probe"):
            validate_oracle(mutated)

    def test_compiled_probe_search_precedes_mathematical_reference(self) -> None:
        probe, witness = compiled_probe()
        self.assertEqual(witness["index"], 3)
        self.assertTrue(all(row["order"]["serial"] == row["order"]["grouped"]
                            for row in probe["rows"][:-1]))
        self.assertEqual(self.frozen["mathematical_order"]["serial"],
                         self.frozen["mathematical_order"]["grouped"])

    def test_numerator_mutation_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.frozen)
        mutated["batches"][0]["numerator_f32"] = "00000000"
        with self.assertRaisesRegex(ValueError, "batch word checksum"):
            validate_oracle(mutated)

    def test_compiled_ce_mutation_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.frozen)
        mutated["compiled"]["ce_f32"][0][0] = "40b1902d"
        with self.assertRaisesRegex(ValueError, "checked native probe"):
            validate_oracle(mutated)


if __name__ == "__main__":
    unittest.main()
