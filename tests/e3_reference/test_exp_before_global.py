"""Authentic D2/M3 host candidate for E3 §9 exp-before-global mutation."""

from __future__ import annotations

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tests.e3_reference.corpus import materialize
from tests.e3_reference.exp_before_global import FIXTURE, derive, validate_frozen
from tests.e3_reference.reference import build
from tests.e3_reference.transcript import (
    TranscriptError, canonical, validate_observed_case,
)


class ExpBeforeGlobalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = TemporaryDirectory(prefix="e3-exp-before-global-")
        cls.corpus = materialize(Path(cls.temporary.name) / "corpus")
        cls.reference = build(cls.corpus)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_frozen_authentic_witness_is_reproducible(self) -> None:
        first = derive(self.reference)
        self.assertEqual(FIXTURE.read_bytes(), canonical(first))
        self.assertEqual(canonical(first), canonical(derive(build(self.corpus))))
        self.assertEqual(first["active_masks"], [[True, True], [True, True], [True, False]])
        self.assertEqual(first["global"]["weight_f32"], "40a00000")
        self.assertEqual(first["global"]["perplexity_f32"], "437f82e7")
        self.assertEqual(first["mutant_weighted_batch_exp_f32"], "437f8748")
        self.assertEqual(first["difference_ulps"], 1121)
        validate_frozen(self.reference, FIXTURE.read_bytes())

    def test_mutant_value_fails_exact_global_transcript(self) -> None:
        case = next(case for case in self.reference["cases"] if case["name"] == "packed-single")
        expected = tuple(case["corpus"]["batches"])
        validate_observed_case(case["synthetic"], expected)
        mutant = copy.deepcopy(case["synthetic"])
        mutant["final"]["metrics_f32"][2] = derive(self.reference)["mutant_weighted_batch_exp_f32"]
        with self.assertRaisesRegex(TranscriptError, "exponential"):
            validate_observed_case(mutant, expected)

    def test_frozen_record_mutations_are_rejected(self) -> None:
        witness = derive(self.reference)
        for field, value in (("difference_ulps", 2),
                             ("model_parameter_f32_sha256", "0" * 64)):
            with self.subTest(field=field):
                mutated = copy.deepcopy(witness)
                mutated[field] = value
                with self.assertRaisesRegex(ValueError, "differs from reference"):
                    validate_frozen(self.reference, canonical(mutated))
        with self.assertRaisesRegex(ValueError, "differs from reference"):
            validate_frozen(self.reference, FIXTURE.read_bytes() + b" ")


if __name__ == "__main__":
    unittest.main()
