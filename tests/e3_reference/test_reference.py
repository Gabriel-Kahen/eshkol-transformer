from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from tests.e3_reference.corpus import materialize
from tests.e3_reference.reference import (
    ROLE_NAMES, _read_limited, _write_exclusive, build, validate_bundle,
)
from tests.e3_reference.transcript import (
    TranscriptError, canonical, validate_mathematical_case, validate_observed_case,
)


class MathematicalReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="e3-math-reference-")
        cls.corpus = materialize(Path(cls.temporary.name) / "corpus")
        cls.reference = build(cls.corpus)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_generation_is_deterministic_and_profile_is_fixed(self) -> None:
        self.assertEqual(canonical(self.reference), canonical(build(self.corpus)))
        self.assertEqual(self.reference["kind"], "reference-bundle")
        self.assertEqual(self.reference["profile"]["identity"], "eshkol-diagnostic-byte-decoder-v1")
        self.assertEqual(self.reference["profile"]["epsilon_bits"], "3727c5ac")
        self.assertEqual(self.reference["model"]["initializer_successor"], [1, 1729, 290, 0])

    def test_all_cases_validate_exact_observed_f32_metrics(self) -> None:
        for case in self.reference["cases"]:
            with self.subTest(case=case["name"]):
                expected = tuple(case["corpus"]["batches"])
                validate_observed_case(case["synthetic"], expected)
                measurements = validate_mathematical_case(
                    case["synthetic"], case["mathematical"], expected,
                    case["mathematical_loss_f64"],
                )
                self.assertLessEqual(measurements["error"], measurements["budget"])
                for batch in case["mathematical"]:
                    self.assertEqual(tuple(role["name"] for role in batch["roles"]), ROLE_NAMES)
                    self.assertEqual(batch["roles"][-1]["shape"], [1, 2, 256])
                    shapes = {role["name"]: role["shape"] for role in batch["roles"]}
                    for name in ("QH", "KH", "VH", "AH"):
                        self.assertEqual(shapes[name], [1, 2, 2, 2])
                    for name in ("FU", "FG"):
                        self.assertEqual(shapes[name], [1, 2, 8])
                    for name in set(ROLE_NAMES) - {"QH", "KH", "VH", "AH", "FU", "FG", "Z"}:
                        self.assertEqual(shapes[name], [1, 2, 4])

    def test_target_and_mask_wiring_mutations_are_detected(self) -> None:
        case = next(item for item in self.reference["cases"] if item["name"] == "packed-single")
        expected = tuple(case["corpus"]["batches"])
        wrong_target = copy.deepcopy(case["synthetic"])
        wrong_target["batches"][0]["targets"] = [3, 0]
        with self.assertRaisesRegex(TranscriptError, "targets"):
            validate_observed_case(wrong_target, expected)
        wrong_input_type = copy.deepcopy(case["synthetic"])
        wrong_input_type["batches"][1]["inputs"] = [False, 255.0]
        with self.assertRaisesRegex(TranscriptError, "inputs"):
            validate_observed_case(wrong_input_type, expected)
        wrong_ordinal_type = copy.deepcopy(case["synthetic"])
        wrong_ordinal_type["batches"][0]["ordinal"] = False
        with self.assertRaisesRegex(TranscriptError, "ordinal"):
            validate_observed_case(wrong_ordinal_type, expected)
        wrong_mask = copy.deepcopy(case["synthetic"])
        wrong_mask["batches"][2]["mask"] = [True, True]
        with self.assertRaisesRegex(TranscriptError, "mask"):
            validate_observed_case(wrong_mask, expected)
        swapped_ce = copy.deepcopy(case["synthetic"])
        swapped_ce["batches"][0]["ce_f32"].reverse()
        with self.assertRaisesRegex(TranscriptError, "ce_f32"):
            validate_observed_case(swapped_ce, expected)
        stale_state = copy.deepcopy(case["synthetic"])
        stale_state["batches"][1]["state_f32"] = stale_state["batches"][0]["state_f32"]
        with self.assertRaisesRegex(TranscriptError, "state_f32"):
            validate_observed_case(stale_state, expected)
        swapped_outputs = copy.deepcopy(case["synthetic"])
        metrics = swapped_outputs["final"]["metrics_f32"]
        metrics[0], metrics[1] = metrics[1], metrics[0]
        with self.assertRaisesRegex(TranscriptError, "final.metrics_f32"):
            validate_observed_case(swapped_outputs, expected)

    def test_bundle_schema_and_filesystem_boundaries_are_strict(self) -> None:
        malformed = copy.deepcopy(self.reference)
        malformed["model"]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "model fields"):
            validate_bundle(malformed)
        root = Path(self.temporary.name)
        dangling = root / "dangling.json"
        dangling.symlink_to(root / "missing-target.json")
        with self.assertRaises(FileExistsError):
            _write_exclusive(dangling, b"x")
        real_parent = root / "real-parent"
        real_parent.mkdir()
        linked_parent = root / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "ancestors"):
            _write_exclusive(linked_parent / "nested" / "result.json", b"x")
        self.assertFalse((real_parent / "nested").exists())
        source = root / "source.json"
        source.write_bytes(b"{}")
        source_link = root / "source-link.json"
        source_link.symlink_to(source)
        with self.assertRaisesRegex(ValueError, "must not be a symlink"):
            _read_limited(source_link)


if __name__ == "__main__":
    unittest.main()
