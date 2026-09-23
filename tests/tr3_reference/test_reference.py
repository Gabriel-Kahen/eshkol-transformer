"""Pinned-PyTorch regeneration and self-consistency tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from tests.tr3_reference.generate_reference import build, canonical
from tests.tr3_reference.schema import validate_output, validate_summary


HERE = Path(__file__).resolve().parent


class TR3ReferenceTests(unittest.TestCase):
    def test_complete_reference_matches_frozen_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tr3-reference-") as raw:
            root = Path(raw) / "generated"
            manifest = build(root)
            validate_output(root, manifest)
            expected = json.loads((HERE / "reference_manifest.json").read_text())
            self.assertEqual(manifest, expected)
            trajectory = json.loads((root / "trajectory.json").read_text())
            self.assertEqual(len(trajectory["entries"]), 33)
            self.assertEqual(len(trajectory["parameter_paths"]), 14)

    def test_independent_regenerations_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tr3-repeat-") as raw:
            first = Path(raw) / "first"
            second = Path(raw) / "second"
            first_manifest = build(first)
            second_manifest = build(second)
            self.assertEqual(canonical(first_manifest), canonical(second_manifest))
            for name in ("corpora.json", "summary.json", "trajectory.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_thresholds_heldout_and_accumulation_controls(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tr3-threshold-") as raw:
            root = Path(raw) / "generated"
            manifest = build(root)
            values = manifest["measurements"]
            self.assertLess(values["final_training_mean_ce"], 0.05)
            self.assertEqual(values["final_active_targets"], values["final_active_argmax"])
            self.assertEqual(values["final_active_unique_argmax"], [True, True])
            self.assertEqual(values["heldout_mask_weights"], [2, 1])
            self.assertLess(values["final_heldout_global_mean_ce"], values["initial_heldout_global_mean_ce"])
            summary = json.loads((root / "summary.json").read_text())
            proof = summary["accumulation_reference"]
            self.assertLessEqual(proof["max_abs_difference"], 2.0e-6)
            self.assertGreater(proof["wrong_per_batch_mean_max_abs_difference"], 0.0)

            mutations = []
            changed = copy.deepcopy(summary)
            changed["accumulation_reference"]["contribution_ordinals"] = [1, 0]
            mutations.append(("contribution ordinals", changed))
            changed = copy.deepcopy(summary)
            changed["accumulation_reference"]["contribution_ordinals"] = [0.0, 1.0]
            mutations.append(("ordinal types", changed))
            changed = copy.deepcopy(summary)
            changed["accumulation_reference"]["weights"] = [2.0, 1.0]
            mutations.append(("weight types", changed))
            changed = copy.deepcopy(summary)
            changed["accumulation_reference"]["bit_identical"] = True
            mutations.append(("bit identity", changed))
            changed = copy.deepcopy(summary)
            changed["accumulation_reference"]["max_abs_difference"] = -1.0
            mutations.append(("negative delta", changed))
            changed = copy.deepcopy(summary)
            changed["accumulation_reference"]["summed_numerator_gradient_sha256"] = "0" * 64
            mutations.append(("gradient digest", changed))
            changed = copy.deepcopy(summary)
            changed["accumulation_reference"]["within_parameter_tolerance"] = False
            mutations.append(("tolerance flag", changed))
            changed = copy.deepcopy(summary)
            changed["training"]["before"]["numerator_bits_after_f64_sum"] = "0000000000000000"
            mutations.append(("numerator f64 word", changed))
            changed = copy.deepcopy(summary)
            changed["heldout"]["after"]["mean_bits_f64"] = "0000000000000000"
            mutations.append(("mean f64 word", changed))
            changed = copy.deepcopy(summary)
            changed["provenance"]["source_sha256"].pop("tests/d2/reference.py")
            mutations.append(("provenance omission", changed))
            changed = copy.deepcopy(summary)
            changed["provenance"]["source_sha256"]["tests/unexpected.py"] = "0" * 64
            mutations.append(("provenance addition", changed))
            for label, changed in mutations:
                with self.subTest(mutation=label), self.assertRaises(ValueError):
                    validate_summary(changed)

    def test_manifest_detects_generated_artifact_mutation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tr3-tamper-") as raw:
            root = Path(raw) / "generated"
            manifest = build(root)
            path = root / "trajectory.json"
            damaged = bytearray(path.read_bytes())
            damaged[-2] ^= 1
            path.write_bytes(damaged)
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                validate_output(root, manifest)


if __name__ == "__main__":
    unittest.main()
