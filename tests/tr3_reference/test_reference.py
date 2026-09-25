"""Pinned-PyTorch regeneration and self-consistency tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from tests.tr3_reference.generate_reference import build, canonical
from tests.tr3_reference.schema import validate_corpora, validate_output, validate_summary


HERE = Path(__file__).resolve().parent


class TR3ReferenceTests(unittest.TestCase):
    @staticmethod
    def _write_artifact(
        root: Path,
        manifest: dict[str, object],
        name: str,
        payload: object,
    ) -> None:
        encoded = canonical(payload)
        (root / name).write_bytes(encoded)
        manifest["artifacts"][name] = {
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }

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
            corpora = json.loads((root / "corpora.json").read_text())
            corpus_batches = validate_corpora(corpora, root)
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
                    validate_summary(changed, corpus_batches)

    def test_refreshed_digest_cross_field_mutations_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tr3-cross-field-") as raw:
            root = Path(raw) / "generated"
            manifest = build(root)
            summary = json.loads((root / "summary.json").read_text())
            mutations = []

            changed = copy.deepcopy(summary)
            changed_manifest = copy.deepcopy(manifest)
            changed["training"]["after"]["batches"][0]["argmax"][0] ^= 1
            changed_manifest["measurements"]["final_active_argmax"][0] ^= 1
            mutations.append(("final argmax", changed, changed_manifest, "unique logits argmax"))

            changed = copy.deepcopy(summary)
            changed_manifest = copy.deepcopy(manifest)
            changed["training"]["after"]["batches"][0]["unique_argmax"][0] = False
            changed_manifest["measurements"]["final_active_unique_argmax"][0] = False
            mutations.append(("final uniqueness", changed, changed_manifest, "unique logits argmax"))

            changed = copy.deepcopy(summary)
            changed_manifest = copy.deepcopy(manifest)
            changed["heldout"]["train_transitions"] = [[1, 2]]
            changed["heldout"]["heldout_transitions"] = [[3, 4]]
            mutations.append(("invented transitions", changed, changed_manifest, "authenticated corpus"))

            changed = copy.deepcopy(summary)
            changed_manifest = copy.deepcopy(manifest)
            changed["training"]["before"]["batches"][0]["inputs"][0] ^= 1
            mutations.append(("unbound training input", changed, changed_manifest, "authenticated corpus"))

            changed = copy.deepcopy(summary)
            changed_manifest = copy.deepcopy(manifest)
            evaluation = changed["heldout"]["after"]
            numerator_word = evaluation["batches"][0]["numerator_bits"]
            numerator = struct.unpack(">f", bytes.fromhex(numerator_word))[0]
            altered = struct.unpack(">f", struct.pack(">f", numerator + 0.25))[0]
            evaluation["batches"][0]["numerator_bits"] = struct.pack(">f", altered).hex()
            batch_numerators = [
                struct.unpack(">f", bytes.fromhex(batch["numerator_bits"]))[0]
                for batch in evaluation["batches"]
            ]
            evaluation["numerator"] = sum(batch_numerators, 0.0)
            evaluation["mean"] = evaluation["numerator"] / evaluation["weight"]
            evaluation["numerator_bits_after_f64_sum"] = struct.pack(">d", evaluation["numerator"]).hex()
            evaluation["mean_bits_f64"] = struct.pack(">d", evaluation["mean"]).hex()
            changed_manifest["measurements"]["final_heldout_global_mean_ce"] = evaluation["mean"]
            mutations.append(("unbound numerator", changed, changed_manifest, "masked per-token"))

            for label, changed, changed_manifest, message in mutations:
                self._write_artifact(root, changed_manifest, "summary.json", changed)
                with self.subTest(mutation=label), self.assertRaisesRegex(ValueError, message):
                    validate_output(root, changed_manifest)

    def test_refreshed_integral_type_lookalikes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tr3-integral-types-") as raw:
            root = Path(raw) / "generated"
            manifest = build(root)
            payloads = {
                name: json.loads((root / name).read_text())
                for name in ("corpora.json", "summary.json", "trajectory.json")
            }
            path = next(iter(payloads["trajectory.json"]["parameter_paths"]))
            mutations = []

            changed_manifest = copy.deepcopy(manifest)
            changed_manifest["version"] = True
            mutations.append(("manifest version bool", None, None, changed_manifest))
            changed_manifest = copy.deepcopy(manifest)
            changed_manifest["configuration"]["initializer_seed"] = 1729.0
            mutations.append(("configuration seed float", None, None, changed_manifest))
            changed_manifest = copy.deepcopy(manifest)
            changed_manifest["configuration"]["optimizer"]["schedule"]["factor"] = True
            mutations.append(("nested float bool", None, None, changed_manifest))
            changed_manifest = copy.deepcopy(manifest)
            changed_manifest["measurements"]["heldout_batches"] = 2.0
            mutations.append(("derived count float", None, None, changed_manifest))
            changed_manifest = copy.deepcopy(manifest)
            changed_manifest["measurements"]["final_active_targets"][0] = 197.0
            mutations.append(("derived token float", None, None, changed_manifest))
            changed_manifest = copy.deepcopy(manifest)
            changed_manifest["artifacts"]["summary.json"]["bytes"] = float(
                changed_manifest["artifacts"]["summary.json"]["bytes"]
            )
            mutations.append(("artifact size float", None, None, changed_manifest))

            changed = copy.deepcopy(payloads["summary.json"])
            changed["version"] = True
            mutations.append(("summary version bool", "summary.json", changed, copy.deepcopy(manifest)))
            changed = copy.deepcopy(payloads["summary.json"])
            changed["updates"] = 32.0
            mutations.append(("summary updates float", "summary.json", changed, copy.deepcopy(manifest)))
            changed = copy.deepcopy(payloads["summary.json"])
            changed["initializer"]["successor"][0] = 1.0
            mutations.append(("initializer word float", "summary.json", changed, copy.deepcopy(manifest)))
            changed = copy.deepcopy(payloads["summary.json"])
            changed["training"]["before"]["token_count"] = 2.0
            mutations.append(("evaluation count float", "summary.json", changed, copy.deepcopy(manifest)))
            changed = copy.deepcopy(payloads["summary.json"])
            changed["provenance"]["proposal"]["section"] = 13.0
            mutations.append(("proposal section float", "summary.json", changed, copy.deepcopy(manifest)))

            changed = copy.deepcopy(payloads["trajectory.json"])
            changed["entries"][1]["completed_updates"] = 1.0
            mutations.append(("completed update float", "trajectory.json", changed, copy.deepcopy(manifest)))
            changed = copy.deepcopy(payloads["trajectory.json"])
            changed["parameter_shapes"][path][0] = 4.0
            mutations.append(("top shape float", "trajectory.json", changed, copy.deepcopy(manifest)))
            changed = copy.deepcopy(payloads["trajectory.json"])
            changed["entries"][0]["parameters"][path]["shape"][0] = 4.0
            mutations.append(("tensor shape float", "trajectory.json", changed, copy.deepcopy(manifest)))
            changed = copy.deepcopy(payloads["trajectory.json"])
            changed["entries"][1]["optimizer_state"][path]["step"] = True
            mutations.append(("optimizer step bool", "trajectory.json", changed, copy.deepcopy(manifest)))

            changed = copy.deepcopy(payloads["corpora.json"])
            changed["version"] = True
            mutations.append(("corpora version bool", "corpora.json", changed, copy.deepcopy(manifest)))
            changed = copy.deepcopy(payloads["corpora.json"])
            changed["tokenizer"]["vocab_size"] = 256.0
            mutations.append(("vocab size float", "corpora.json", changed, copy.deepcopy(manifest)))
            changed = copy.deepcopy(payloads["corpora.json"])
            changed["datasets"]["train"]["shards"][0][0] = 3.0
            mutations.append(("corpus token float", "corpora.json", changed, copy.deepcopy(manifest)))
            changed = copy.deepcopy(payloads["corpora.json"])
            changed["resume_cases"]["accumulation-1"]["contributions"][0]["epoch"] = 1.0
            mutations.append(("resume epoch float", "corpora.json", changed, copy.deepcopy(manifest)))

            for label, name, changed, changed_manifest in mutations:
                for original_name, original in payloads.items():
                    (root / original_name).write_bytes(canonical(original))
                if name is not None:
                    self._write_artifact(root, changed_manifest, name, changed)
                with self.subTest(mutation=label), self.assertRaises(ValueError):
                    validate_output(root, changed_manifest)

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
