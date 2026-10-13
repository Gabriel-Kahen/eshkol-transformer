from __future__ import annotations

import copy
from pathlib import Path
import re
import struct
import tempfile
import unittest

from tests.e3_reference.correctness import (
    CONFIG, build_reference, materialize, selection_proof, validate_manifest,
    validate_materialized, verify_observed,
)
from tests.d2.reference import ReferenceConfig, decode_reference_cursor, encode_reference_cursor
from tests.e3_private.parse_correctness_observation import parse_correctness
from tests.e3_reference.transcript import canonical
from tests.e3_reference.transcript import validate_observed_case


class CorrectnessCasesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="e3-correctness-")
        cls.root = Path(cls.temporary.name)
        cls.manifest = materialize(cls.root / "corpus")
        cls.reference = build_reference(cls.manifest)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_two_authentic_corpora_and_independent_m3_selection(self) -> None:
        manifest = self.manifest
        self.assertEqual([case["source_bytes"] for case in manifest["cases"]],
                         ["030e1f0e1f0e", "030f39c5e7c5"])
        self.assertEqual([case["total_rows"] for case in manifest["cases"]], [3, 3])
        self.assertEqual([[row["mask"] for row in case["batches"]]
                          for case in manifest["cases"]],
                         [[[True, True], [True, True], [True, False]]] * 2)
        self.assertEqual([case["synthetic"]["final"]["metrics_f32"][3]
                          for case in self.reference["cases"]],
                         ["3f800000", "00000000"])
        proof = selection_proof(self.reference)
        self.assertEqual([case["winners"] for case in proof["cases"]],
                         [[[14, 31], [14, 31], [14, 56]],
                          [[14, 56], [196, 230], [196, 56]]])
        self.assertGreater(proof["cases"][0]["minimum_rank_gap_f64"], 2.0e-4)
        for case in self.reference["cases"]:
            verify_observed(manifest, case["name"], case["synthetic"])
        changed = copy.deepcopy(self.reference)
        changed["cases"][0]["corpus"]["declared_correct"] = False
        with self.assertRaisesRegex(ValueError, "declaration"):
            selection_proof(changed)

    def test_resource_generation_is_deterministic_and_rejects_tampering(self) -> None:
        second = self.root / "second"
        self.assertEqual(canonical(self.manifest), canonical(materialize(second)))
        first_files = {str(p.relative_to(self.root / "corpus")): p.read_bytes()
                       for p in (self.root / "corpus").rglob("*") if p.is_file()}
        second_files = {str(p.relative_to(second)): p.read_bytes()
                        for p in second.rglob("*") if p.is_file()}
        self.assertEqual(first_files, second_files)
        validate_materialized(second, self.manifest)
        with self.assertRaises(FileExistsError):
            materialize(second)
        shard = second / "all-correct" / "shard-0000000000000000.ets"
        shard.write_bytes(shard.read_bytes() + b"x")
        with self.assertRaisesRegex(ValueError, "file identity"):
            validate_materialized(second, self.manifest)

    def test_cursor_config_matches_compiled_driver_and_rejects_default_limit(self) -> None:
        driver = (Path(__file__).resolve().parents[2] /
                  "tests/e3_private/driver.esk").read_text()
        limits = re.findall(r"':maximum-batch-bytes\s+(\d+)", driver)
        self.assertEqual(limits, [str(CONFIG.maximum_batch_bytes)])
        self.assertEqual(CONFIG.maximum_batch_bytes, 34)
        for case in self.manifest["cases"]:
            for field in ("start_cursor", "end_cursor"):
                self.assertEqual(
                    decode_reference_cursor(bytes.fromhex(case[field])).config, CONFIG)
        changed = copy.deepcopy(self.manifest)
        case = changed["cases"][0]
        original = decode_reference_cursor(bytes.fromhex(case["start_cursor"]))
        default_config = ReferenceConfig(1, 2, None, 1, True)
        self.assertEqual(default_config.maximum_batch_bytes, 1_048_576)
        case["start_cursor"] = encode_reference_cursor(
            original.manifest_digest, original.tokenizer_fingerprint,
            original.vocab_size, default_config, original.total_rows,
            original.next_ordinal).hex()
        with self.assertRaisesRegex(ValueError, "cursor identity/count mismatch"):
            validate_manifest(changed)
        compiled_cursor = self.manifest["cases"][0]["start_cursor"].encode("ascii")
        raw = (b"frame\nE3RESULT\t0,0,0,0,0,0\nE3CURSOR\t" +
               compiled_cursor + b"\t" + compiled_cursor +
               b"\nE3-NUMERICAL-RUNTIME-PASS\n")
        with self.assertRaisesRegex(ValueError, "caller cursor was not restored"):
            parse_correctness(raw, changed, "all-correct")

    def test_manifest_rejects_identity_class_schema_and_rows(self) -> None:
        mutations = (
            (lambda m: m["cases"][0].__setitem__("source_bytes", "030e1f0e1f0f"), "identity"),
            (lambda m: m["cases"][0].__setitem__("declared_correct", False), "class"),
            (lambda m: m["cases"][0].__setitem__("total_rows", 2), "row count"),
            (lambda m: m["cases"][0]["batches"][0]["targets"].__setitem__(0, 15), "rows"),
            (lambda m: m["cases"][1]["batches"][0]["inputs"].__setitem__(0, 256), "rows"),
            (lambda m: m["cases"][0]["batches"].pop(), "rows"),
            (lambda m: m.__setitem__("extra", 1), "schema"),
            (lambda m: m["cases"].pop(), "case count"),
        )
        for mutate, label in mutations:
            with self.subTest(label=label):
                changed = copy.deepcopy(self.manifest)
                mutate(changed)
                with self.assertRaises(ValueError):
                    validate_manifest(changed)

    def test_observation_rejects_wrong_class_and_counts(self) -> None:
        correct = copy.deepcopy(self.reference["cases"][0]["synthetic"])
        correct["final"]["metrics_f32"][3] = "00000000"
        with self.assertRaises(ValueError):
            verify_observed(self.manifest, "all-correct", correct)
        wrong = copy.deepcopy(self.reference["cases"][1]["synthetic"])
        wrong["final"]["counters"] = [4, 3]
        with self.assertRaises(ValueError):
            verify_observed(self.manifest, "all-wrong", wrong)
        with self.assertRaisesRegex(ValueError, "case mismatch"):
            verify_observed(self.manifest, "missing", self.reference["cases"][0]["synthetic"])

    def test_all_correct_real_rows_kill_wrong_vocabulary_axis(self) -> None:
        case = self.reference["cases"][0]
        observed = case["synthetic"]
        rows = self.manifest["cases"][0]["batches"]
        self.assertEqual([row["inputs"] for row in rows],
                         [[3, 14], [31, 14], [31, 0]])
        self.assertEqual([row["mask"] for row in rows],
                         [[True, True], [True, True], [True, False]])
        self.assertEqual([len(batch["roles"]) for batch in case["mathematical"]],
                         [21, 21, 21])
        normal_winners = []
        wrong_winners = []
        for batch in observed["batches"]:
            logits = [struct.unpack(">f", bytes.fromhex(word))[0]
                      for word in batch["logits_f32"]]
            normal_winners.append([
                max(range(256), key=lambda j: (logits[position*256+j], -j))
                for position in range(2)
            ])
            wrong_winners.append([
                max(range(256), key=lambda j: (logits[2*j+position], -j))
                for position in range(2)
            ])
        self.assertEqual(normal_winners, [[14, 31], [14, 31], [14, 56]])
        self.assertEqual(wrong_winners, [[135, 143], [7, 143], [156, 22]])
        self.assertEqual(sum(int(mask and winner == target)
                             for batch, winners in zip(rows, wrong_winners)
                             for mask, winner, target in zip(batch["mask"], winners,
                                                              batch["targets"])), 0)
        mutant = copy.deepcopy(observed)
        for batch in mutant["batches"]:
            batch["correct_f32"] = "00000000"
            batch["state_f32"][2] = "00000000"
        mutant["final"]["metrics_f32"][3] = "00000000"
        with self.assertRaisesRegex(ValueError, "correct_f32: target/argmax wiring mismatch"):
            validate_observed_case(mutant, tuple(rows))


if __name__ == "__main__":
    unittest.main()
