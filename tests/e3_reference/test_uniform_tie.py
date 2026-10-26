"""Host controls for the genuine D1/D2 zero corpus and C1 model state."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from tests.e3_reference.uniform_tie import (
    ALIAS, BAD_SHAPES, CONFIG, FINGERPRINT, NAME, RAW, SHAPES, VOCAB_SIZE,
    build_reference, checkpoints,
    materialize, validate_materialized, zero_checkpoint,
)
from tests.c1.test_checkpoint_format import parse_golden
from tests.d2.reference import ReferenceConfig, ReferenceDataset, decode_reference_cursor
from tests.d2.test_resources import load_d1_resource


class UniformTieTest(unittest.TestCase):
    def test_authentic_zero_resources_and_complete_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            left = Path(temporary) / "left"
            right = Path(temporary) / "right"
            first = materialize(left)
            second = materialize(right)
            self.assertEqual(first["cases"][0]["source_bytes"], RAW.hex())
            self.assertEqual((left / "uniform-manifest.json").read_bytes(),
                             (right / "uniform-manifest.json").read_bytes())
            self.assertEqual(build_reference(first), build_reference(second))
            validate_materialized(left, first)
            case = build_reference(first)["cases"][0]
            self.assertEqual(case["name"], NAME)
            self.assertEqual(CONFIG.maximum_batch_bytes, 34)
            self.assertEqual(decode_reference_cursor(bytes.fromhex(
                first["cases"][0]["start_cursor"])).config, CONFIG)
            self.assertEqual(decode_reference_cursor(bytes.fromhex(
                first["cases"][0]["end_cursor"])).config, CONFIG)
            self.assertEqual([batch["mask"] for batch in first["cases"][0]["batches"]],
                             [[True, True], [True, True], [True, False]])
            self.assertEqual(case["synthetic"]["final"], {
                "counters": [5, 3],
                "metrics_f32": ["40b17218", "40a00000", "43800000", "3f800000"],
            })
            for batch in case["mathematical"]:
                self.assertEqual(len(batch["roles"]), 21)
                for role in batch["roles"]:
                    self.assertTrue(all(int(word, 16) == 0
                                        for word in role["values_f64"]), role["name"])
            for batch in case["synthetic"]["batches"]:
                self.assertEqual(batch["ce_f32"], ["40b17218", "40b17218"])
                self.assertTrue(all(word == "00000000" for word in batch["logits_f32"]))

    def test_checksum_provider_shape_tie_and_manifest_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixture"
            manifest = materialize(root)
            self.assertEqual(len(parse_golden(zero_checkpoint(),
                                              b"i2-dense-cpu-f32-v1").records), 15)
            self.assertEqual(parse_golden(zero_checkpoint(),
                                          b"i2-dense-cpu-f32-v1").aliases[0].members,
                             ALIAS)
            self.assertNotEqual(checkpoints()["bad-checksum.etcp"], zero_checkpoint())
            self.assertNotEqual(checkpoints()["bad-provider.etcp"], zero_checkpoint())
            self.assertNotEqual(checkpoints()["bad-shape.etcp"], zero_checkpoint())
            self.assertNotEqual(checkpoints()["bad-tie.etcp"], zero_checkpoint())
            self.assertNotEqual(SHAPES, BAD_SHAPES)
            for field, mutation in (
                ("source ID", lambda p: p["cases"][0].update(source_bytes="000000000001")),
                ("mask", lambda p: p["cases"][0]["batches"][2].update(mask=[True, True])),
                ("C1 provider", lambda p: p["checkpoint"].update(provider="other")),
                ("C1 tie", lambda p: p["checkpoint"].update(alias_indices=[6, 8])),
                ("C1 digest", lambda p: p["checkpoint"].update(sha256="0" * 64)),
                ("schema", lambda p: p.update(extra=1)),
            ):
                with self.subTest(field=field):
                    changed = copy.deepcopy(manifest)
                    mutation(changed)
                    with self.assertRaises(ValueError):
                        validate_materialized(root, changed, check_manifest=False)
            original = (root / "zero-state.etcp").read_bytes()
            (root / "zero-state.etcp").write_bytes(checkpoints()["bad-tie.etcp"])
            with self.assertRaises(ValueError):
                validate_materialized(root, manifest)
            (root / "zero-state.etcp").write_bytes(original)
            validate_materialized(root, manifest)

    def test_default_batch_byte_cursor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixture"
            manifest = materialize(root)
            loaded, digest = load_d1_resource(
                root / NAME, expected_fingerprint=FINGERPRINT,
                expected_vocab=VOCAB_SIZE, maximum_total_tokens=6,
            )
            old_config = ReferenceConfig(1, 2, None, 1, True)
            self.assertEqual(old_config.maximum_batch_bytes, 1_048_576)
            dataset = ReferenceDataset(
                loaded, old_config, manifest_digest=digest,
                tokenizer_fingerprint=FINGERPRINT, vocab_size=VOCAB_SIZE,
            )
            old_start = dataset.snapshot().hex()
            while dataset.next_batch() is not None:
                pass
            old_end = dataset.snapshot().hex()
            for field, old_cursor in (("start_cursor", old_start),
                                      ("end_cursor", old_end)):
                changed = copy.deepcopy(manifest)
                changed["cases"][0][field] = old_cursor
                self.assertNotEqual(old_cursor, manifest["cases"][0][field])
                self.assertEqual(decode_reference_cursor(bytes.fromhex(
                    old_cursor)).config, old_config)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_materialized(root, changed, check_manifest=False)


if __name__ == "__main__":
    unittest.main()
