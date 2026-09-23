from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest

from tests.d2.reference import D2ReferenceError, ReferenceConfig, ReferenceDataset, decode_reference_cursor, resign_reference_cursor
from tests.tr3_reference import schema
from tests.tr3_reference.corpus import materialize


class TR3SchemaTests(unittest.TestCase):
    def test_rejects_noncanonical_json_and_unknown_schema_field(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tr3-schema-") as raw:
            path = Path(raw) / "record.json"
            path.write_text('{ "a": 1 }\n')
            with self.assertRaisesRegex(ValueError, "noncanonical JSON"):
                schema._load(path, 100)
            record = {
                "argmax": [0, 0], "inputs": [0, 0], "mask": [True, False],
                "numerator_bits": "3f800000", "per_token_loss_bits": ["3f800000", "00000000"],
                "targets": [0, 0], "unique_argmax": [True, True], "weight": 1,
                "unexpected": 0,
            }
            with self.assertRaisesRegex(ValueError, "fields differ"):
                schema._batch(record, "batch")

    def test_rejects_bad_shape_truncated_word_and_token_index(self) -> None:
        tensor = {
            "shape": [4], "encoding": "ieee754-f32-hex-be",
            "data": ["00000000"] * 4,
        }
        schema._tensor(tensor, [4], "tensor")
        for mutation, message in (
            ({**tensor, "shape": [2, 2]}, "shape mismatch"),
            ({**tensor, "data": tensor["data"][:-1]}, "incorrect value count"),
            ({**tensor, "data": ["00000000"] * 3 + ["7f800000"]}, "nonfinite"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                schema._tensor(mutation, [4], "tensor")
        batch = {
            "argmax": [0, 0], "inputs": [0, 0], "mask": [True, False],
            "numerator_bits": "3f800000", "per_token_loss_bits": ["3f800000", "00000000"],
            "targets": [256, 0], "unique_argmax": [True, True], "weight": 1,
        }
        with self.assertRaisesRegex(ValueError, "expected exact integer"):
            schema._batch(batch, "batch")
        batch["targets"] = [-1, 0]
        with self.assertRaisesRegex(ValueError, "expected exact integer"):
            schema._batch(batch, "batch")
        batch["targets"] = [0, 0]
        batch["mask"] = [False, False]
        batch["weight"] = 0
        with self.assertRaisesRegex(ValueError, "empty masks"):
            schema._batch(batch, "batch")

    def test_authentic_resources_and_resume_traces_validate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tr3-corpus-parent-") as raw:
            root = Path(raw) / "generated"
            payload = {
                "format": "eshkol-tr3-training-reference",
                "version": 1,
                "kind": "corpora-and-cursors",
                "profile": "tr3-fixed-n1-t2-v256-d4-v1",
                **materialize(root),
            }
            schema.validate_corpora(payload, root)
            self.assertEqual(
                [item["batch"]["weight"] for item in payload["resume_cases"]["accumulation-2-unequal-mask"]["contributions"]],
                [1, 2],
            )

    def test_rejects_cursor_checksum_identity_and_ordinal(self) -> None:
        config = ReferenceConfig(1, 2, None, 1, True)
        dataset = ReferenceDataset(
            ((3, 197, 41),), config, manifest_digest=bytes(32),
            tokenizer_fingerprint="fp", vocab_size=256,
        )
        cursor = dataset.snapshot()
        damaged = bytearray(cursor)
        damaged[-1] ^= 1
        with self.assertRaisesRegex(D2ReferenceError, "checksum mismatch"):
            decode_reference_cursor(bytes(damaged))

        foreign = ReferenceDataset(
            ((3, 197, 41),), config, manifest_digest=bytes([1]) * 32,
            tokenizer_fingerprint="fp", vocab_size=256,
        ).snapshot()
        with self.assertRaisesRegex(D2ReferenceError, "identity/config/corpus mismatch"):
            dataset.seek(foreign)

        bad_ordinal = bytearray(cursor)
        total = struct.unpack_from("<Q", bad_ordinal, 112)[0]
        struct.pack_into("<Q", bad_ordinal, 120, total + 1)
        malformed = resign_reference_cursor(bytes(bad_ordinal))
        self.assertGreater(decode_reference_cursor(malformed).next_ordinal, total)
        with self.assertRaisesRegex(D2ReferenceError, "next ordinal exceeds"):
            dataset.seek(malformed)


if __name__ == "__main__":
    unittest.main()
