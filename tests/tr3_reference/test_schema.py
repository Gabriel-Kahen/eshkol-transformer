from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

from tests.d2.reference import D2ReferenceError, ReferenceConfig, ReferenceDataset, decode_reference_cursor, resign_reference_cursor
from tests.d2.test_resources import write_d1_resource
from tests.tr3_reference import schema
from tests.tr3_reference.constants import TOKENIZER_FINGERPRINT
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
            ({**tensor, "shape": [2, 2]}, "shape"),
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
            a1 = payload["resume_cases"]["accumulation-1"]
            mutations = []

            changed = copy.deepcopy(payload)
            changed["resume_cases"]["accumulation-1"]["saved_next_ordinal"] = 999
            mutations.append(("saved ordinal", changed))
            changed = copy.deepcopy(payload)
            self.assertEqual(a1["end_cursor"], a1["saved_cursor"])
            changed["resume_cases"]["accumulation-1"]["end_cursor"] = a1[
                "epoch_start_cursor"
            ]
            mutations.append(("end cursor", changed))
            changed = copy.deepcopy(payload)
            changed["resume_cases"]["accumulation-1"]["epoch_start_cursor"] = a1["saved_cursor"]
            mutations.append(("epoch start", changed))
            changed = copy.deepcopy(payload)
            changed["resume_cases"]["accumulation-2-unequal-mask"]["saved_cursor"] = a1["saved_cursor"]
            mutations.append(("cursor identity/config", changed))
            changed = copy.deepcopy(payload)
            changed["resume_cases"]["accumulation-2-unequal-mask"]["contributions"][0]["epoch"] = 1
            mutations.append(("epoch sequence", changed))
            changed = copy.deepcopy(payload)
            changed["resume_cases"]["accumulation-2-unequal-mask"]["contributions"][0]["batch"]["targets"][0] ^= 1
            mutations.append(("contribution replay", changed))
            changed = copy.deepcopy(payload)
            changed["resume_cases"]["accumulation-2-unequal-mask"]["epoch_crossings"] = 0
            mutations.append(("crossing count", changed))
            for label, changed in mutations:
                with self.subTest(mutation=label), self.assertRaises(ValueError):
                    schema.validate_corpora(changed, root)

    def test_rejects_self_consistent_replacement_corpus(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tr3-profile-parent-") as raw:
            root = Path(raw) / "generated"
            payload = {
                "format": "eshkol-tr3-training-reference",
                "version": 1,
                "kind": "corpora-and-cursors",
                "profile": "tr3-fixed-n1-t2-v256-d4-v1",
                **materialize(root),
            }
            replacement = ((3, 196, 40),)
            directory = root / "train"
            manifest = write_d1_resource(
                directory,
                replacement,
                fingerprint=TOKENIZER_FINGERPRINT,
                vocab=256,
            )
            dataset = ReferenceDataset(
                replacement,
                ReferenceConfig(1, 2, None, 1, True),
                manifest_digest=manifest[-32:],
                tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
                vocab_size=256,
            )
            start = dataset.snapshot()
            batches = []
            while (batch := dataset.next_batch()) is not None:
                batches.append(
                    {
                        "inputs": list(batch.inputs[0]),
                        "targets": list(batch.targets[0]),
                        "mask": list(batch.loss_mask[0]),
                        "weight": sum(batch.loss_mask[0]),
                    }
                )
            payload["datasets"]["train"] = {
                "packing": True,
                "shards": [list(shard) for shard in replacement],
                "batches": batches,
                "start_cursor": start.hex(),
                "end_cursor": dataset.snapshot().hex(),
                "manifest_trailer_sha256": manifest[-32:].hex(),
                "files": [
                    {
                        "path": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in sorted(directory.iterdir(), key=lambda item: item.name)
                ],
            }
            with self.assertRaisesRegex(ValueError, "frozen contract"):
                schema.validate_corpora(payload, root)

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
