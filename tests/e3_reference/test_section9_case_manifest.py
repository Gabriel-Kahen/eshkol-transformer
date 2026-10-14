from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from tests.e3_reference.corpus import materialize
from tests.e3_reference.reference import build
from tests.e3_reference.section9_case_manifest import (
    MAX_BYTES, accept_observed, decode, encode, expected_payload,
)


class Section9CaseManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="e3-section9-cases-")
        cls.root = Path(cls.temporary.name)
        cls.corpus = materialize(cls.root / "first")
        cls.bundle = build(cls.corpus)
        cls.payload = expected_payload(cls.corpus, cls.bundle, cls.root / "first")
        cls.raw = encode(cls.payload)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_fresh_directories_are_byte_identical_and_replay_end_cursor(self) -> None:
        second = materialize(self.root / "second")
        second_bundle = build(second)
        second_raw = encode(expected_payload(second, second_bundle, self.root / "second"))
        (self.root / "first-section9.json").write_bytes(self.raw)
        (self.root / "second-section9.json").write_bytes(second_raw)
        self.assertEqual(
            (self.root / "first-section9.json").read_bytes(),
            (self.root / "second-section9.json").read_bytes(),
        )
        self.assertEqual(self.raw, second_raw)
        self.assertLess(len(self.raw), MAX_BYTES)
        self.assertEqual(
            decode(self.raw, self.corpus, self.bundle, self.root / "first"), self.payload,
        )
        self.assertEqual(len(self.payload["cases"]), 5)
        for case in self.payload["cases"]:
            self.assertEqual(case["expected_batch_count"], len(case["rows"]))
            self.assertNotEqual(case["d2"]["start_cursor"], case["d2"]["end_cursor"])

    def test_observation_admission_requires_valid_manifest_first(self) -> None:
        observed = self.bundle["cases"][0]["synthetic"]
        accept_observed(
            self.raw, self.corpus, self.bundle, self.root / "first",
            "packed-single", observed,
        )
        malformed = copy.deepcopy(self.payload)
        malformed["cases"][0]["rows"][0]["targets"][0] ^= 1
        with self.assertRaisesRegex(ValueError, "identity differ"):
            accept_observed(
                encode(malformed), self.corpus, self.bundle, self.root / "first",
                "packed-single", observed,
            )

    def test_mutated_identity_and_order_are_rejected(self) -> None:
        edits = (
            lambda data: data["cases"][0]["d2"].__setitem__("end_cursor", "00"),
            lambda data: data["cases"][0]["d2"].__setitem__("start_cursor", "00"),
            lambda data: data["cases"][0]["d1"]["files"][1].__setitem__("sha256", "0" * 64),
            lambda data: data["cases"][0]["d1"]["files"][1].__setitem__("bytes", 1),
            lambda data: data["model"].__setitem__("parameter_f32_sha256", "0" * 64),
            lambda data: data["cases"][0]["rows"][0]["inputs"].__setitem__(0, 0),
            lambda data: data["cases"][0]["d2"].__setitem__("batch_size", True),
            lambda data: data.__setitem__("version", True),
            lambda data: data["cases"].reverse(),
            lambda data: data["cases"][0].__setitem__("extra", 1),
        )
        for edit in edits:
            with self.subTest(edit=edit):
                malformed = copy.deepcopy(self.payload)
                edit(malformed)
                with self.assertRaisesRegex(ValueError, "identity differ"):
                    decode(encode(malformed), self.corpus, self.bundle, self.root / "first")

    def test_duplicate_extra_noncanonical_and_oversize_rejected(self) -> None:
        duplicate = self.raw.replace(b'"version":1', b'"version":1,"version":1', 1)
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            decode(duplicate, self.corpus, self.bundle, self.root / "first")
        nested = self.raw.replace(b'"case_id":"packed-single"', b'"case_id":"packed-single","case_id":"packed-single"', 1)
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            decode(nested, self.corpus, self.bundle, self.root / "first")
        extra = copy.deepcopy(self.payload)
        extra["new_field"] = None
        with self.assertRaisesRegex(ValueError, "identity differ"):
            decode(encode(extra), self.corpus, self.bundle, self.root / "first")
        with self.assertRaisesRegex(ValueError, "canonical"):
            decode(self.raw[:-1] + b" \n", self.corpus, self.bundle, self.root / "first")
        with self.assertRaisesRegex(ValueError, "65536 bytes"):
            decode(self.raw + b" " * MAX_BYTES, self.corpus, self.bundle, self.root / "first")

    def test_changed_source_and_physical_d1_bytes_are_rejected(self) -> None:
        altered = copy.deepcopy(self.bundle)
        altered["model"]["parameter_f32_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity differ"):
            decode(self.raw, self.corpus, altered, self.root / "first")
        shard = self.root / "first" / "packed-single" / "shard-0000000000000000.ets"
        original = shard.read_bytes()
        try:
            shard.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "D1 physical file"):
                decode(self.raw, self.corpus, self.bundle, self.root / "first")
        finally:
            shard.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
