from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from tests.d2.reference import (
    Row, decode_reference_cursor, logical_rows, resign_reference_cursor,
)
from tests.e3_reference.corpus import FINGERPRINT, materialize, validate_payload
from tests.e3_reference.transcript import decode
from tests.t1.reference import decode_artifact


def tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class CorpusReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="e3-reference-")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_authentic_resources_are_deterministic_and_have_exact_rows(self) -> None:
        first, second = self.root / "first", self.root / "second"
        payload = materialize(first)
        materialize(second)
        self.assertEqual(tree(first), tree(second))
        self.assertEqual(decode((first / "corpus-reference.json").read_bytes()), payload)
        self.assertEqual(payload["tokenizer"]["fingerprint"], FINGERPRINT)
        self.assertEqual(payload["tokenizer"]["vocab_size"], 256)
        parsed = decode_artifact((first / "tokenizer.etok").read_bytes())
        self.assertEqual(parsed.fingerprint, FINGERPRINT)
        self.assertEqual(parsed.identity.utf8_policy, "raw")
        self.assertEqual(parsed.identity.specials, ())
        cases = {case["name"]: case for case in payload["cases"]}
        packed = cases["packed-single"]["batches"]
        self.assertEqual(
            packed,
            [
                {"inputs": [3, 197], "targets": [197, 0], "mask": [True, True]},
                {"inputs": [0, 255], "targets": [255, 7], "mask": [True, True]},
                {"inputs": [7, 0], "targets": [7, 0], "mask": [True, False]},
            ],
        )
        self.assertEqual(packed, cases["packed-sharded"]["batches"])
        self.assertEqual(cases["packed-suffix"]["batches"], packed[1:])
        self.assertEqual(sum(sum(row["mask"]) for row in packed), 5)

    def test_unpacking_and_nonzero_cursor_remain_distinct(self) -> None:
        payload = materialize(self.root / "fixture")
        cases = {case["name"]: case for case in payload["cases"]}
        three = cases["unpacked-three"]["batches"]
        pairs = cases["unpacked-pairs"]["batches"]
        self.assertEqual([sum(batch["mask"]) for batch in three], [2, 2])
        self.assertEqual([sum(batch["mask"]) for batch in pairs], [1, 1, 1])
        self.assertNotEqual(three, pairs)
        fields = decode_reference_cursor(bytes.fromhex(cases["packed-suffix"]["start_cursor"]))
        self.assertEqual(fields.next_ordinal, 1)
        self.assertEqual(fields.total_rows, 3)

    def test_generator_refuses_to_replace_an_existing_path(self) -> None:
        target = self.root / "existing"
        target.mkdir()
        with self.assertRaises(FileExistsError):
            materialize(target)
        dangling = self.root / "dangling"
        dangling.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            materialize(dangling)
        real_parent = self.root / "real-parent"
        real_parent.mkdir()
        linked_parent = self.root / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "ancestors"):
            materialize(linked_parent / "nested" / "fixture")
        self.assertFalse((real_parent / "nested").exists())

    def test_empty_singleton_and_minimal_streams_have_reachable_semantics(self) -> None:
        self.assertEqual(logical_rows(((),), 2, True), ())
        self.assertEqual(logical_rows(((7,),), 2, True), ())
        self.assertEqual(
            logical_rows(((7, 8),), 2, True),
            (Row((7, 0), (8, 0), (True, False)),),
        )

    def test_corpus_payload_types_and_cursor_identity_are_strict(self) -> None:
        payload = materialize(self.root / "fixture")
        malformed = copy.deepcopy(payload)
        malformed["cases"][0]["start_ordinal"] = False
        with self.assertRaisesRegex(ValueError, "start_ordinal"):
            validate_payload(malformed)
        malformed = copy.deepcopy(payload)
        malformed["cases"][0]["start_cursor"] = payload["cases"][2]["start_cursor"]
        with self.assertRaisesRegex(ValueError, "cursor identity or position"):
            validate_payload(malformed)
        malformed = copy.deepcopy(payload)
        malformed["cases"][0]["batches"][0]["inputs"][0] = 3.0
        with self.assertRaisesRegex(ValueError, "inputs"):
            validate_payload(malformed)
        malformed = copy.deepcopy(payload)
        malformed["cases"][0]["batches"][0]["mask"][0] = 1
        with self.assertRaisesRegex(ValueError, "mask"):
            validate_payload(malformed)
        malformed = copy.deepcopy(payload)
        malformed["cases"][0]["corpus"]["manifest_trailer_sha256"] = "z" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_payload(malformed)
        malformed = copy.deepcopy(payload)
        cursor = bytearray.fromhex(malformed["cases"][0]["start_cursor"])
        cursor[128:160] = bytes(32)
        malformed["cases"][0]["start_cursor"] = resign_reference_cursor(bytes(cursor)).hex()
        with self.assertRaisesRegex(ValueError, "cursor and manifest digest"):
            validate_payload(malformed)


if __name__ == "__main__":
    unittest.main()
