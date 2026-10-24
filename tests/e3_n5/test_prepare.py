from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.d2.reference import decode_reference_cursor
from tests.d2.test_resources import ResourceError, load_d1_resource
from tests.e3_n5.prepare import CASES, FINGERPRINT, VOCAB_SIZE, materialize


def files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


class N5PreflightTests(unittest.TestCase):
    def test_exact_authentic_edges_and_one_over(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "first"
            receipt = materialize(root)
            self.assertEqual(receipt, json.loads((root / "preflight.json").read_text()))
            cases = {case["name"]: case for case in receipt["cases"]}
            self.assertEqual(tuple(cases), CASES)
            self.assertEqual(
                [(cases[name]["total_rows"], cases[name]["skip"],
                  cases[name]["remaining_rows"], cases[name]["active_tokens"])
                 for name in CASES],
                [(0, 0, 0, 0), (0, 0, 0, 0), (1, 1, 0, 0),
                 (1, 0, 1, 1), (8192, 0, 8192, 16384),
                 (8193, 0, 8193, 16386)],
            )
            self.assertEqual(cases["one-token"]["first_row"],
                             {"inputs": [3, 0], "targets": [197, 0],
                              "mask": [True, False]})
            self.assertEqual(cases["one-token"]["rows"],
                             [cases["one-token"]["first_row"]])
            self.assertEqual(cases["cap-8192"]["shard_tokens"], [2048] * 8 + [1])
            self.assertEqual(cases["cap-8193"]["shard_tokens"], [2048] * 8 + [3])
            for name in CASES:
                case = cases[name]
                loaded, digest = load_d1_resource(
                    root / name, expected_fingerprint=FINGERPRINT,
                    expected_vocab=VOCAB_SIZE, maximum_manifest_bytes=1048576,
                    maximum_shard_bytes=1048576, maximum_total_tokens=65536)
                self.assertEqual(list(map(len, loaded)), case["shard_tokens"])
                self.assertEqual(digest.hex(), case["manifest_digest"])
                entry = decode_reference_cursor(bytes.fromhex(case["entry_cursor"]))
                end = decode_reference_cursor(bytes.fromhex(case["end_cursor"]))
                self.assertEqual((entry.total_rows, entry.next_ordinal),
                                 (case["total_rows"], case["skip"]))
                self.assertEqual(end.next_ordinal, case["total_rows"])
                self.assertEqual(entry.manifest_digest, digest)
                self.assertEqual(len(case["rows"]), case["remaining_rows"])
                self.assertEqual(sum(sum(row["mask"]) for row in case["rows"]),
                                 case["active_tokens"])

    def test_repeatable_and_rejects_existing_or_corrupt_resource(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first, second = (Path(temporary) / name for name in ("first", "second"))
            materialize(first)
            materialize(second)
            self.assertEqual(files(first), files(second))
            with self.assertRaises(FileExistsError):
                materialize(first)
            linked = Path(temporary) / "linked"
            linked.symlink_to(first, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "ancestors"):
                materialize(linked / "nested")
            shard = first / "cap-8193" / "shard-0000000000000008.ets"
            data = bytearray(shard.read_bytes())
            data[-33] ^= 1
            shard.write_bytes(data)
            with self.assertRaises(ResourceError):
                load_d1_resource(first / "cap-8193",
                                 expected_fingerprint=FINGERPRINT,
                                 expected_vocab=VOCAB_SIZE)


if __name__ == "__main__":
    unittest.main()
