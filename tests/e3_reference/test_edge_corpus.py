from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tests.d2.reference import decode_reference_cursor, resign_reference_cursor
from tests.e3_reference.edge_corpus import materialize, validate
from tests.e3_reference.transcript import canonical


def files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*")) if path.is_file()}


class EdgeCorpusTests(unittest.TestCase):
    def test_twice_byte_identical_and_exact_d2_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = materialize(root / "a")
            second = materialize(root / "b")
            self.assertEqual(files(root / "a"), files(root / "b"))
            self.assertEqual(first, second)
            self.assertEqual(json.loads((root / "a" / "edge-corpus.json").read_bytes()), first)
            self.assertEqual(canonical(first), (root / "a" / "edge-corpus.json").read_bytes())
            cases = {case["name"]: case for case in first["cases"]}
            self.assertEqual([len(cases[name]["batches"]) for name in
                              ("empty", "singleton", "entry-eos", "one-valid-final")],
                             [0, 0, 0, 2])
            self.assertEqual([row["mask"] for row in cases["one-valid-final"]["batches"]],
                             [[True, True], [True, False]])
            self.assertEqual(cases["one-valid-final"]["batches"][-1],
                             {"inputs": [0, 0], "targets": [255, 0],
                              "mask": [True, False]})
            self.assertEqual(decode_reference_cursor(bytes.fromhex(
                cases["entry-eos"]["start_cursor"])).next_ordinal, 1)
            for case in first["cases"]:
                if case["outcome"] == "invalid-state":
                    self.assertEqual(case["start_cursor"], case["traversal_end_cursor"])

    def test_rejects_schema_cursor_manifest_and_file_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "case"
            original = materialize(root)
            for mutate in (
                lambda p: p.update(extra=True),
                lambda p: p["cases"][0].update(start_ordinal=False),
                lambda p: p["cases"][3]["batches"][1]["mask"].__setitem__(1, 1),
                lambda p: p["cases"][0].update(manifest_sha256="0" * 64),
                lambda p: p["cases"][2].update(start_cursor=p["cases"][0]["start_cursor"]),
            ):
                bad = copy.deepcopy(original)
                mutate(bad)
                with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                    validate(root, bad)
            bad = copy.deepcopy(original)
            cursor = bytearray.fromhex(bad["cases"][3]["start_cursor"])
            cursor[128:160] = bytes(32)
            bad["cases"][3]["start_cursor"] = resign_reference_cursor(bytes(cursor)).hex()
            with self.assertRaises(ValueError):
                validate(root, bad)
            path = root / "singleton" / "manifest.etm"
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
            with self.assertRaises(ValueError):
                validate(root, original)

    def test_fresh_output_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "case"
            materialize(root)
            with self.assertRaises(FileExistsError):
                materialize(root)


if __name__ == "__main__":
    unittest.main()
