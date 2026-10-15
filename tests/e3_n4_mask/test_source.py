"""Bounded source and checked-resource custody for the N4 mask census."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tests.e3_n4_mask.prepare import CASES, materialize

ROOT = Path(__file__).resolve().parents[2]


class MaskCensusSource(unittest.TestCase):
    def test_checked_resources_are_reproducible(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            self.assertEqual(materialize(base / "a"), materialize(base / "b"))
            self.assertEqual(
                sorted((path.relative_to(base / "a"), path.read_bytes())
                       for path in (base / "a").rglob("*") if path.is_file()),
                sorted((path.relative_to(base / "b"), path.read_bytes())
                       for path in (base / "b").rglob("*") if path.is_file()),
            )
        self.assertEqual({name: mask for name, _, _, mask in CASES},
                         {"p2": [1, 0], "u2": [1, 0],
                          "p3": [1, 1], "u3": [1, 1]})

    def test_same_registry_fixture_and_native_boundary(self) -> None:
        root = (ROOT / "tests/e3_n4_mask/root.esk").read_text()
        fixture = (ROOT / "tests/e3_n4_mask/runtime.esk").read_text()
        writer = (ROOT / "native/d2_native.c").read_text()
        self.assertEqual(root.count('(load "../../native/e3_private_root.esk")'), 1)
        self.assertEqual(root.count('(load "runtime.esk")'), 1)
        self.assertEqual(fixture.count('(load '), 0)
        for required in (
            "t1-private-tokenizer-byte", "d2-token-dataset-open",
            "d2-token-dataset-next-batch", "d2-token-batch-loss-mask",
            "d2-token-tensor-with-view-internal", "d2-token-batch-release!",
            "d2-token-dataset-close!", '(n4-case root "p2" #t 0)',
            '(n4-case root "u2" #f 0)', '(n4-case root "p3" #t 1)',
            '(n4-case root "u3" #f 1)',
        ):
            self.assertIn(required, fixture)
        self.assertEqual(writer.count("batch->mask_data[destination + index] = 1u;"), 1)


if __name__ == "__main__":
    unittest.main()
