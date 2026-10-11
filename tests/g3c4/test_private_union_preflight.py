"""Source-only negative preflight; no compiler or native fixture is launched."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = (Path(__file__).resolve().parents[2] / "scripts" /
          "preflight-g3c4-public-t1-private-union.py")
SPEC = importlib.util.spec_from_file_location("private_union_preflight", SCRIPT)
assert SPEC and SPEC.loader
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class PrivateUnionPreflightTest(unittest.TestCase):
    def test_one_root_and_mixed_package_negatives(self) -> None:
        good = b'; fixture\n(load "g3g_manual_package_root.esk")\n(display 1)\n'
        preflight.one_root(good, "fixture")
        for mutant in (good.replace(b"(display 1)",
                                    b'(load "g3g_manual_package_root.esk")'),
                       good.replace(b"g3g_manual_package_root.esk",
                                    b"other_root.esk"),
                       good + b"(require transformer.generation)\n",
                       good + b"libeshkol_transformer_g3g_manual\n"):
            with self.subTest(mutant=mutant), self.assertRaises(ValueError):
                preflight.one_root(mutant, "fixture")

    def test_closure_rejects_malformed_and_duplicate(self) -> None:
        self.assertEqual(preflight.closure(b"native/a.c\nnative/b.c\n"),
                         ("native/a.c", "native/b.c"))
        for mutant in (b"", b"native/a.c\nnative/a.c\n",
                       b"native/a.c\n\n", b"../native/a.c\n",
                       b"/native/a.c\n", b"native//a.c\n",
                       b"native/a.c\n../outside\n"):
            with self.subTest(mutant=mutant), self.assertRaises(ValueError):
                preflight.closure(mutant)

    def test_fixed_manifest_rejects_missing_extra_and_reorder(self) -> None:
        expected = ("one.o", "two.o")
        preflight.fixed_manifest(b"one.o\ntwo.o\n", expected)
        for mutant in (b"one.o\n", b"one.o\ntwo.o\nthree.o\n",
                       b"two.o\none.o\n"):
            with self.subTest(mutant=mutant), self.assertRaises(ValueError):
                preflight.fixed_manifest(mutant, expected)

    def test_ordinary_rejects_changed_type_and_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input").write_bytes(b"pin")
            self.assertEqual(preflight.ordinary(root, "input"), b"pin")
            (root / "link").symlink_to("input")
            (root / "directory").mkdir()
            (root / "directory" / "inner").write_bytes(b"pin")
            (root / "parent-link").symlink_to("directory")
            for name in ("link", "directory", "parent-link/inner",
                         "missing", "../input", "/input"):
                with self.subTest(name=name), self.assertRaises(
                        (ValueError, OSError)):
                    preflight.ordinary(root, name)

if __name__ == "__main__":
    unittest.main()
