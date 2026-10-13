"""Negative checks for the source-only G3-R native builder contract."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.g3r import native_source_contract as contract


class NativeSourceContract(unittest.TestCase):
    def test_object_and_nested_source_plan_is_explicit(self):
        objects = {name: (source, flags) for name, source, flags in contract.OBJECTS}
        self.assertEqual(len(objects), 20)
        self.assertEqual(len(contract.input_paths()), len(set(contract.input_paths())))
        self.assertIn("ET_G3R_SAVE_BIT_WITNESS_TESTING", objects["f32_tensor"][1])
        self.assertIn("ET_O2_TESTING", objects["o2_optimizer"][1])
        self.assertIn("ET_G3R_COMMITTED_HISTORY_PRIVATE",
                      objects["g3r_same_owner_save"][1])
        self.assertIn("ET_G3R_SAME_OWNER_SAVE_PRIVATE",
                      objects["capture_partial_claim_cut_bridge"][1])
        self.assertEqual(contract.NESTED["g3t_transport"],
                         ("src/eshkol_transformer/m3_model.c",
                          "src/eshkol_transformer/m3t_transport.c"))
        self.assertNotIn("src/eshkol_transformer/m3t_transport.c",
                         {source for source, _ in objects.values()})

    def test_missing_aliased_and_wrong_git_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            source = root / "native/f32_tensor.c"
            source.parent.mkdir()
            with patch.object(contract, "git", return_value=b"approved"):
                with self.assertRaisesRegex(ValueError, "missing"):
                    contract.verify_file(root, "native/f32_tensor.c")
                source.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "pinned Git bytes"):
                    contract.verify_file(root, "native/f32_tensor.c")
                source.write_bytes(b"approved")
                self.assertEqual(len(contract.verify_file(root, "native/f32_tensor.c")), 64)
                alias = root / "native/alias.c"
                alias.symlink_to(source)
                with self.assertRaisesRegex(ValueError, "aliased"):
                    contract.verify_file(root, "native/alias.c")
                with self.assertRaisesRegex(ValueError, "noncanonical"):
                    contract.verify_file(root, "native/../native/f32_tensor.c")

    def test_dirty_wrong_head_and_writable_root_are_rejected(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            def answer(_root, *args):
                if args == ("rev-parse", "--show-toplevel"):
                    return f"{root}\n".encode()
                if args == ("rev-parse", "HEAD"):
                    return f"{contract.SOURCE_HEAD}\n".encode()
                if args == ("rev-parse", "HEAD^{tree}"):
                    return f"{contract.SOURCE_TREE}\n".encode()
                if args[0] == "status":
                    return b"?? dirty.c\n"
                raise AssertionError(args)
            with patch.object(contract, "git", side_effect=answer):
                with self.assertRaisesRegex(ValueError, "dirty"):
                    contract.verify_root(root)
            with patch.object(contract, "git", side_effect=lambda r, *a:
                              b"0" * 40 if a == ("rev-parse", "HEAD") else answer(r, *a)):
                with self.assertRaisesRegex(ValueError, "wrong exact commit"):
                    contract.verify_root(root)
            with patch.object(contract, "git", side_effect=lambda r, *a:
                              b"" if a[0] == "status" else answer(r, *a)):
                with self.assertRaisesRegex(ValueError, "not read-only"):
                    contract.verify_root(root)

    def test_depfile_membership_and_aliases_fail_closed(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            source = root / "native/f32_tensor.c"
            header = root / "native/f32_parameter_internal.h"
            source.parent.mkdir()
            source.write_bytes(b"source")
            header.write_bytes(b"header")
            depfile = root / "deps.d"
            target = "/out/f32_tensor.o"
            with patch.object(contract, "verify_root", return_value={}), \
                 patch.object(contract, "git", side_effect=lambda r, *a:
                              b"source" if a[-1].endswith(":native/f32_tensor.c")
                              else b"header"):
                depfile.write_text(f"{target}: native/f32_tensor.c native/f32_parameter_internal.h\n")
                self.assertEqual(len(contract.normalize_depfile(root, "f32_tensor", depfile,
                                                                target)), 2)
                depfile.write_text(f"{target}: native/f32_parameter_internal.h\n")
                with self.assertRaisesRegex(ValueError, "omitted required"):
                    contract.normalize_depfile(root, "f32_tensor", depfile, target)
                depfile.write_text(f"{target}: native/./f32_tensor.c\n")
                with self.assertRaisesRegex(ValueError, "aliased"):
                    contract.normalize_depfile(root, "f32_tensor", depfile, target)
                depfile.write_text(f"{target}: /etc/passwd native/f32_tensor.c\n")
                with self.assertRaisesRegex(ValueError, "escaped root"):
                    contract.normalize_depfile(root, "f32_tensor", depfile, target)


if __name__ == "__main__":
    unittest.main()
