#!/usr/bin/env python3
"""Source-only negative controls for retained terminal compiler evidence."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_p2_g2_terminal_emitted_evidence import (
    TERMINAL_FFI, check_depfile, check_dynamic, check_imports,
    check_source_record, check_terminal_ffi,
    dep_words, exact_files,
)
from prepare_terminal_asan_ir import prepare, scan_pair


class TerminalEmittedEvidenceTests(unittest.TestCase):
    def test_dependency_rule_rejects_extra_duplicate_and_variable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.d"
            path.write_text("/out/a.o: /src/a.c /src/a.h\n")
            self.assertEqual(dep_words(path), ("/out/a.o", ["/src/a.c", "/src/a.h"]))
            for wrong in (
                "/out/a.o: /src/a.c /src/a.h /src/a.h\n",
                "/out/a.o: /src/a.c $(EXTRA)\n",
                "/out/a.o: /src/a.c\n/out/b.o: /src/b.c\n",
                "/out/a.o: /src/a.c # hidden\n",
            ):
                path.write_text(wrong)
                with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                    dep_words(path)

    def test_dependency_and_source_identity_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.esk"
            source.write_text("(display 1)\n")
            target = root / "output.o"
            dep = root / "output.d"
            dep.write_text(f"{target}: {source}\n")
            check_depfile(dep, target, {source})
            with self.assertRaises(ValueError):
                check_depfile(dep, target, {source, root / "unreviewed.h"})
            from hashlib import sha256
            record = f"{sha256(source.read_bytes()).hexdigest()}  {source}\n"
            (root / "normal-source.before").write_text(record)
            (root / "normal-source.after").write_text(record)
            check_source_record(root, "normal", source)
            source.write_text("(display 2)\n")
            with self.assertRaises(ValueError):
                check_source_record(root, "normal", source)

    def test_missing_empty_and_symlink_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "raw.o").write_bytes(b"object")
            exact_files(root, {"raw.o"})
            with self.assertRaises(ValueError):
                exact_files(root, {"raw.o", "raw.o.ll"})
            (root / "raw.o.ll").write_bytes(b"")
            with self.assertRaises(ValueError):
                exact_files(root, {"raw.o", "raw.o.ll"})
            (root / "raw.o.ll").unlink()
            (root / "raw.o.ll").symlink_to("raw.o")
            with self.assertRaises(ValueError):
                exact_files(root, {"raw.o", "raw.o.ll"})

    def test_import_closure_rejects_new_symbol_and_wrong_wrap(self):
        check_imports({"known", "malloc", "__real_malloc"}, {"known"})
        for symbols in ({"known", "secret_callback"},
                        {"known", "__real_secret_callback"}):
            with self.subTest(symbols=symbols), self.assertRaises(ValueError):
                check_imports(symbols, {"known"})
        with self.assertRaises(ValueError):
            check_dynamic({"unreviewed_symbol"}, "normal")

    def test_terminal_ffi_rejects_missing_and_extra_import(self):
        check_terminal_ffi(set(TERMINAL_FFI))
        with self.assertRaises(ValueError):
            check_terminal_ffi(set(TERMINAL_FFI) - {"et_g3c4_private_p2g2_terminal_commit_v1"})
        with self.assertRaises(ValueError):
            check_terminal_ffi(set(TERMINAL_FFI) | {"et_g3c4_private_p2g2_terminal_hidden_v1"})

    def test_asan_rewrite_is_exact_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, derived, receipt = (root / name for name in
                                     ("raw.ll", "asan.ll", "receipt.json"))
            raw.write_bytes(b"define i32 @f() {\nentry:\n  ret i32 0\n}\n")
            prepare(raw, derived, receipt)
            self.assertEqual(scan_pair(raw, derived)["definitions"], 1)
            derived.write_bytes(derived.read_bytes().replace(b"ret i32 0", b"ret i32 1"))
            with self.assertRaises(ValueError):
                scan_pair(raw, derived)
            raw.write_bytes(b"define i32 @f() sanitize_address {\n  ret i32 0\n}\n")
            with self.assertRaises(ValueError):
                scan_pair(raw, derived)


if __name__ == "__main__":
    unittest.main()
