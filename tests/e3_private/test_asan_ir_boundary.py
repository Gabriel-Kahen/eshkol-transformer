"""Bounded source and actual Clang controls for E3 generated ASan IR."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.e3_private.check_asan_ir import verify


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "tests/shared/prepare_eshkol_asan_ir.py"
CHECKER = ROOT / "tests/e3_private/check_asan_ir.py"
MANIFEST = ROOT / "tests/e3_private/asan_ir_sources.sha256"
APPROVED_HELPER_SHA = "876543d05a6bd3bc321c31e88abb9129d11ef924c9392fd897a5bdfb54625f83"
RAW_IR = (b"define i32 @load(ptr %p) {\n"
          b"entry:\n"
          b"  %v = load i32, ptr %p, align 4\n"
          b"  ret i32 %v\n"
          b"}\n"
          b"define void @store(ptr %p, i32 %v) {\n"
          b"entry:\n"
          b"  store i32 %v, ptr %p, align 4\n"
          b"  ret void\n"
          b"}\n")


class E3AsanIRBoundaryTests(unittest.TestCase):
    def test_exact_approved_helper_and_closed_source_manifest(self) -> None:
        self.assertEqual(hashlib.sha256(HELPER.read_bytes()).hexdigest(),
                         APPROVED_HELPER_SHA)
        lines = MANIFEST.read_text().splitlines()
        self.assertEqual([line.split("  ", 1)[1] for line in lines],
                         ["tests/shared/prepare_eshkol_asan_ir.py",
                          "tests/e3_private/check_asan_ir.py"])
        checked = subprocess.run(
            ["sha256sum", "--quiet", "-c", str(MANIFEST)],
            cwd=ROOT, capture_output=True, check=False,
        )
        self.assertEqual((checked.returncode, checked.stdout, checked.stderr),
                         (0, b"", b""))

    def test_e1b_sanitized_private_branch_only(self) -> None:
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        seam = builder.split('e1b_private_compile_ir="${e1b_tmp}/private.ll"', 1)[1]
        seam = seam.split('  -o "${e1b_tmp}/private.o"', 1)[0]
        self.assertIn('if [[ "${e3_private_test_sanitize}" == 1 ]]; then', seam)
        self.assertIn('[[ "${package_policy}" == e3-private-aggregate ]]', seam)
        self.assertIn('retention|retention-first-eval-mutant|retention-overflow|mode-graph|numerical|uniform)', seam)
        self.assertIn('e1b_private_compile_ir="${e1b_tmp}/private.asan.ll"', seam)
        self.assertIn('"${e1b_cc}" "${e3_private_test_sanitize_flags[@]}" -c -x ir "${e1b_private_compile_ir}"',
                      builder)
        self.assertIn('check_asan_ir.py', builder)
        self.assertIn('"${e1b_tmp}/private.o"', builder)
        self.assertIn('"${e1b_tmp}/private.asan-imports.txt"', builder)
        self.assertIn('"${evidence_dir}.tmp.$$/private.asan.ll"', builder)
        self.assertIn('"${evidence_dir}.tmp.$$/private.ll"', builder)

    def test_generated_asan_tuple_admission_is_closed(self) -> None:
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        case = 'case "${e3_tuple_kind:-}" in' + builder.split(
            'case "${e3_tuple_kind:-}" in', 1)[1].split('esac', 1)[0] + 'esac\n'
        shell = ('set -euo pipefail\n'
                 'e3_tuple_kind=$1\n'
                 'die() { exit 77; }\n' + case)
        for tuple_name in ("retention", "retention-overflow", "mode-graph",
                           "numerical", "uniform"):
            with self.subTest(tuple_name=tuple_name):
                result = subprocess.run(["bash", "-c", shell, "bash", tuple_name],
                                        capture_output=True, check=False)
                self.assertEqual((result.returncode, result.stdout, result.stderr),
                                 (0, b"", b""))
        for tuple_name in ("", "base", "retention-overflows", "uniform-extra",
                           "diagnostic"):
            with self.subTest(tuple_name=tuple_name):
                result = subprocess.run(["bash", "-c", shell, "bash", tuple_name],
                                        capture_output=True, check=False)
                self.assertEqual((result.returncode, result.stdout, result.stderr),
                                 (77, b"", b""))

    @unittest.skipUnless(all(shutil.which(name) for name in ("clang", "nm")),
                         "Clang and nm are required for native IR controls")
    def test_real_generated_address_checks_and_tamper_rejection(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-asan-ir-") as temporary:
            directory = Path(temporary)
            raw = directory / "private.ll"
            derived = directory / "private.asan.ll"
            receipt = directory / "private.asan.json"
            object_file = directory / "private.o"
            imports = directory / "private.asan-imports.txt"
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            raw.write_bytes(RAW_IR)
            prepared = subprocess.run(
                [sys.executable, "-B", str(HELPER), str(raw), str(derived), str(receipt)],
                cwd=directory, env=environment, capture_output=True, check=False,
            )
            self.assertEqual((prepared.returncode, prepared.stdout, prepared.stderr),
                             (0, b"", b""))
            compiled = subprocess.run(
                ["clang", "-O1", "-g", "-fPIC", "-fsanitize=address,undefined",
                 "-fno-omit-frame-pointer", "-c", "-x", "ir", str(derived),
                 "-o", str(object_file)], cwd=directory,
                capture_output=True, check=False,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            symbols = subprocess.run(
                ["nm", "-u", "--format=posix", str(object_file)],
                cwd=directory, env=environment, capture_output=True, check=False,
            )
            self.assertEqual((symbols.returncode, symbols.stderr), (0, b""))
            imports.write_bytes(symbols.stdout)
            verify(raw, derived, receipt, object_file, imports)
            checked = subprocess.run(
                [sys.executable, "-B", str(CHECKER), str(raw), str(derived),
                 str(receipt), str(object_file), str(imports)],
                cwd=directory, env=environment, capture_output=True, check=False,
            )
            self.assertEqual((checked.returncode, checked.stdout, checked.stderr),
                             (0, b"", b""))
            original = derived.read_bytes()
            derived.write_bytes(original + b";tampered\n")
            with self.assertRaisesRegex(ValueError, "extra bytes"):
                verify(raw, derived, receipt, object_file, imports)
            derived.write_bytes(original)
            receipt_bytes = receipt.read_bytes()
            receipt.write_bytes(receipt_bytes.replace(b'"definitions":2', b'"definitions":1'))
            with self.assertRaisesRegex(ValueError, "receipt"):
                verify(raw, derived, receipt, object_file, imports)
            receipt.write_bytes(receipt_bytes)
            imports.write_bytes(b"__asan_init U\n")
            with self.assertRaisesRegex(ValueError, "imports"):
                verify(raw, derived, receipt, object_file, imports)
            imports.write_bytes(symbols.stdout)
            raw_object = directory / "raw.o"
            raw_imports = directory / "raw-imports.txt"
            subprocess.run(
                ["clang", "-O1", "-g", "-fPIC", "-fsanitize=address,undefined",
                 "-c", "-x", "ir", str(raw), "-o", str(raw_object)],
                cwd=directory, capture_output=True, check=True,
            )
            raw_symbols = subprocess.run(
                ["nm", "-u", "--format=posix", str(raw_object)],
                cwd=directory, capture_output=True, check=True,
            )
            raw_imports.write_bytes(raw_symbols.stdout)
            with self.assertRaisesRegex(ValueError, "report imports"):
                verify(raw, derived, receipt, raw_object, raw_imports)
            alias = directory / "alias.json"
            alias.symlink_to(receipt)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                verify(raw, derived, alias, object_file, imports)


if __name__ == "__main__":
    unittest.main()
