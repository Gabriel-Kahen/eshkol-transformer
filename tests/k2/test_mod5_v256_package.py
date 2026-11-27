"""Static and negative checks for the separate K2 package evidence."""

from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tests.k2 import check_mod5_v256_package as audit


class Mod5V256PackageTests(unittest.TestCase):
    def test_expected_scheme_dependencies_extend_only_the_canonical_sources(self):
        canonical = set(audit.manifest(audit.DEFAULT, "source_closure").read_text().splitlines())
        expected = set(audit.manifest(
            audit.PROFILE, "scheme_expected_closure").read_text().splitlines())
        private_source = "native/k2_mod5_v256_extension.esk"
        witness = "tests/k2/mod5_v256_runtime.esk"
        self.assertEqual(expected, canonical | {private_source, witness})
        self.assertFalse(expected & {"lib/transformer/capabilities.esk", "native/c2_wave2_root.esk"})
        self.assertIn(f'(load "{Path(private_source).name}")', (audit.ROOT / witness).read_text())
        for source in canonical:
            self.assertNotIn(Path(private_source).name, (audit.ROOT / source).read_text())

    def test_canonical_check_rejects_each_kind_of_inventory_drift(self):
        with TemporaryDirectory() as directory:
            evidence = Path(directory)
            for name, kind in audit.CANONICAL.items():
                (evidence / name).write_bytes(audit.manifest(audit.DEFAULT, kind).read_bytes())
            with redirect_stdout(io.StringIO()) as output:
                audit.canonical(evidence)
            self.assertEqual(len(output.getvalue().splitlines()), len(audit.CANONICAL))
            for name in audit.CANONICAL:
                with self.subTest(inventory=name):
                    path = evidence / name
                    original = path.read_bytes()
                    path.write_bytes(original + b"et_k2_private_mod5_v256_require_v1\n")
                    with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, name):
                        audit.canonical(evidence)
                    path.write_bytes(original)

    def test_compiler_dependency_receipt_rejects_missing_and_foreign_sources(self):
        with TemporaryDirectory() as directory:
            depfile = Path(directory) / "fixture.d"
            expected = audit.manifest(audit.PROFILE, "scheme_expected_closure")
            sources = expected.read_text().splitlines()
            depfile.write_text("fixture.o: " + (" \\\n ").join(sources) + "\n")
            actual = {path.relative_to(audit.ROOT).as_posix()
                      for path in audit.depfile_paths(depfile)}
            audit.check_closure(actual, expected)
            with self.assertRaisesRegex(ValueError, "missing"):
                audit.check_closure(actual - {sources[0]}, expected)
            with self.assertRaisesRegex(ValueError, "unexpected"):
                audit.check_closure(actual | {"native/c2_wave2_root.esk"}, expected)

    def test_opt_in_authority_is_absent_from_installed_manifests(self):
        for kind in ("defined_symbols", "public_exports", "public_strings", "private_renames"):
            with self.subTest(inventory=kind):
                self.assertNotIn("v256", audit.manifest(audit.DEFAULT, kind).read_text())
        default = audit.manifest(audit.PROFILE, "default_native_symbols").read_text()
        opt_in = audit.manifest(audit.PROFILE, "native_symbols").read_text()
        self.assertNotIn("et_k2_private_mod5_v256", default)
        self.assertEqual(opt_in.count("D FUNC GLOBAL HIDDEN et_k2_private_mod5_v256_"), 5)
        exports = audit.manifest(audit.PROFILE, "native_exports").read_text()
        self.assertNotIn("_private_", exports)


if __name__ == "__main__":
    unittest.main()
