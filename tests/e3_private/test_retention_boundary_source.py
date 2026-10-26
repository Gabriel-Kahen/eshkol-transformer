"""Fixed-source, noninstalled root-allocation boundary contract."""
import importlib.util
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/generate-e3-retention-boundary-source.py"
SPEC = importlib.util.spec_from_file_location("boundary_generator", GENERATOR)
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)


class BoundarySourceTest(unittest.TestCase):
    def test_measured_normal_imports_only_and_closed_sanitizer_policy(self):
        prefix = ROOT / "tests/e3_private/retention_arena_diagnostic_package"
        normal = Path(str(prefix) + "_undefined_symbols.txt")
        non = Path(str(prefix) + "_sanitized_noninstrumentation_undefined_symbols.txt")
        instrumentation = Path(str(prefix) + "_sanitized_instrumentation_undefined_symbols.txt")
        self.assertEqual(hashlib.sha256(normal.read_bytes()).hexdigest(),
                         "fddea863533a943fbcb356bcfab8d095158c82ee48ece1e88199690755e6e2a6")
        self.assertEqual(hashlib.sha256(non.read_bytes()).hexdigest(),
                         "eeed6ee49b420c709bffbbd0f90262455bef8f13f979ae2ba30016e15f3b1293")
        self.assertEqual(hashlib.sha256(instrumentation.read_bytes()).hexdigest(),
                         "40e37a1554ecf2df0f5b39923ca936babbb345fd9958eab2eaf86fb4cbc98271")
        words = normal.read_text().splitlines()
        measured = ["et_e3_retention_boundary_mark_v1",
                    "et_e3_retention_boundary_read_v1"]
        self.assertEqual(len(words), 181)
        self.assertEqual(words, sorted(set(words)))
        self.assertEqual([word for word in words if "retention_boundary_" in word], measured)
        with tempfile.TemporaryDirectory() as temporary:
            actual = Path(temporary) / "actual.txt"
            for variant in (words, words[:-1], sorted(words + ["unauthorized_import"])):
                actual.write_text("\n".join(variant) + "\n")
                result = subprocess.run(
                    ["bash", "-c", 'source "$1"; e1b_compare_exact_undefined_allowlist "$2" "$3"',
                     "bash", str(ROOT / "scripts/common.sh"), str(normal), str(actual)],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0 if variant is words else 1)

    def test_fixed_generation_and_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            subprocess.run(["python3", "-B", str(ROOT / "scripts/generate-e3-d2-source.py"),
                            "--output-dir", temporary], check=True,
                           stdout=subprocess.DEVNULL)
            receipt = GEN.generate(temporary)
            source = Path(temporary) / "source"
            d2 = (source / "e3_d2_dataset.esk").read_text()
            e1 = (source / "transformer/error_core.esk").read_text()
            self.assertEqual(GEN.digest(d2.encode()), receipt["d2_probe_sha256"])
            self.assertEqual(GEN.digest(e1.encode()), receipt["e1_probe_sha256"])
            self.assertLess(d2.index("(d2-boundary-mark 3)"),
                            d2.index("(region-close region caught)"))
            self.assertLess(d2.index("(region-close region caught)"),
                            d2.index("(d2-boundary-mark 4)"))
            self.assertLess(d2.index("(d2-boundary-mark 4)"),
                            d2.index("(vector d2-region-exception-marker promoted)"))
            self.assertLess(d2.index("(d2-boundary-mark 5)"),
                            d2.index("(d2-boundary-mark 6)"))
            self.assertLess(e1.index("(e1-boundary-mark 1)"),
                            e1.index("(build-error 'transformer-error-raise"))
            self.assertLess(e1.index("(e1-boundary-mark 2)"),
                            e1.index("(raise built)"))
            self.assertEqual(d2.count("et_e3_retention_boundary_mark_v1"), 1)
            self.assertEqual(e1.count("et_e3_retention_boundary_mark_v1"), 1)
            self.assertIn("(extern void d2-boundary-mark", d2)
            self.assertIn("(extern void e1-boundary-mark", e1)

    def test_generation_rejects_drift_and_second_overlay(self):
        self.assertRaises(ValueError, GEN.replace_one, "x", "y", "z")
        with tempfile.TemporaryDirectory() as temporary:
            subprocess.run(["python3", "-B", str(ROOT / "scripts/generate-e3-d2-source.py"),
                            "--output-dir", temporary], check=True,
                           stdout=subprocess.DEVNULL)
            GEN.generate(temporary)
            with self.assertRaisesRegex(ValueError, "fixed generated D2"):
                GEN.generate(temporary)

    def test_policy_and_hook_are_diagnostic_only(self):
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        policy = (ROOT / "scripts/e3-private-package-policy.sh").read_text()
        native = (ROOT / "tests/e3_private/retention_arena.cpp").read_text()
        driver = (ROOT / "tests/e3_private/retention_arena_diagnostic_driver.esk").read_text()
        closure = (ROOT / "tests/e3_private/retention_boundary_package_source_closure.txt").read_text()
        self.assertIn('"${e3_retention_arena_diagnostic_tuple}" != 1', builder)
        self.assertIn('"${e1b_tmp}/boundary-source-receipt.json"', builder)
        self.assertIn('"${e1b_tmp}/boundary-d2.esk"', builder)
        self.assertIn('"${e1b_tmp}/boundary-e1.esk"', builder)
        self.assertIn("@BUILD@/source/transformer/error_core.esk", closure)
        self.assertNotIn("lib/transformer/error_core.esk", closure)
        self.assertIn("boundary E1 overlay differs from fixed generated bytes", policy)
        self.assertIn("static int64_t boundary_bytes[10]", native)
        self.assertIn('extern "C" void et_e3_retention_boundary_mark_v1', native)
        self.assertIn('extern "C" int64_t et_e3_retention_boundary_read_v1', native)
        self.assertIn("get_global_arena_shared()", native)
        self.assertIn("'boundary-marker-count", driver)
        self.assertIn("'boundary-no-dataset-escape", driver)


if __name__ == "__main__":
    unittest.main()
