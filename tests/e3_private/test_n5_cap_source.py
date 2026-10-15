from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.generate_e3_n5_cap_mutant import DERIVED_SHA, SOURCE_SHA, derive, run

ROOT = Path(__file__).resolve().parents[2]


class N5CapSourceTest(unittest.TestCase):
    def test_single_causal_comparator_cut(self) -> None:
        source = (ROOT / "internal/e3/lib/e3_d2_restore.esk").read_bytes()
        derived = derive(source)
        self.assertEqual(source.count(b"(> remaining 8192)"), 1)
        self.assertEqual(derived.count(b"(>= remaining 8192)"), 1)
        self.assertEqual(derived.replace(b"(>= remaining 8192)",
                                         b"(> remaining 8192)"), source)
        self.assertFalse(8192 > 8192)
        self.assertTrue(8192 >= 8192)
        self.assertTrue(8193 > 8192)
        self.assertTrue(8193 >= 8192)

    def test_refuses_source_drift(self) -> None:
        source = (ROOT / "internal/e3/lib/e3_d2_restore.esk").read_bytes()
        with self.assertRaisesRegex(ValueError, "source differs"):
            derive(source + b"\n")

    def test_generated_overlay_and_exact_private_closure(self) -> None:
        normal = (ROOT / "tests/e3_private/fault_package_source_closure.txt").read_text()
        mutant = (ROOT / "tests/e3_private/fault_n5_cap_mutant_source_closure.txt").read_text()
        self.assertEqual(mutant, normal.replace(
            "internal/e3/lib/e3_d2_restore.esk",
            "@BUILD@/source/e3_d2_restore.esk"))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "e3_d2_restore.esk"
            receipt = run(output, check=False)
            self.assertEqual(receipt["source_sha256"], SOURCE_SHA)
            self.assertEqual(receipt["derived_sha256"], DERIVED_SHA)
            self.assertEqual(output.read_bytes(), derive(
                (ROOT / "internal/e3/lib/e3_d2_restore.esk").read_bytes()))
            self.assertEqual(run(output, check=True), receipt)
            output.write_bytes(output.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "overlay drifted"):
                run(output, check=True)

    def test_driver_follows_actual_preflight_and_boundary(self) -> None:
        driver = (ROOT / "tests/e3_private/n5_cap_driver.esk").read_text()
        root = (ROOT / "tests/e3_private/fault_driver_root.esk").read_text()
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        runtime = (ROOT / "tests/e3_private/n5_cap_runtime.esk").read_text()
        self.assertIn('(load "n5_cap_driver.esk")', root)
        self.assertEqual(driver.count("(e3-evaluate-into! model dataset frame)"), 2)
        self.assertIn("(e3-test-skip! dataset 1)", driver)
        self.assertIn("(d2-token-dataset-cursor dataset)", driver)
        self.assertIn("(e3-test-retention-model-snapshot model)", driver)
        self.assertIn("(e3-test-retention-modes frame)", driver)
        self.assertIn("(e3-test-retention-probes)", driver)
        self.assertIn("(e3-test-fault-publication frame destinations)", driver)
        self.assertIn("(e3-test-seed-gradient", driver)
        self.assertIn("(= (vector-ref entry 15) 29)", driver)
        self.assertIn("(= (e3-counter-ref frame 0) 16384)", driver)
        self.assertIn("(= (e3-counter-ref frame 1) 8192)", driver)
        self.assertIn('"${e3_tuple_kind}" == fault', builder)
        self.assertIn("E3_N5_CAP_MUTANT", builder)
        self.assertIn("et_e3_fault_run_v1", runtime)
        self.assertNotIn("(load ", runtime)


if __name__ == "__main__":
    unittest.main()
