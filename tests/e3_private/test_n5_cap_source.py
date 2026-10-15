from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.generate_e3_n5_cap_mutant import DERIVED_SHA, SOURCE_SHA, derive, run

ROOT = Path(__file__).resolve().parents[2]


class N5CapSourceTest(unittest.TestCase):
    def test_new_eshkol_forms_are_balanced(self) -> None:
        for name in ("n5_cap_driver.esk", "n5_cap_runtime.esk"):
            source = (ROOT / "tests/e3_private" / name).read_text()
            depth = 0
            quoted = escaped = False
            for line in source.splitlines():
                comment = False
                for char in line:
                    if comment:
                        continue
                    if quoted:
                        if escaped:
                            escaped = False
                        elif char == "\\":
                            escaped = True
                        elif char == '"':
                            quoted = False
                    elif char == ";":
                        comment = True
                    elif char == '"':
                        quoted = True
                    elif char == "(":
                        depth += 1
                    elif char == ")":
                        depth -= 1
                        self.assertGreaterEqual(depth, 0, name)
            self.assertEqual(depth, 0, name)
            self.assertFalse(quoted, name)

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
        self.assertEqual(driver.count("(e3-evaluate-into! model dataset frame)"), 3)
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

    def test_compiled_successor_has_exact_oracle_and_resource_profiles(self) -> None:
        runner = (ROOT / "scripts/test-e3-n5-cap-compiled.sh").read_text()
        builder = (ROOT / "scripts/build-e3-n5-cap-test.sh").read_text()
        driver = (ROOT / "tests/e3_private/n5_cap_driver.esk").read_text()
        runtime = (ROOT / "tests/e3_private/n5_cap_runtime.esk").read_text()
        dispatcher = (ROOT / "tests/e3_private/fault_driver.esk").read_text()
        for cut in (
            "--boundary", 'diff -r "${out}/boundary-a" "${out}/boundary-b"',
            'for profile in normal repeat poison sanitized',
            'ESHKOL_ARENA_POISON="${poison}"',
            'ASAN_OPTIONS=detect_leaks=1:halt_on_error=1',
            'UBSAN_OPTIONS=halt_on_error=1',
            "peak_rss_kib=%M", '"payload.tsv"',
            'cat "${out}/${oracle}.stdout" >>"${out}/${stem}.expected"',
            'cmp "${out}/${stem}.expected" "${out}/${stem}.stdout"',
            '[[ "$(wc -l <"${out}/${stem}.stdout")" == 1 ]]',
        ):
            self.assertIn(cut, runner, cut)
        self.assertIn('E3_FAULT_SANITIZE=', builder)
        self.assertIn("(define (e3-test-n5-oracle", driver)
        self.assertIn("(e3-test-n5-unsupported! 'n5-one-over", driver)
        self.assertIn("(e3-test-skip! dataset 1)", driver)
        self.assertIn("(e3-test-n5-unsupported!\n             'n5-mutant", driver)
        self.assertIn("(eq? (vector-ref scenario 0) 'n5-oracle)", dispatcher)
        self.assertIn("(if (not mutant?) (print-metrics", runtime)
        self.assertIn("(e3-n5-rss-record rss-before rss-after)", runtime)
        self.assertIn('E3_RETENTION_RSS_PATH="${out}/${stem}.rss"', runner)


if __name__ == "__main__":
    unittest.main()
