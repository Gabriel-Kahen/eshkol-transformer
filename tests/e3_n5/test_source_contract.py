from __future__ import annotations

import unittest
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class N5SourceContractTests(unittest.TestCase):
    def test_preflight_order_and_test_only_observer(self) -> None:
        restore = (ROOT / "internal/e3/lib/e3_d2_restore.esk").read_text()
        extension = (ROOT / "native/e3_private_extension.esk").read_text()
        frame = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        driver = (ROOT / "tests/e3_mode_graph/driver.esk").read_text()
        self.assertLess(restore.index("(> remaining 8192)"),
                        restore.index("(vector-set! record 4 dataset)"))
        self.assertLess(extension.index("(e3-stage-set! entry 2)"),
                        extension.index("(e3-d2-prepare! frame dataset)"))
        self.assertLess(extension.index("(e3-d2-prepare! frame dataset)"),
                        extension.index("(e3-native-acquire (vector-ref entry 4))"))
        self.assertLess(extension.index("(e3-native-acquire (vector-ref entry 4))"),
                        extension.index("(p1-e3-mode-enter!"))
        self.assertLess(extension.index("(p1-e3-mode-enter!"),
                        extension.rindex("(e3-traverse! frame dataset)"))
        manifest = json.loads((ROOT / "tests/e3_n5/observer_manifest.json").read_text())
        self.assertEqual(manifest["kind"], "e3-n5-private-observer-v1")
        self.assertEqual(manifest["package"], "mode-graph-noninstalled")
        self.assertEqual(manifest["compile_guard"], "ET_E3_RETENTION_TESTING")
        self.assertEqual(manifest["probe_symbol"], "et_e3_test_retention_probe_v1")
        self.assertEqual([row["selector"] for row in manifest["selectors"]],
                         [29, 30, 31, 32])
        self.assertEqual([row["failure_delta"] for row in manifest["selectors"]],
                         [0, 0, 0, 0])
        self.assertEqual([row["retry_delta"] for row in manifest["selectors"]],
                         [1, 3, 63, 3])
        self.assertEqual([row["selector"] for row in manifest["last_row_selectors"]],
                         [33, 34, 35, 36])
        self.assertEqual([row["exact_limit_value"]
                          for row in manifest["last_row_selectors"]],
                         [255, 0, 1, 0])
        for row in manifest["selectors"]:
            self.assertIn(f'case {row["selector"]}: return {row["counter"]};', frame)
            self.assertIn(f'int64_t {row["native_entry"]}(', frame)
            self.assertIn(f'++{row["counter"]};', frame)
        for selector, field in ((33, "e3_n5_last_targets[0]"),
                                (34, "e3_n5_last_targets[1]"),
                                (35, "e3_n5_last_mask[0]"),
                                (36, "e3_n5_last_mask[1]")):
            self.assertIn(f"case {selector}: return {field};", frame)
        for path in manifest["package_manifests"]:
            self.assertTrue((ROOT / path).is_file(), path)
        self.assertIn("#ifdef ET_E3_RETENTION_TESTING", frame)
        self.assertIn("'n5-no-native-entry", driver)
        self.assertIn("'n5-observer-positive-control", driver)
        self.assertIn("'n5-retry-output", driver)
        self.assertIn("'n5-typed-stage-two", driver)

    def test_exact_limit_compiled_witness_is_fail_closed(self) -> None:
        driver = (ROOT / "tests/e3_mode_graph/driver.esk").read_text()
        runtime = (ROOT / "tests/e3_n5/runtime.esk").read_text()
        gate = (ROOT / "scripts/test-e3-n5-private.sh").read_text()
        generator = (ROOT / "scripts/generate-e3-mode-graph-source.py").read_text()
        self.assertIn("(vector-length paths) 3", driver)
        self.assertLess(driver.index("'n5-exact-first-no-partial-output"),
                        driver.index("'n5-exact-counts"))
        self.assertIn("(equal? publication (e3-mg-publication frame destinations))",
                      driver)
        self.assertIn("(vector 255 0 1 0)", driver)
        self.assertIn("(vector-ref result 1) 1182792704", driver)
        self.assertIn("(vector-ref result 4) 16383", driver)
        self.assertIn("(vector-ref result 5) 8192", driver)
        self.assertIn("(and (>= ordinal 4) (not (vector-ref e3-mode-graph-cell 9)))",
                      generator)
        self.assertIn("(vector-set! e3-mode-graph-cell 9 #t)", driver)
        self.assertIn("(vector-set! e3-mode-graph-cell 9 #f)", driver)
        self.assertIn("(car (cdr (cdr args)))", runtime)
        self.assertIn('"${out}/corpus-a/exact-limit"', gate)
        self.assertIn('cmp - "${out}/run-${variant}.stdout"', gate)
        self.assertNotIn("grep -Fx 'E3-N5-PRIVATE-PASS", gate)


if __name__ == "__main__":
    unittest.main()
