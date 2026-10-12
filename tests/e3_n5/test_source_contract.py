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
        for row in manifest["selectors"]:
            self.assertIn(f'case {row["selector"]}: return {row["counter"]};', frame)
            self.assertIn(f'int64_t {row["native_entry"]}(', frame)
            self.assertIn(f'++{row["counter"]};', frame)
        for path in manifest["package_manifests"]:
            self.assertTrue((ROOT / path).is_file(), path)
        self.assertIn("#ifdef ET_E3_RETENTION_TESTING", frame)
        self.assertIn("'n5-no-native-entry", driver)
        self.assertIn("'n5-observer-positive-control", driver)
        self.assertIn("'n5-retry-output", driver)
        self.assertIn("'n5-typed-stage-two", driver)


if __name__ == "__main__":
    unittest.main()
