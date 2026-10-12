from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class EdgeObserverSourceTests(unittest.TestCase):
    def test_private_observer_calls_real_evaluator_and_checks_six_outputs(self):
        source = (ROOT / "tests/e3_private/edge_driver.esk").read_text()
        self.assertIn("(e3-evaluate-into! model dataset frame)", source)
        self.assertIn("(e3-edge-six=? before after)", source)
        self.assertIn("(e3-edge-cursor=? cursor end)", source)
        self.assertIn("(transformer-error-category caught)", source)
        self.assertIn("et_e3_test_published_counter_raw_v1", source)
        self.assertIn("(d2-token-dataset-close! dataset)", source)
        self.assertNotIn("(set! success #t)", source)
        bridge = (ROOT / "tests/e3_private/edge_bridge.c").read_text()
        runtime = (ROOT / "tests/e3_private/edge_runtime.esk").read_text()
        self.assertIn("et_e3_private_edge_run_cabi_v1", bridge)
        self.assertIn("et_e3_edge_run_v1", runtime)

    def test_observer_remains_in_separate_noninstalled_root(self):
        source = (ROOT / "tests/e3_private/edge_driver_root.esk").read_text()
        self.assertIn('"../../native/e3_private_driver_root.esk"', source)
        self.assertIn('"edge_driver.esk"', source)
        self.assertNotIn('edge_driver.esk', (ROOT / "native/e3_private_driver_root.esk").read_text())


if __name__ == "__main__":
    unittest.main()
