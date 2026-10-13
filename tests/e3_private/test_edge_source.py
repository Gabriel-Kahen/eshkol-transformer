"""Source-only isolation and assertion controls for the pending E3 edge gate."""

from pathlib import Path
import unittest

from tests.e3_private.test_retention_first_baseline import form


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "tests/e3_private/retention_driver.esk"
RUNTIME = ROOT / "tests/e3_private/edge_runtime.esk"
RUNNER = ROOT / "scripts/test-e3-edge-private.sh"


class EdgeSourceTests(unittest.TestCase):
    def test_closed_edge_path_checks_outcome_and_state(self) -> None:
        source = DRIVER.read_text()
        edge = form(source, "(define (e3-test-edge-run")
        self.assertLess(edge.index("(e3-test-skip! dataset 1)"),
                        edge.index("(cursor (d2-token-dataset-cursor dataset))"))
        self.assertLess(edge.index("(outputs (e3-test-edge-outputs"),
                        edge.index("(e3-evaluate-into! model dataset frame)"))
        for assertion in (
            "'edge-zero-weight-rejection", "'edge-tail-success", "'edge-cursor",
            "'edge-native-owners", "'edge-modes", "'edge-model-gradient",
            "'edge-rng", "'edge-close", "(source-code 12)",
            "(= (e3-test-u32-ref (vector-ref destinations 3)) 1065353216)",
        ):
            self.assertIn(assertion, edge)
        self.assertNotIn("logit", edge)
        self.assertNotIn("cross-entropy", edge)
        self.assertIn("((eq? phase 'edge)", source)
        self.assertEqual(source.count("(define (e3-test-edge-run"), 1)

    def test_runner_and_runtime_are_complete_and_private(self) -> None:
        runtime = RUNTIME.read_text()
        runner = RUNNER.read_text()
        for marker in ("(define (check ", "(define (print-cursor ",
                       "(let loop ((index 0) (paths args))"):
            self.assertTrue(form(runtime, marker))
        self.assertIn("'retention 'edge", runtime)
        self.assertIn("et_e3_retention_run_v1", runtime)
        self.assertNotIn("diagnostic-evaluate-fixed!", runtime)
        for marker in ("normal-a normal-b poisoned-a sanitized",
                       "tests/e3_private/edge_runtime.esk",
                       "tests.e3_horizon.edge_generate",
                       "tests.e3_horizon.edge_accuracy",
                       "e3_retention_verify_sanitizer_imports"):
            self.assertIn(marker, runner)
        self.assertNotIn("n5_runtime.esk", runner)


if __name__ == "__main__":
    unittest.main()
