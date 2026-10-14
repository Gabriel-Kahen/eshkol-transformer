"""Source-only isolation and assertion controls for the pending E3 edge gate."""

from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

from tests.e3_private.test_retention_first_baseline import form
from tests.e3_private.edge_phase_check import CASES, PHASES, check as check_phases


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "tests/e3_private/retention_driver.esk"
RUNTIME = ROOT / "tests/e3_private/edge_runtime.esk"
RUNNER = ROOT / "scripts/test-e3-edge-private.sh"
BUILDER = ROOT / "scripts/build-e3-retention-test.sh"


class EdgeSourceTests(unittest.TestCase):
    def test_closed_edge_path_checks_outcome_and_state(self) -> None:
        source = DRIVER.read_text()
        edge = form(source, "(define (e3-test-edge-run")
        self.assertLess(edge.index("(e3-test-skip! dataset 1)"),
                        edge.index("(d2-token-dataset-cursor dataset)"))
        self.assertLess(edge.index("(e3-test-edge-outputs frame destinations)"),
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

    def test_durable_phase_and_exception_controls(self) -> None:
        source = DRIVER.read_text()
        edge = form(source, "(define (e3-test-edge-run")
        dispatch = form(source, "(define (e3-test-retention-dispatch")
        runtime = RUNTIME.read_text()
        runner = RUNNER.read_text()
        for phase in PHASES[1:]:
            self.assertIn(f"(e3-test-edge-mark! case '{phase})", edge)
        self.assertLess(edge.index("'d2-open"), edge.index("edge-pre-eval-control"))
        self.assertLess(edge.index("(e3-evaluate-into! model dataset frame)"),
                        edge.index("edge-post-eval-control"))
        self.assertLess(edge.index("'outcome-check"),
                        edge.index("'edge-zero-weight-rejection"))
        self.assertIn("(flush-output-port port)", source)
        self.assertIn("(e3-test-edge-report! caught) (raise caught)", dispatch)
        for field in ("transformer-error-category", "transformer-error-operation",
                      "transformer-error-message"):
            self.assertIn(field, source)
        for control in ("--edge-pre-fault", "--edge-post-fault"):
            self.assertIn(control, runtime)
        self.assertIn("for control in pre post; do", runner)
        self.assertIn('"--edge-${control}-fault"', runner)
        self.assertIn('die "edge ${control} exception control unexpectedly passed"', runner)
        self.assertIn("edge_phase_check", runner)
        self.assertIn("[[ \"${status}\" -eq 1 ]]", runner)

    def test_phase_checker_rejects_false_pass_and_distinguishes_faults(self) -> None:
        normal = "".join(f"E3-EDGE-PHASE case={case} phase={phase}\n"
                         for case in CASES for phase in PHASES)
        check_phases(normal, "normal")
        for mode, stop, value in (("pre", "d2-open", "edge-pre-eval-control"),
                                  ("post", "post-eval", "edge-post-eval-control")):
            phase_lines = "".join(f"E3-EDGE-PHASE case=empty phase={phase}\n"
                                  for phase in PHASES[:PHASES.index(stop) + 1])
            exception = (f"E3-EDGE-EXCEPTION case=empty phase={stop} "
                         f"category=non-transformer-error value={value}\n")
            check_phases(phase_lines + exception, mode)
            for bad, wrong_mode in ((phase_lines, mode),
                                    (phase_lines + exception, "normal"),
                                    (phase_lines + exception, "post" if mode == "pre" else "pre"),
                                    (normal + exception, "normal")):
                with self.assertRaises(ValueError):
                    check_phases(bad, wrong_mode)

    def test_missing_or_nonexecutable_builder_fails_before_output_creation(self) -> None:
        self.assertTrue(BUILDER.is_file())
        self.assertTrue(os.access(BUILDER, os.X_OK))
        for present in (False, True):
            with self.subTest(builder_present=present), tempfile.TemporaryDirectory() as tmp:
                scripts = Path(tmp) / "scripts"
                scripts.mkdir()
                shutil.copyfile(ROOT / "scripts/common.sh", scripts / "common.sh")
                shutil.copyfile(RUNNER, scripts / RUNNER.name)
                if present:
                    builder = scripts / BUILDER.name
                    builder.write_text("#!/usr/bin/env bash\nexit 99\n")
                    builder.chmod(0o644)
                output = Path(tmp) / "edge-output"
                result = subprocess.run(
                    ["bash", str(scripts / RUNNER.name), str(output)],
                    capture_output=True, text=True, check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("E3 retention test builder is missing or not executable",
                              result.stderr)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
