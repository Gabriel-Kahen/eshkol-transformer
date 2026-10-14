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


def audit_causal_controls(driver: str, runner: str) -> None:
    edge = form(driver, "(define (e3-test-edge-run")
    opened = edge.index("(d2-token-dataset-open")
    before_open = edge[edge.index("'d2-open"):opened]
    if "(raise " in before_open:
        raise ValueError("pre control bypasses D2 open")
    checked = edge.index("'edge-zero-weight-rejection")
    post = edge.index("(e3-test-retention-check 'edge-postcondition-control")
    if not (edge.index("(e3-evaluate-into! model dataset frame)") < checked < post):
        raise ValueError("post control bypasses the evaluated outcome check")
    if "(raise " in edge[checked:post]:
        raise ValueError("post control raises before its assertion")
    if "(= (e3-counter-ref frame 0) -1)" not in edge[post:]:
        raise ValueError("post control does not fail a real assertion")
    if ('first_corpus="${missing_corpus}"' not in runner or
            '"${first_corpus}"' not in runner or
            'extra=(--edge-post-fault)' not in runner):
        raise ValueError("runner does not select both causal controls")


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
        audit_causal_controls(source, runner)
        for phase in PHASES[1:]:
            self.assertIn(f"(e3-test-edge-mark! case '{phase})", edge)
        self.assertLess(edge.index("'d2-open"),
                        edge.index("(d2-token-dataset-open"))
        self.assertLess(edge.index("(e3-evaluate-into! model dataset frame)"),
                        edge.index("'edge-postcondition-control"))
        self.assertLess(edge.index("'outcome-check"),
                        edge.index("'edge-zero-weight-rejection"))
        self.assertLess(edge.index("'edge-zero-weight-rejection"),
                        edge.index("'edge-postcondition-control"))
        self.assertIn("(= (e3-counter-ref frame 0) -1)", edge)
        self.assertIn("(flush-output-port (current-output-port))", source)
        self.assertIn("(flush-output-port port)", source)
        self.assertIn("(e3-test-edge-report! caught) (raise caught)", dispatch)
        for field in ("transformer-error-category", "transformer-error-operation",
                      "transformer-error-message"):
            self.assertIn(field, source)
        self.assertIn("--edge-post-fault", runtime)
        self.assertNotIn("--edge-pre-fault", runtime)
        self.assertIn("for control in pre post; do", runner)
        self.assertIn('missing_corpus="${out}/corpus-a-missing"', runner)
        self.assertIn('first_corpus="${missing_corpus}"', runner)
        self.assertIn('extra=(--edge-post-fault)', runner)
        self.assertIn('"${first_corpus}"', runner)
        self.assertIn('"${extra[@]}"', runner)
        self.assertIn('E3-RETENTION-FAIL: edge-postcondition-control', runner)
        self.assertIn("edge_phase_check", runner)
        self.assertIn('expected_status=134', runner)

    def test_causal_source_mutations_are_rejected(self) -> None:
        source, runner = DRIVER.read_text(), RUNNER.read_text()
        mutants = (
            (source.replace("(let ((dataset (d2-token-dataset-open",
                            "(raise 'synthetic-pre-control)\n"
                            "    (let ((dataset (d2-token-dataset-open", 1), runner),
            (source.replace("(e3-test-retention-check 'edge-postcondition-control",
                            "(raise 'synthetic-post-control)\n"
                            "            (e3-test-retention-check 'edge-postcondition-control", 1), runner),
            (source, runner.replace('first_corpus="${missing_corpus}"',
                                    'first_corpus="${out}/corpus-a-empty"', 1)),
        )
        for mutated_source, mutated_runner in mutants:
            with self.assertRaises(ValueError):
                audit_causal_controls(mutated_source, mutated_runner)

    def test_phase_checker_rejects_false_pass_and_distinguishes_faults(self) -> None:
        normal = "".join(f"E3-EDGE-PHASE case={case} phase={phase}\n"
                         for case in CASES for phase in PHASES)
        check_phases(normal, "normal")
        pre = "".join(f"E3-EDGE-PHASE case=empty phase={phase}\n"
                      for phase in PHASES[:PHASES.index("d2-open") + 1])
        pre += ('E3-EDGE-EXCEPTION case=empty phase=d2-open '
                'category=io operation=token-dataset-open '
                'message="required file is missing"\n')
        post = "".join(f"E3-EDGE-PHASE case=empty phase={phase}\n"
                       for phase in PHASES[:PHASES.index("outcome-check") + 1])
        post += "E3-EDGE-PHASE case=empty phase=postcondition-control\n"
        check_phases(pre, "pre")
        check_phases(post, "post")
        for valid, mode in ((pre, "pre"), (post, "post")):
            for bad, wrong_mode in ((valid.rsplit("\n", 2)[0] + "\n", mode),
                                    (valid, "normal"),
                                    (valid, "post" if mode == "pre" else "pre"),
                                    (normal + valid, "normal")):
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
