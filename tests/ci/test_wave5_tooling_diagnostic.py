"""Keep optional Wave 5 execution separate from required CI evidence."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tests.ci.topology import field, jobs, run, steps


ROOT = Path(__file__).resolve().parents[2]
GATES = (
    "test-t2-trainer", "test-data4-pipeline", "test-beam-search",
    "test-constrained-decode", "test-paged-kv", "test-spec-decode",
)


class Wave5DiagnosticTests(unittest.TestCase):
    def test_dispatch_and_called_workflow_are_diagnostic_only(self):
        ci = (ROOT / ".github/workflows/ci.yml").read_text()
        called = (ROOT / ".github/workflows/wave5-tooling-diagnostic.yml").read_text()
        entries = jobs(ci)
        self.assertIn("      wave5_tooling_diagnostic:\n", ci)
        self.assertIn("        default: false\n        type: boolean\n", ci)
        self.assertIn("&& 'wave5-tooling-diagnostic' || github.event.pull_request.number || github.ref", ci)
        self.assertIn("  cancel-in-progress: ${{ !inputs.wave5_tooling_diagnostic }}", ci)
        self.assertIn("      expected_head: ${{ inputs.wave5_expected_head }}\n", entries["wave5-tooling-diagnostic"])
        self.assertIn("      expected_tree: ${{ inputs.wave5_expected_tree }}\n", entries["wave5-tooling-diagnostic"])
        self.assertEqual(field(entries["wave5-tooling-diagnostic"], 4, "uses"),
                         "./.github/workflows/wave5-tooling-diagnostic.yml")
        self.assertIn("  workflow_call:\n", called)
        self.assertNotIn("  workflow_dispatch:\n", called)
        self.assertIn("permissions:\n  contents: read\n", called)
        self.assertIn("runs-on: ubuntu-22.04", called)
        for needle in (
            "refs/heads/codex/wave5-tooling-*",
            '[[ "$GITHUB_SHA" == "$EXPECTED_HEAD" ]]',
            "git rev-parse 'HEAD^{tree}'",
            "llvm-config-21 --version",
            "= '21.1.8'",
            "make toolchain",
            "persist-credentials: false",
            "scripts/run-wave5-tooling-diagnostic.sh",
            "if: always()",
            "sha256sum -c SHA256SUMS",
            "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, called)
        self.assertNotIn("full-coverage.yml", called)
        self.assertNotIn("continue-on-error", called)
        self.assertIn("Run six focused leaf gates", steps(jobs(called)["tooling"]))
        runner = (ROOT / "scripts/run-wave5-tooling-diagnostic.sh").read_text()
        self.assertIn("timeout --kill-after=10s 2400s make", runner)
        self.assertNotIn("timeout --foreground", runner)

    def test_mocked_gate_orchestration_keeps_all_receipts_on_failure(self):
        script = ROOT / "scripts/run-wave5-tooling-diagnostic.sh"
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            bin_dir = work / "bin"
            bin_dir.mkdir()
            fake_make = bin_dir / "make"
            fake_make.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'ran %s\\n' \"$1\"\n"
                "[[ \"$1\" != \"${MOCK_FAIL_GATE:-}\" ]]\n"
            )
            fake_make.chmod(0o755)
            for failure in ("", "test-constrained-decode"):
                with self.subTest(failure=failure):
                    evidence = work / ("pass" if not failure else "failure")
                    evidence.mkdir()
                    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                           "MOCK_FAIL_GATE": failure}
                    result = subprocess.run(["bash", str(script), str(evidence)],
                                            cwd=ROOT, env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, bool(failure), result.stderr)
                    rows = (evidence / "gates.tsv").read_text().splitlines()
                    self.assertEqual(rows[0], "gate\tresult\texit_code")
                    self.assertEqual([row.split("\t")[0] for row in rows[1:]], list(GATES))
                    for gate, row in zip(GATES, rows[1:]):
                        expected = "fail\t1" if gate == failure else "pass\t0"
                        self.assertEqual(row.split("\t", 1)[1], expected)
                        self.assertEqual((evidence / gate / "stdout").read_text(), f"ran {gate}\n")
                        self.assertEqual((evidence / gate / "stderr").read_text(), "")

    def test_dispatch_guard_rejects_conflicting_modes(self):
        ci = (ROOT / ".github/workflows/ci.yml").read_text()
        guard = "\n".join(run(steps(jobs(ci)["topology"])["Reject simultaneous diagnostics"]))
        for attention in (False, True):
            for data4 in (False, True):
                for wave5 in (False, True):
                    with self.subTest(attention=attention, data4=data4, wave5=wave5):
                        env = {**os.environ,
                               "ATTENTION_DIAGNOSTIC": str(attention).lower(),
                               "DATA4_DIAGNOSTIC": str(data4).lower(),
                               "WAVE5_DIAGNOSTIC": str(wave5).lower()}
                        result = subprocess.run(["bash", "-e", "-c", guard], env=env)
                        self.assertEqual(result.returncode == 0,
                                         sum((attention, data4, wave5)) <= 1)

    def test_timeout_terminates_gate_descendants(self):
        script = ROOT / "scripts/run-wave5-tooling-diagnostic.sh"
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            bin_dir = work / "bin"
            bin_dir.mkdir()
            marker = work / "late-child-output"
            fake_make = bin_dir / "make"
            fake_make.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \"$1\" == test-t2-trainer ]]; then\n"
                "  (sleep 0.5; printf late > \"$MOCK_MARKER\") &\n"
                "  wait\n"
                "fi\n"
            )
            fake_make.chmod(0o755)
            fake_timeout = bin_dir / "timeout"
            fake_timeout.write_text(
                "#!/usr/bin/env bash\n"
                "shift 2\n"
                "exec /usr/bin/timeout --kill-after=1s 0.1s \"$@\"\n"
            )
            fake_timeout.chmod(0o755)
            evidence = work / "evidence"
            evidence.mkdir()
            env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                   "MOCK_MARKER": str(marker)}
            result = subprocess.run(["bash", str(script), str(evidence)],
                                    cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("test-t2-trainer\tfail\t124", (evidence / "gates.tsv").read_text())
            self.assertEqual(len((evidence / "gates.tsv").read_text().splitlines()), 7)
            subprocess.run(["sleep", "0.7"], check=True)
            self.assertFalse(marker.exists(), "timed-out descendant wrote after the gate")

    def test_default_make_targets_do_not_depend_on_optional_gates(self):
        make = (ROOT / "Makefile").read_text()
        for gate in GATES:
            self.assertIn(f"{gate}:\n\t/usr/bin/bash scripts/{gate}.sh", make)
        default_targets = make.split("\ntest: ", 1)[1].split("\n\n", 1)[0]
        self.assertFalse(any(gate in default_targets for gate in GATES))


if __name__ == "__main__":
    unittest.main()
