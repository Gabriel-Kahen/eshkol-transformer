"""Exercise the workflow's final status check without a hosted runner."""

import os
from pathlib import Path
import subprocess
import textwrap
import unittest


WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml"
ACCEPTANCE = WORKFLOW.with_name("acceptance.yml")


class WorkflowGateTests(unittest.TestCase):
    def test_final_status_accepts_only_completed_required_scope(self):
        gate = WORKFLOW.read_text().split("  f0-linux:\n", 1)[1]
        command = textwrap.dedent(gate.split("        run: |\n", 1)[1])
        for topology in ("success", "failure", "cancelled", "skipped", ""):
            for full in ("true", "false", ""):
                for blocking in ("success", "failure", "cancelled", "skipped", ""):
                    with self.subTest(topology=topology, full=full, blocking=blocking):
                        result = subprocess.run(
                            ["bash", "-euo", "pipefail", "-c", command],
                            env={**os.environ, "TOPOLOGY_RESULT": topology,
                                 "RUN_FULL": full, "BLOCKING_RESULT": blocking},
                            capture_output=True,
                        )
                        expected = topology == "success" and (
                            (full, blocking) in (("true", "success"), ("false", "skipped"))
                        )
                        self.assertEqual(result.returncode == 0, expected)

    def test_workflow_wires_selection_to_matrix_and_final_gate(self):
        workflow = WORKFLOW.read_text()
        topology, rest = workflow.split("  blocking:\n", 1)
        blocking, gate = rest.split("  f0-linux:\n", 1)
        self.assertIn("fetch-depth: 0", topology)
        self.assertIn("run_full: ${{ steps.changes.outputs.run_full }}", topology)
        self.assertIn("CI_BASE_SHA: ${{ github.event.pull_request.base.sha }}", topology)
        self.assertIn("CI_HEAD_SHA: ${{ github.sha }}", topology)
        self.assertIn("run_full=true", topology)
        self.assertIn('if [[ "$CI_EVENT" == pull_request ]]; then', topology)
        self.assertIn("if: needs.topology.outputs.run_full == 'true'", blocking)
        self.assertIn("if: ${{ always() }}", gate)
        self.assertIn("needs: [topology, blocking]", gate)
        self.assertIn("RUN_FULL: ${{ needs.topology.outputs.run_full }}", gate)

    def test_n2_oracle_uses_isolated_baseline_cpu_dispatch(self):
        for path in (WORKFLOW, ACCEPTANCE):
            with self.subTest(workflow=path.name):
                workflow = path.read_text()
                self.assertIn("ATEN_CPU_CAPABILITY=default", workflow)
                self.assertIn(
                    'torch.backends.cpu.get_cpu_capability() == "DEFAULT"',
                    workflow,
                )
                self.assertIn("N2_ORACLE_PYTHON=$n2_oracle_python", workflow)
                self.assertIn("O2_ORACLE_PYTHON=$oracle_python", workflow)
                self.assertIn("A2_ORACLE_PYTHON=$oracle_python", workflow)
                self.assertIn("Q0_PYTHON=$oracle_python", workflow)


if __name__ == "__main__":
    unittest.main()
