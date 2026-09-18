"""Exercise the workflow's final status check without a hosted runner."""

import os
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from itertools import product

from tests.ci.topology import jobs, run, steps


WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml"
ACCEPTANCE = WORKFLOW.with_name("acceptance.yml")
FULL_COVERAGE = WORKFLOW.with_name("full-coverage.yml")


class WorkflowGateTests(unittest.TestCase):
    def assert_n2_reference_environment_is_pinned(self, workflow):
        install = workflow.split(
            "      - name: Install pinned development oracle\n", 1
        )[1].split("      - name: Verify pinned Eshkol toolchain\n", 1)[0]
        self.assertIn("export ATEN_CPU_CAPABILITY=default", install)
        self.assertIn("export MKL_CBWR=COMPATIBLE", install)
        self.assertIn(
            'os.environ.get("ATEN_CPU_CAPABILITY") == "default"', install
        )
        self.assertIn('os.environ.get("MKL_CBWR") == "COMPATIBLE"', install)
        self.assertIn(
            'torch.backends.cpu.get_cpu_capability() == "DEFAULT"', install
        )
        self.assertEqual(workflow.count("MKL_CBWR"), 2)

    def test_final_status_accepts_only_completed_required_scope(self):
        gate = WORKFLOW.read_text().split("  f0-linux:\n", 1)[1]
        command = textwrap.dedent(gate.split("        run: |\n", 1)[1])
        for topology in ("success", "failure", "cancelled", "skipped", ""):
            for scope in ("full", "docs", "reused", "true", ""):
                for blocking in ("success", "failure", "cancelled", "skipped", ""):
                    with self.subTest(topology=topology, scope=scope, blocking=blocking):
                        result = subprocess.run(
                            ["bash", "-euo", "pipefail", "-c", command],
                            env={**os.environ, "TOPOLOGY_RESULT": topology,
                                 "SCOPE": scope, "BLOCKING_RESULT": blocking},
                            capture_output=True,
                        )
                        expected = topology == "success" and (
                            (scope, blocking) in (("full", "success"), ("docs", "skipped"), ("reused", "skipped"))
                        )
                        self.assertEqual(result.returncode == 0, expected)

    def test_workflow_wires_selection_to_matrix_and_final_gate(self):
        workflow = WORKFLOW.read_text()
        topology, rest = workflow.split("  blocking:\n", 1)
        blocking, gate = rest.split("  f0-linux:\n", 1)
        self.assertIn("fetch-depth: 0", topology)
        self.assertIn("run_full: ${{ steps.scope.outputs.run_full }}", topology)
        self.assertIn("CI_BASE_SHA: ${{ github.event.pull_request.base.sha }}", topology)
        self.assertIn("CI_HEAD_SHA: ${{ github.sha }}", topology)
        self.assertIn("run_full=true", topology)
        self.assertIn('if [[ "$CI_EVENT" == pull_request ]]; then', topology)
        self.assertIn("if: needs.topology.outputs.run_full == 'true'", blocking)
        self.assertIn("if: ${{ always() }}", gate)
        self.assertIn("needs: [topology, blocking]", gate)
        self.assertIn("SCOPE: ${{ needs.topology.outputs.scope }}", gate)

    def test_scope_requires_verified_base_for_main_prose(self):
        command = "\n".join(run(steps(jobs(WORKFLOW.read_text())["topology"])["Select execution scope"]))
        accepted = {
            ("pull_request", "false", "", ""): "docs",
            ("pull_request", "true", "", ""): "full",
            ("push", "false", "false", "true"): "docs",
            ("push", "true", "true", "false"): "reused",
            ("push", "true", "false", "false"): "full",
            # Code A pending/cancelled, docs B: no base proof must run full.
            ("push", "false", "false", "false"): "full",
            ("merge_group", "true", "", ""): "full",
            ("workflow_dispatch", "true", "", ""): "full",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            for values in product(
                ("push", "pull_request", "merge_group", "workflow_dispatch", "unknown"),
                ("true", "false", ""), ("true", "false", ""), ("true", "false", "")
            ):
                with self.subTest(values=values):
                    output.write_text("")
                    env = dict(zip(("CI_EVENT", "REQUIRED", "REUSED", "DOCS_VERIFIED"), values))
                    result = subprocess.run(
                        ["/usr/bin/bash", "-euo", "pipefail", "-c", command],
                        env={**os.environ, **env, "GITHUB_OUTPUT": str(output)},
                        capture_output=True,
                    )
                    self.assertEqual(result.returncode == 0, values in accepted)
                    if values in accepted:
                        scope = accepted[values]
                        self.assertEqual(output.read_text(),
                            f"scope={scope}\nrun_full={'true' if scope == 'full' else 'false'}\n")

    def test_n2_oracle_uses_isolated_baseline_cpu_dispatch(self):
        reusable_workflow = "uses: ./.github/workflows/full-coverage.yml"
        for path in (WORKFLOW, ACCEPTANCE):
            with self.subTest(caller=path.name):
                self.assertIn(reusable_workflow, path.read_text())

        workflow = FULL_COVERAGE.read_text()
        self.assertIn("if: matrix.suite == 'native-numerics'", workflow)
        self.assertEqual(
            workflow.count("if: matrix.suite == 'native-numerics'"),
            2,
        )
        self.assert_n2_reference_environment_is_pinned(workflow)
        self.assertIn("N2_ORACLE_PYTHON=$n2_oracle_python", workflow)
        self.assertIn("O2_ORACLE_PYTHON=$oracle_python", workflow)
        self.assertIn("A2_ORACLE_PYTHON=$oracle_python", workflow)
        self.assertIn("Q0_PYTHON=$oracle_python", workflow)

    def test_n2_reference_environment_rejects_missing_or_escaped_mkl_pin(self):
        workflow = FULL_COVERAGE.read_text()
        mutations = (
            workflow.replace("export MKL_CBWR=COMPATIBLE\\n", "", 1),
            workflow.replace(
                'assert os.environ.get("MKL_CBWR") == "COMPATIBLE"; ', "", 1
            ),
            workflow.replace(
                "    env:\n      CC: clang-21",
                "    env:\n      MKL_CBWR: COMPATIBLE\n      CC: clang-21",
                1,
            ),
        )
        for mutation in mutations:
            with self.subTest(), self.assertRaises(AssertionError):
                self.assert_n2_reference_environment_is_pinned(mutation)

    def test_n2_wrapper_overrides_inherited_dispatch_only_for_wrapped_python(self):
        workflow = FULL_COVERAGE.read_text()
        install = workflow.split(
            "      - name: Install pinned development oracle\n", 1
        )[1].split("      - name: Verify pinned Eshkol toolchain\n", 1)[0]
        wrapper_format = install.split("          printf '", 1)[1].split("' \\\n", 1)[0]
        inherited = {
            **os.environ,
            "ATEN_CPU_CAPABILITY": "inherited-aten",
            "MKL_CBWR": "inherited-mkl",
        }
        probe = (
            "import json, os; print(json.dumps({"
            "'aten': os.environ.get('ATEN_CPU_CAPABILITY'), "
            "'mkl': os.environ.get('MKL_CBWR')}))"
        )
        with tempfile.TemporaryDirectory() as directory:
            wrapper = Path(directory) / "n2-oracle-python"
            subprocess.run(
                ["bash", "-c", 'printf "$1" "$2" > "$3"', "_",
                 wrapper_format, sys.executable, str(wrapper)],
                check=True,
            )
            wrapper.chmod(0o500)
            wrapped = subprocess.run(
                [str(wrapper), "-c", probe], env=inherited,
                check=True, capture_output=True, text=True,
            )
            direct = subprocess.run(
                [sys.executable, "-c", probe], env=inherited,
                check=True, capture_output=True, text=True,
            )
        self.assertEqual(
            json.loads(wrapped.stdout), {"aten": "default", "mkl": "COMPATIBLE"}
        )
        self.assertEqual(
            json.loads(direct.stdout),
            {"aten": "inherited-aten", "mkl": "inherited-mkl"},
        )


if __name__ == "__main__":
    unittest.main()
