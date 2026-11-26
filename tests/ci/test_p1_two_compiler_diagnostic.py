from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC = ROOT / ".github/workflows/p1-two-compiler-diagnostic.yml"
FULL = ROOT / ".github/workflows/full-coverage.yml"
PROTOTYPE = "dee51487413b7caa11958575866d13c8591de619"


def block(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


class P1TwoCompilerDiagnosticTests(unittest.TestCase):
    def test_trigger_and_lane_are_isolated(self):
        diagnostic = DIAGNOSTIC.read_text()
        self.assertIn("  pull_request:\n    branches: [codex/p1-two-compiler-hosted-diagnostic-20261125]", diagnostic)
        self.assertNotIn("  push:\n", diagnostic)
        self.assertNotIn("  workflow_dispatch:\n", diagnostic)
        self.assertEqual(diagnostic.count("    runs-on: ubuntu-22.04"), 1)
        self.assertIn("timeout-minutes: 120", diagnostic)
        self.assertIn("ref: ${{ github.event.pull_request.base.sha }}", diagnostic)
        self.assertIn("head.repo.full_name == github.repository", diagnostic)
        self.assertIn("startsWith(github.head_ref, 'codex/p1-two-compiler-trigger-')", diagnostic)
        self.assertIn("git merge-base --is-ancestor " + PROTOTYPE + " HEAD", diagnostic)
        self.assertIn("git diff --name-only " + PROTOTYPE + " HEAD", diagnostic)
        self.assertIn("scripts/ci-build-prerequisites.sh parameter-state", diagnostic)
        self.assertIn("make test-ci-parameters-after-build", diagnostic)
        self.assertNotIn("make test-acceptance", diagnostic)
        self.assertNotIn("make test-ci-core", diagnostic)
        self.assertIn("P1_TWO_COMPILERS: '1'", diagnostic)
        self.assertIn("if: always()", diagnostic)
        self.assertIn("sha256sum -c SHA256SUMS", diagnostic)
        self.assertIn("source_head\\t%s\\nsource_tree\\t%s", diagnostic)
        self.assertIn("run_id\\t%s\\nrun_attempt\\t%s", diagnostic)

    def test_parameter_lane_keeps_full_coverage_setup(self):
        diagnostic = DIAGNOSTIC.read_text()
        full = FULL.read_text()
        source_env = block(full, "    env:\n", "    steps:\n")
        diagnostic_env = block(diagnostic, "    env:\n", "    steps:\n")
        self.assertEqual(
            diagnostic_env,
            source_env.replace(
                "P1_COMPILER_TIMEOUT_SECONDS: ${{ matrix.suite == 'parameter-state' && '1200' || '900' }}",
                "P1_COMPILER_TIMEOUT_SECONDS: '1200'",
            ) + "      P1_TWO_COMPILERS: '1'\n",
        )
        self.assertIn("test \"$(llvm-config-21 --version)\" = '21.1.8'", diagnostic)
        self.assertIn("gpg --show-keys --with-colons", diagnostic)
        self.assertIn("fpr:::::::::6084F3CF814B57C1CF12EFD515CF4D18AF4F7421:", diagnostic)
        cache = block(full, "      - name: Restore pinned Eshkol toolchain\n", "      - name: Select pinned oracle Python\n")
        self.assertIn(cache.strip(), diagnostic)
        self.assertIn("make toolchain", diagnostic)

    def test_exact_source_authentication_accepts_only_expected_base(self):
        if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout:
            self.skipTest("requires committed clean candidate")
        diagnostic = DIAGNOSTIC.read_text()
        step = block(diagnostic, "      - name: Verify exact diagnostic source and record identity\n",
                     "      - name: Install host dependencies\n")
        script = textwrap.dedent(step.split("        run: |\n", 1)[1])
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = root / "summary"
            base_env = {**os.environ, "RUNNER_TEMP": temp,
                        "GITHUB_BASE_REF": "codex/p1-two-compiler-hosted-diagnostic-20261125",
                        "BASE_SHA": head, "PR_HEAD_SHA": head,
                        "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1",
                        "GITHUB_STEP_SUMMARY": str(summary)}
            for change, expected in (({}, True), ({"BASE_SHA": "0" * 40}, False),
                                     ({"GITHUB_BASE_REF": "main"}, False)):
                with self.subTest(change=change):
                    result = subprocess.run(["bash", "-euo", "pipefail", "-c", script],
                                            cwd=ROOT, env={**base_env, **change},
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode == 0, expected, result.stderr)
                    if expected:
                        identity = (root / "p1-two-compiler-diagnostic/identity.tsv").read_text()
                        self.assertIn(f"source_head\t{head}\n", identity)
                        self.assertIn(f"prototype_head\t{PROTOTYPE}\n", identity)
                        shutil.rmtree(root / "p1-two-compiler-diagnostic")


if __name__ == "__main__":
    unittest.main()
