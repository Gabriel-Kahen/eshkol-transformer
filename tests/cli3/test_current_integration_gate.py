"""Mutation checks for the fresh-build evidence comparator, without a compiler."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class CurrentIntegrationGateTests(unittest.TestCase):
    def run_gate(self, mutation="", existing=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copyfile(ROOT / "scripts/test-cli3-current-integration.sh",
                            scripts / "test-cli3-current-integration.sh")
            (scripts / "common.sh").write_text('''
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
verify_toolchain() { :; }
require_command() { command -v "$1" >/dev/null; }
die() { echo "$*" >&2; exit 1; }
''')
            (scripts / "build-cli3.sh").write_text('''
set -euo pipefail
mkdir -p "$1/cli3.o.evidence"
for file in cli3.o libeshkol_transformer_cli3.a eshkol-transformer; do
  printf identical >"$1/$file"
done
printf same-closure >"$1/cli3.o.evidence/source-closure.txt"
printf same-undefined >"$1/cli3.o.evidence/undefined.txt"
if [[ "$1" == */second ]]; then
  case "${MUTATION:-}" in
    executable) printf changed >"$1/eshkol-transformer" ;;
    closure) printf changed >"$1/cli3.o.evidence/source-closure.txt" ;;
    inventory) printf extra >"$1/cli3.o.evidence/unexpected.txt" ;;
  esac
fi
''')
            (scripts / "test-cli3-b.sh").write_text('''
set -euo pipefail
[[ ! -e "$CLI3_B_EVIDENCE_DIR" ]]
mkdir "$CLI3_B_EVIDENCE_DIR"
printf '%s\\n' "$1" >>"${TRACE:?}"
''')
            (scripts / "test-cli3-production-generate.sh").write_text('''
set -euo pipefail
[[ $# == 2 && -f "$2/eshkol-transformer" ]]
mkdir "$1"
printf '%s\\n' "$2" >>"${TRACE:?}"
for mode in greedy categorical; do printf oracle >"$1/$mode-first.json"; done
''')
            evidence = root / "evidence"
            if existing:
                evidence.mkdir()
            trace = root / "calls.txt"
            result = subprocess.run(
                ["bash", str(scripts / "test-cli3-current-integration.sh"),
                 str(evidence)], capture_output=True, text=True,
                env={**os.environ, "MUTATION": mutation, "TRACE": str(trace)})
            calls = trace.read_text().splitlines() if trace.exists() else []
            return result, calls

    def test_equal_builds_run_both_complete_fixture_suites(self):
        result, calls = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CLI3-CURRENT-INTEGRATION-PASS", result.stdout)
        self.assertEqual(len(calls), 4)
        self.assertTrue(calls[0].endswith("/first/eshkol-transformer"))
        self.assertTrue(calls[2].endswith("/second/eshkol-transformer"))

    def test_drift_rejects_before_runtime(self):
        for mutation in ("executable", "closure", "inventory"):
            with self.subTest(mutation=mutation):
                result, calls = self.run_gate(mutation)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("CLI3-CURRENT-INTEGRATION-PASS", result.stdout)
                self.assertEqual(calls, [])

    def test_evidence_cannot_be_reused(self):
        result, calls = self.run_gate(existing=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be fresh", result.stderr)
        self.assertEqual(calls, [])
