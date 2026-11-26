"""Cheap synthetic exit cases for the trial launcher's actual EXIT receipt."""

from pathlib import Path
import os
import re
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/launch-c2-sha-span-save-trial.sh"


class ReceiptTests(unittest.TestCase):
    def run_case(self, original: int, fault: str = "") -> tuple[subprocess.CompletedProcess[str], str]:
        source = LAUNCHER.read_text()
        match = re.search(r"^finish\(\) \{\n.*?^\}\n(?=trap finish EXIT$)",
                          source, re.MULTILINE | re.DOTALL)
        self.assertIsNotNone(match)
        with tempfile.TemporaryDirectory(prefix="c2-span-save-receipt.") as temporary:
            base = Path(temporary)
            evidence = base / "evidence"
            evidence.mkdir()
            wrapper = base / "sha256sum"
            wrapper.write_text(
                "#!/usr/bin/bash\n"
                'if [[ "${SIM_FAIL:-}" == generate && "${1:-}" != -c ]]; then exit 9; fi\n'
                'if [[ "${SIM_FAIL:-}" == verify && "${1:-}" == -c ]]; then exit 9; fi\n'
                'exec /usr/bin/sha256sum "$@"\n')
            wrapper.chmod(0o755)
            harness = (
                "set -euo pipefail\n"
                f"evidence={shlex.quote(str(evidence))}\n"
                "snapshot() { :; }\n"
                + match.group() +
                "trap finish EXIT\n"
                "printf 'start\\n' > \"$evidence/run-status.txt\"\n"
                "printf 'fixture\\n' > \"$evidence/fixture.txt\"\n"
                f"exit {original}\n"
            )
            result = subprocess.run(
                ["bash", "-c", harness], text=True, capture_output=True,
                env=dict(os.environ, PATH=str(base) + os.pathsep + os.environ["PATH"],
                         SIM_FAIL=fault), check=False)
            status = (evidence / "run-status.txt").read_text()
            seal = evidence / "SHA256SUMS"
            if fault == "":
                self.assertTrue(seal.is_file())
                self.assertTrue(subprocess.run(
                    ["/usr/bin/sha256sum", "-c", str(seal)], cwd=evidence,
                    capture_output=True, check=False).returncode == 0)
            else:
                self.assertIn("evidence seal failed", result.stderr)
            return result, status

    def test_success_requires_verified_seal(self) -> None:
        result, status = self.run_case(0)
        self.assertEqual(result.returncode, 0)
        self.assertIn("exit=0", status)

    def test_generation_failure_turns_success_into_failure(self) -> None:
        result, status = self.run_case(0, "generate")
        self.assertEqual(result.returncode, 1)
        self.assertIn("exit=0", status)

    def test_verification_failure_turns_success_into_failure(self) -> None:
        result, _ = self.run_case(0, "verify")
        self.assertEqual(result.returncode, 1)

    def test_original_failure_survives_seal_failure(self) -> None:
        result, status = self.run_case(37, "generate")
        self.assertEqual(result.returncode, 37)
        self.assertIn("exit=37", status)


if __name__ == "__main__":
    unittest.main()
