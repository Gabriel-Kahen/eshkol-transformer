"""Check that the P1 gate's isolated phases preserve the full-suite contract."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
P1_SCRIPT = PROJECT_ROOT / "scripts" / "test-p1.sh"


class P1PhaseDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = P1_SCRIPT.read_text(encoding="utf-8")

    def _run_dispatch(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        preflight = self.script.split("require_command ar", 1)[0]
        preflight, replacements = re.subn(
            r'^source .*common\.sh"$',
            "die() { printf '%s\\n' \"$*\" >&2; exit 1; }",
            preflight,
            count=1,
            flags=re.MULTILINE,
        )
        self.assertEqual(replacements, 1)
        harness = preflight + """
for candidate in identity runtime boundary; do
  if p1_phase_enabled "${candidate}"; then
    printf '%s\\n' "${candidate}"
  fi
done
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh") as script:
            script.write(harness)
            script.flush()
            return subprocess.run(
                ["bash", script.name, *arguments],
                cwd=PROJECT_ROOT,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def test_zero_args_and_explicit_all_select_every_phase(self) -> None:
        expected = "identity\nruntime\nboundary\n"
        for arguments in ((), ("--phase", "all")):
            with self.subTest(arguments=arguments):
                result = self._run_dispatch(*arguments)
                self.assertEqual((result.returncode, result.stdout), (0, expected))

    def test_each_explicit_phase_selects_only_itself(self) -> None:
        for phase in ("identity", "runtime", "boundary"):
            with self.subTest(phase=phase):
                result = self._run_dispatch("--phase", phase)
                self.assertEqual((result.returncode, result.stdout), (0, f"{phase}\n"))

    def test_unknown_or_malformed_arguments_fail(self) -> None:
        cases = (
            ("identity",),
            ("--phase",),
            ("--phase", "unknown"),
            ("--phase", "runtime", "extra"),
            ("--wrong", "runtime"),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = self._run_dispatch(*arguments)
                self.assertNotEqual(result.returncode, 0)

    def test_full_gate_assigns_each_contiguous_body_once(self) -> None:
        identity_guard = "if p1_phase_enabled identity; then"
        runtime_guard = "if p1_phase_enabled runtime; then"
        boundary_guard = "if p1_phase_enabled boundary; then"
        for guard in (identity_guard, runtime_guard, boundary_guard):
            self.assertEqual(self.script.count(guard), 1)

        shared, remainder = self.script.split(identity_guard, 1)
        identity, remainder = remainder.split(runtime_guard, 1)
        runtime, boundary = remainder.split(boundary_guard, 1)

        self.assertEqual(shared.count('"${PROJECT_ROOT}/scripts/build-p1-identity.sh"'), 2)
        self.assertEqual(shared.count('"${PROJECT_ROOT}/scripts/build-p1-package.sh"'), 2)
        self.assertIn("test-trusted-symbols.actual", shared)

        self.assertIn("test-p1-identity-cxx-public", identity)
        self.assertIn("test-p1-failpoints-sanitized", identity)
        self.assertEqual(identity.count("for run in 1 2; do"), 0)

        self.assertIn('compile_public_object "${p1_public_source}"', runtime)
        self.assertIn('compile_trusted_aot "${p1_registry_test_source}"', runtime)
        self.assertIn("P1 runtime gate wrote stderr", runtime)
        self.assertEqual(runtime.count("for run in 1 2; do"), 2)

        self.assertIn("negative_sources=(", boundary)
        self.assertIn("negative_guessed_private_native.esk", boundary)
        self.assertIn("P1 production Eshkol roots contain a forbidden", boundary)
        self.assertEqual(boundary.count("for run in 1 2; do"), 3)
        self.assertEqual(self.script.count("P1 PASS: E1B-integrated"), 1)
        self.assertIn('if [[ "${p1_phase}" == all ]]; then', boundary)


if __name__ == "__main__":
    unittest.main()
