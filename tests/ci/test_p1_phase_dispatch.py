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
for candidate in public state registry; do
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
        expected = "public\nstate\nregistry\n"
        for arguments in ((), ("--phase", "all")):
            with self.subTest(arguments=arguments):
                result = self._run_dispatch(*arguments)
                self.assertEqual((result.returncode, result.stdout), (0, expected))

    def test_each_explicit_phase_selects_only_itself(self) -> None:
        for phase in ("public", "state", "registry"):
            with self.subTest(phase=phase):
                result = self._run_dispatch("--phase", phase)
                self.assertEqual((result.returncode, result.stdout), (0, f"{phase}\n"))

    def test_unknown_or_malformed_arguments_fail(self) -> None:
        cases = (
            ("public",),
            ("--phase",),
            ("--phase", "unknown"),
            ("--phase", "state", "extra"),
            ("--wrong", "registry"),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = self._run_dispatch(*arguments)
                self.assertNotEqual(result.returncode, 0)

    def test_full_gate_preserves_each_artifact_family(self) -> None:
        self.assertIn(
            "if p1_phase_enabled public; then\n"
            "p1_phase_mark BEGIN public-package",
            self.script,
        )
        production = self.script.split("p1_phase_mark BEGIN public-package", 1)[1]
        production, shared_and_remainder = production.split(
            "p1_phase_mark PASS public-package", 1
        )
        shared, remainder = shared_and_remainder.split(
            "p1_phase_mark PASS shared", 1
        )
        self.assertEqual(
            production.count('"${PROJECT_ROOT}/scripts/build-p1-identity.sh"'), 2
        )
        self.assertEqual(
            production.count('"${PROJECT_ROOT}/scripts/build-p1-package.sh"'), 2
        )
        self.assertIn("test-trusted-symbols.actual", shared)
        self.assertNotIn("build-p1-package.sh", shared)
        self.assertIn(
            "if p1_phase_enabled public; then\n"
            'if nm -a "${p1_public_archive}" "${p1_trusted_archive}"',
            shared,
        )

        identity = remainder.split("p1_phase_mark PASS public-identity", 1)[0]
        self.assertIn("if p1_phase_enabled public; then", identity)
        self.assertIn("test-p1-identity-cxx-public", identity)
        self.assertIn("test-p1-failpoints-sanitized", identity)

        runtime, boundary = remainder.split(
            "p1_phase_mark BEGIN public-boundary", 1
        )
        self.assertIn(
            'if p1_phase_enabled public; then\n'
            '  compile_public_object "${p1_public_source}"',
            runtime,
        )
        self.assertIn(
            'if p1_phase_enabled state; then\n'
            '  compile_trusted_object "${p1_test_source}"',
            runtime,
        )
        self.assertIn(
            'if p1_phase_enabled registry; then\n'
            '  compile_trusted_aot "${p1_registry_test_source}"',
            runtime,
        )
        self.assertIn("P1 public runtime gate wrote stderr", runtime)
        self.assertIn("P1 state runtime gate wrote stderr", runtime)
        self.assertIn("P1 registry runtime gate wrote stderr", runtime)

        self.assertIn("negative_sources=(", boundary)
        self.assertIn("negative_guessed_private_native.esk", boundary)
        self.assertIn("P1 production Eshkol roots contain a forbidden", boundary)
        self.assertIn("p1_phase_mark PASS public", boundary)

        self.assertEqual(self.script.count("for run in 1 2; do"), 5)
        self.assertEqual(self.script.count("compile_public_object "), 4)
        self.assertEqual(self.script.count("compile_trusted_object "), 1)
        self.assertEqual(self.script.count("compile_public_aot "), 3)
        self.assertEqual(self.script.count("compile_trusted_aot "), 2)
        self.assertEqual(self.script.count("P1 PASS: E1B-integrated"), 1)
        self.assertIn('if [[ "${p1_phase}" == all ]]; then', boundary)


if __name__ == "__main__":
    unittest.main()
