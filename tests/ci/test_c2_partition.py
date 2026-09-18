from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRIVER = PROJECT_ROOT / "scripts" / "test-c2.sh"

GROUPS = (
    "c2-format",
    "c2-state",
    "c2-save",
    "c2-load",
    "c2-public",
    "c2-operational",
)
EXPECTED_COMMANDS = (
    "test-c2-format.sh",
    "test-c2-core.sh --core-only",
    "test-c2-checkpoint-inspect.sh",
    "test-c2-codec.sh",
    "test-c2-x1-canonical.sh",
    "test-c2-d2-cursor-pair.sh",
    "test-c2-persistence-policy.sh",
    "test-c2-training-state-owner.sh",
    "test-c2-model-encode.sh",
    "test-c2-o2-encode.sh",
    "test-c2-checkpoint-save.sh",
    "test-c2-checkpoint-load.sh --load-only",
    "test-c2-public.sh",
    "test-c2-checkpoint-operational.sh --operational-only",
)


class C2PartitionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.scripts = self.root / "scripts"
        self.scripts.mkdir()
        shutil.copy2(DRIVER, self.scripts / DRIVER.name)
        (self.scripts / "common.sh").write_text(
            "#!/usr/bin/bash\n"
            'PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." '
            '&& pwd)"\n',
            encoding="utf-8",
        )
        self.log = self.root / "commands.log"
        for command in EXPECTED_COMMANDS:
            name = command.split()[0]
            path = self.scripts / name
            path.write_text(
                "#!/usr/bin/bash\n"
                "set -euo pipefail\n"
                'printf "%s" "$(basename -- "$0")" >>"${C2_STUB_LOG}"\n'
                'for argument in "$@"; do\n'
                '  printf " %s" "${argument}" >>"${C2_STUB_LOG}"\n'
                "done\n"
                'printf "\\n" >>"${C2_STUB_LOG}"\n'
                'if [[ "${C2_FAIL_GATE:-}" == "$(basename -- "$0")" ]]; then\n'
                '  exit "${C2_FAIL_STATUS:-1}"\n'
                "fi\n",
                encoding="utf-8",
            )
            path.chmod(0o755)

    def _run(
        self, *arguments: str, fail_gate: str = "", fail_status: int = 1
    ) -> subprocess.CompletedProcess[str]:
        self.log.write_text("", encoding="utf-8")
        return subprocess.run(
            ["/usr/bin/bash", str(self.scripts / DRIVER.name), *arguments],
            cwd=self.root,
            env={
                **os.environ,
                "C2_STUB_LOG": str(self.log),
                "C2_FAIL_GATE": fail_gate,
                "C2_FAIL_STATUS": str(fail_status),
            },
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _commands(self) -> tuple[str, ...]:
        return tuple(self.log.read_text(encoding="utf-8").splitlines())

    def test_default_runs_each_transitive_leaf_once(self) -> None:
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._commands(), EXPECTED_COMMANDS)
        self.assertEqual(
            len({command.split()[0] for command in self._commands()}),
            len(self._commands()),
        )
        self.assertEqual(
            sum(command.startswith("test-c2-format.sh") for command in self._commands()),
            1,
        )
        self.assertEqual(result.stdout.count("C2 GROUP PASS:"), len(GROUPS))
        self.assertEqual(result.stdout.count("C2 GATE PASS:"), 1)

    def test_partition_union_is_command_equivalent_to_default(self) -> None:
        default = self._run()
        self.assertEqual(default.returncode, 0, default.stderr)
        default_commands = self._commands()

        partition_commands: list[str] = []
        for group in GROUPS:
            with self.subTest(group=group):
                result = self._run("--group", group)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("C2 GATE PASS:", result.stdout)
                partition_commands.extend(self._commands())

        self.assertEqual(tuple(partition_commands), default_commands)

    def test_composed_groups_use_canonical_order(self) -> None:
        result = self._run(
            "--group",
            "c2-operational",
            "--group",
            "c2-format",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self._commands(),
            EXPECTED_COMMANDS[:7] + EXPECTED_COMMANDS[-1:],
        )

    def test_invalid_selection_fails_before_running_a_gate(self) -> None:
        cases = (
            ("--group",),
            ("--unknown", "c2-format"),
            ("--group", "unknown"),
            ("c2-format",),
            ("--group", "c2-format", "extra"),
            ("--group", "c2-format", "--group", "c2-format"),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = self._run(*arguments)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(self._commands(), ())

    def test_gate_failure_stops_group_and_propagates_exact_status(self) -> None:
        result = self._run(
            "--group",
            "c2-state",
            fail_gate="test-c2-model-encode.sh",
            fail_status=37,
        )
        self.assertEqual(result.returncode, 37)
        self.assertEqual(
            self._commands(),
            (
                "test-c2-training-state-owner.sh",
                "test-c2-model-encode.sh",
            ),
        )
        self.assertNotIn("C2 GROUP PASS:", result.stdout)
        self.assertNotIn("C2 GATE PASS:", result.stdout)

    def test_nested_regression_options_preserve_standalone_defaults(self) -> None:
        contracts = {
            "test-c2-core.sh": (
                "core_only=0",
                "core_only=1",
                "--core-only",
                'if [[ "${core_only}" == 0 ]]; then',
                "test-c2-format.sh",
            ),
            "test-c2-checkpoint-load.sh": (
                "load_only=0",
                "load_only=1",
                "--load-only",
                'if [[ "${load_only}" == 0 ]]; then',
                "C2 PRIVATE CHECKPOINT LOAD FOCUSED PASS:",
                "test-c2-format.sh test-c2-core.sh test-c2-checkpoint-inspect.sh",
                "test-c2-o2-encode.sh test-c2-training-state-owner.sh",
            ),
            "test-c2-checkpoint-operational.sh": (
                "operational_only=0",
                "operational_only=1",
                "--operational-only",
                'if [[ "${operational_only}" == 0 ]]; then',
                "C2 CHECKPOINT OPERATIONAL EVIDENCE FOCUSED PASS:",
                "test-c2-format.sh test-c2-core.sh test-c2-checkpoint-save.sh",
                "test-c2-checkpoint-load.sh",
            ),
        }
        for name, fragments in contracts.items():
            with self.subTest(script=name):
                source = (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")
                for fragment in fragments:
                    self.assertIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
