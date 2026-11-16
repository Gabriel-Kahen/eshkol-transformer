"""Keep the DATA4 diagnostic separate from ordinary CI admission."""

from pathlib import Path
import unittest

from tests.ci.topology import field, jobs


ROOT = Path(__file__).resolve().parents[2]


class Data4DiagnosticWorkflowTests(unittest.TestCase):
    def test_dispatch_is_opt_in_and_does_not_claim_full_coverage(self) -> None:
        ci = (ROOT / ".github/workflows/ci.yml").read_text()
        self.assertIn("      data4_installed_diagnostic:\n", ci)
        self.assertIn("        default: false\n        type: boolean\n", ci)
        self.assertIn("      data4_expected_head:\n", ci)
        self.assertIn("      data4_expected_tree:\n", ci)
        entries = jobs(ci)
        self.assertEqual(field(entries["data4-diagnostic"], 4, "if"),
                         "${{ github.event_name == 'workflow_dispatch' && inputs.data4_installed_diagnostic }}")
        self.assertIn("      expected_head: ${{ inputs.data4_expected_head }}\n",
                      entries["data4-diagnostic"])
        self.assertIn("      expected_tree: ${{ inputs.data4_expected_tree }}\n",
                      entries["data4-diagnostic"])
        self.assertIn("!inputs.data4_installed_diagnostic", entries["topology"])
        self.assertIn("!inputs.data4_installed_diagnostic", entries["f0-linux"])

    def test_called_workflow_pins_identity_and_seals_evidence(self) -> None:
        diagnostic = (ROOT / ".github/workflows/data4-installed-diagnostic.yml").read_text()
        self.assertIn("  workflow_call:\n", diagnostic)
        self.assertNotIn("  workflow_dispatch:\n", diagnostic)
        for needle in (
            "refs/heads/codex/et-data4-installed-20261116",
            '[[ "$GITHUB_SHA" == "$EXPECTED_HEAD" ]]',
            "git rev-parse 'HEAD^{tree}'",
            "test \"$(llvm-config-21 --version)\" = '21.1.8'",
            "make toolchain",
            'test "$memory_kib" -ge $((8 * 1024 * 1024))',
            'test "$disk_bytes" -ge $((6 * 1024 * 1024 * 1024))',
            "scripts/test-cli3-data4-installed.sh",
            "if: always()",
            "sha256sum -c SHA256SUMS",
            "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, diagnostic)

    def test_focused_gate_configures_and_uses_disk_backed_temp(self) -> None:
        gate = (ROOT / "scripts/test-cli3-data4-installed.sh").read_text()
        self.assertIn('export TMPDIR="${evidence}"', gate)
        self.assertLess(gate.index("step prerequisite-plan"),
                        gate.index("step configure /usr/bin/bash"))
        self.assertLess(gate.index("step configure /usr/bin/bash"),
                        gate.index("step data4-source /usr/bin/bash"))
        self.assertLess(gate.index("step data4-source /usr/bin/bash"),
                        gate.index("step build-cli3 env"))


if __name__ == "__main__":
    unittest.main()
