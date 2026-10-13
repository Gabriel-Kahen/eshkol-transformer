"""Source-only negative checks for the versioned TR3 lease failure gate."""

import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
GATE = "scripts/test-tr3-lease-failures-successor.sh"


def run(*args, cwd):
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)


class FailureSuccessorGateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tr3-failure-source-")
        self.addCleanup(self.temp.cleanup)
        self.clone = pathlib.Path(self.temp.name) / "repo"
        result = run("git", "clone", "--shared", "--quiet", str(ROOT), str(self.clone), cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)

    def check_source(self, *extra):
        return run("bash", GATE, "--source-check", *extra, cwd=self.clone)

    def test_clean_exact_source(self):
        result = self.check_source()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("source-only preflight PASS", result.stdout)

    def test_manifest_and_gate_mutations_fail_closed(self):
        for name in (GATE, "tests/tr3_lease/failure_runtime_successor_v3.tsv"):
            with self.subTest(name=name):
                path = self.clone / name
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                self.assertNotEqual(self.check_source().returncode, 0)
                path.write_bytes(original)

    def test_stale_contract_hash_fails_closed(self):
        path = self.clone / "tests/tr3_lease/test_restore_successor_contract.py"
        path.write_bytes(path.read_bytes() + b"\n")
        result = self.check_source()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("successor-contract test changed", result.stderr)

    def test_clean_descendant_production_change_fails_closed(self):
        self.commit_mutation("native/tr3_lease_public_gates_extension.esk")
        result = self.check_source()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("frozen production lease sources changed", result.stderr)

    def commit_mutation(self, name):
        path = self.clone / name
        path.write_bytes(path.read_bytes() + b"\n")
        run("git", "add", str(path.relative_to(self.clone)), cwd=self.clone)
        result = run("git", "-c", "user.name=Codex Test", "-c",
                     "user.email=codex-test@example.invalid", "commit", "-qm",
                     "negative source mutation", cwd=self.clone)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_clean_descendant_fixture_generator_change_fails_closed(self):
        self.commit_mutation("tests/tr3b/prepare_fixture.py")
        result = self.check_source()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unreviewed checkout change", result.stderr)

    def test_clean_descendant_root_smoke_change_fails_closed(self):
        self.commit_mutation("tests/tr3_lease/root_smoke.esk")
        result = self.check_source()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unreviewed checkout change", result.stderr)

    def test_clean_descendant_d2_resource_change_fails_closed(self):
        self.commit_mutation("tests/d2/test_resources.py")
        result = self.check_source()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("reviewed lease production source or native witness changed", result.stderr)

    def test_probe_and_diagnostic_modes_rejected(self):
        for option in ("--allocation-probes", "--diagnostic-only", "--optimize"):
            with self.subTest(option=option):
                self.assertNotEqual(self.check_source(option).returncode, 0)


if __name__ == "__main__":
    unittest.main()
