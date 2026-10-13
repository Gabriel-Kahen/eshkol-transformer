"""Source-only checks for the versioned TR3 functional lease gate."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE = "scripts/test-tr3-lease-functional-successor.sh"
MANIFEST = "tests/tr3_lease/functional_successor_candidate.tsv"
BASE = "41cbf6c310106c5d37cada181ecd6e8b0f91f84e"
TREE = "50aef09425000e1f222858ce5af1189839583c2c"


def run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)


class FunctionalSuccessorGate(unittest.TestCase):
    def test_versioned_identity_and_runtime_contract(self) -> None:
        entries = [line.split("\t", 1) for line in (ROOT / MANIFEST).read_text().splitlines()]
        self.assertTrue(all(len(entry) == 2 for entry in entries))
        self.assertEqual(len(entries), len(dict(entries)))
        manifest = dict(entries)
        self.assertEqual(manifest["format_version"], "3")
        self.assertEqual(manifest["trainer_source_commit"], BASE)
        self.assertEqual(manifest["trainer_source_tree"], TREE)
        self.assertEqual(
            manifest["successor_contract_sha256"],
            hashlib.sha256(
                (ROOT / "tests/tr3_lease/test_restore_successor_contract.py").read_bytes()
            ).hexdigest(),
        )
        gate = (ROOT / GATE).read_text()
        self.assertIn("--strict-types --no-stdlib -O 0", gate)
        self.assertIn("horizon=1024 loop_arena_delta=0", gate)
        self.assertIn("horizon=8192 loop_arena_delta=0", gate)
        self.assertIn("ESHKOL_ARENA_POISON=1", gate)
        self.assertIn("--network none", gate)
        self.assertNotIn("--diagnostic-only", gate)
        self.assertNotIn("--allocation-probes", gate)
        fixture = (ROOT / "tests/tr3_lease/lease_runtime.esk").read_text()
        for case in (
            "same-core-t1-shell",
            "released",
            "copied witness self rejects",
            "original witness survives negative cases",
        ):
            self.assertIn(case, fixture)
        marker = "expected_gate_self_sha256="
        expected = next(line[len(marker):] for line in gate.splitlines() if line.startswith(marker))
        normalized = gate.replace(marker + expected, marker + "__SELF__", 1)
        self.assertEqual(hashlib.sha256(normalized.encode()).hexdigest(), expected)

    def test_source_preflight_accepts_only_clean_unchanged_descendants(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tr3-functional-gate-") as directory:
            checkout = Path(directory) / "checkout"
            cloned = run(ROOT, "git", "clone", "--quiet", "--shared", str(ROOT), str(checkout))
            self.assertEqual(cloned.returncode, 0, cloned.stderr)
            for args in (("user.name", "Source Test"), ("user.email", "source@test.invalid")):
                self.assertEqual(run(checkout, "git", "config", *args).returncode, 0)

            def preflight() -> subprocess.CompletedProcess[str]:
                return run(checkout, "bash", GATE, "--source-preflight")

            passed = preflight()
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertIn("source preflight only", passed.stdout)
            self.assertNotIn("SUCCESSOR PASS", passed.stdout)

            untracked = checkout / "untracked-source-test"
            untracked.write_text("untracked\n")
            self.assertNotEqual(preflight().returncode, 0)
            untracked.unlink()

            production = checkout / "native/tr3_lease_core_extension.esk"
            production.write_bytes(production.read_bytes() + b"\n")
            self.assertNotEqual(preflight().returncode, 0)
            self.assertEqual(run(checkout, "git", "add", "native/tr3_lease_core_extension.esk").returncode, 0)
            self.assertEqual(run(checkout, "git", "commit", "-qm", "negative production change").returncode, 0)
            self.assertIn("production source differs", preflight().stderr)

            self.assertEqual(run(checkout, "git", "reset", "--hard", "HEAD^").returncode, 0)
            candidate = checkout / MANIFEST
            candidate.write_text(candidate.read_text().replace("format_version\t3", "format_version\t4"))
            self.assertEqual(run(checkout, "git", "add", MANIFEST).returncode, 0)
            self.assertEqual(run(checkout, "git", "commit", "-qm", "negative manifest version").returncode, 0)
            self.assertIn("unsupported functional successor manifest format", preflight().stderr)

            self.assertEqual(run(checkout, "git", "reset", "--hard", "HEAD^").returncode, 0)
            note = checkout / "docs/tr3-functional-source-test.txt"
            note.write_text("unrelated documentation\n")
            self.assertEqual(run(checkout, "git", "add", str(note.relative_to(checkout))).returncode, 0)
            self.assertEqual(run(checkout, "git", "commit", "-qm", "allowed documentation").returncode, 0)
            self.assertEqual(preflight().returncode, 0)


if __name__ == "__main__":
    unittest.main()
