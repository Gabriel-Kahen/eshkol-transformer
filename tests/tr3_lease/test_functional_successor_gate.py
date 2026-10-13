"""Source-only checks for the versioned TR3 functional lease gate."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE = "scripts/test-tr3-lease-functional-successor.sh"
MANIFEST = "tests/tr3_lease/functional_successor_candidate.tsv"
BASE = "1a58fc8c750dff107f0c4cd36d0022ceeca8abd4"
TREE = "c528ee6db2088236d6e03fad1ec625e2cbab14af"
REVIEWED_GATE = BASE


def run(cwd: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True, check=False)


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
        self.assertIn("reviewed_gate_commit=" + REVIEWED_GATE, gate)
        self.assertIn("TR3_FUNCTIONAL_REVIEWED_HEAD", gate)
        self.assertNotIn("repair_anchor=", gate)
        self.assertNotIn("--diagnostic-only", gate)
        self.assertNotIn("--allocation-probes", gate)
        fixture = (ROOT / "tests/tr3_lease/lease_runtime.esk").read_text()
        for case in (
            "enrollment roots exact T2 shell, entry, and raw core",
            "same-core-t1-shell",
            "equal fingerprint from foreign registered core rejects",
            "mutated registry core rejects",
            "private pre-handoff rejects non-idle phase",
            "released",
            "copied witness self rejects",
            "original witness survives negative cases",
        ):
            self.assertIn(case, fixture)
        marker = "expected_gate_self_sha256="
        expected = next(line[len(marker):] for line in gate.splitlines() if line.startswith(marker))
        normalized = gate.replace(marker + expected, marker + "__SELF__", 1)
        self.assertEqual(hashlib.sha256(normalized.encode()).hexdigest(), expected)

    def test_source_preflight_accepts_only_the_externally_pinned_head(self) -> None:
        reviewed_head = run(ROOT, "git", "rev-parse", "HEAD").stdout.strip()
        self.assertRegex(reviewed_head, r"^[0-9a-f]{40}$")
        with tempfile.TemporaryDirectory(prefix="tr3-functional-gate-") as directory:
            checkout = Path(directory) / "checkout"
            cloned = run(ROOT, "git", "clone", "--quiet", "--shared", str(ROOT), str(checkout))
            self.assertEqual(cloned.returncode, 0, cloned.stderr)
            for args in (("user.name", "Source Test"), ("user.email", "source@test.invalid")):
                self.assertEqual(run(checkout, "git", "config", *args).returncode, 0)

            def preflight(pin: str | None = reviewed_head) -> subprocess.CompletedProcess[str]:
                env = os.environ.copy()
                env.pop("TR3_FUNCTIONAL_REVIEWED_HEAD", None)
                if pin is not None:
                    env["TR3_FUNCTIONAL_REVIEWED_HEAD"] = pin
                return run(checkout, "bash", GATE, "--source-preflight", env=env)

            self.assertIn("externally reviewed full commit ID", preflight(None).stderr)

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
            self.assertIn("externally reviewed functional successor head", preflight().stderr)

            self.assertEqual(run(checkout, "git", "reset", "--hard", "HEAD^").returncode, 0)
            candidate = checkout / MANIFEST
            candidate.write_text(candidate.read_text().replace("format_version\t3", "format_version\t4"))
            self.assertEqual(run(checkout, "git", "add", MANIFEST).returncode, 0)
            self.assertEqual(run(checkout, "git", "commit", "-qm", "negative manifest version").returncode, 0)
            self.assertIn("candidate manifest changed", preflight().stderr)

            self.assertEqual(run(checkout, "git", "reset", "--hard", "HEAD^").returncode, 0)
            for path in (
                "scripts/common.sh",
                "tests/tr3_lease/root_smoke.esk",
                "tests/tr3b/prepare_fixture.py",
                "tests/d2/test_resources.py",
                "tests/d1/reference_format.py",
                "tests/d2/reference.py",
                "tests/__init__.py",
                "tests/tr3b/__init__.py",
                "tests/d2/__init__.py",
                "tests/tr3_lease/__init__.py",
            ):
                source = checkout / path
                source.write_bytes(source.read_bytes() + b"\n")
                self.assertEqual(run(checkout, "git", "add", path).returncode, 0)
                self.assertEqual(run(checkout, "git", "commit", "-qm", "negative live input").returncode, 0)
                failed = preflight()
                self.assertNotEqual(failed.returncode, 0, path)
                self.assertIn(
                    "common shell input changed" if path == "scripts/common.sh" else
                    "externally reviewed functional successor head",
                    failed.stderr, path,
                )
                self.assertEqual(run(checkout, "git", "reset", "--hard", "HEAD^").returncode, 0)

            injected = checkout / "sitecustomize.py"
            injected.write_text("# Python startup hook\n")
            self.assertEqual(run(checkout, "git", "add", "sitecustomize.py").returncode, 0)
            self.assertEqual(run(checkout, "git", "commit", "-qm", "negative import hook").returncode, 0)
            self.assertIn("externally reviewed functional successor head", preflight().stderr)
            self.assertEqual(run(checkout, "git", "reset", "--hard", "HEAD^").returncode, 0)

            note = checkout / "docs/tr3-functional-source-test.txt"
            note.write_text("unrelated documentation\n")
            self.assertEqual(run(checkout, "git", "add", str(note.relative_to(checkout))).returncode, 0)
            self.assertEqual(run(checkout, "git", "commit", "-qm", "allowed documentation").returncode, 0)
            self.assertIn("externally reviewed functional successor head", preflight().stderr)

            # A committed descendant must not authorize its own gate edit by
            # updating the self hash stored inside that same gate.
            gate_path = checkout / GATE
            changed_gate = gate_path.read_text() + "\n# committed descendant tamper\n"
            marker = "expected_gate_self_sha256="
            old_hash = next(
                line[len(marker):]
                for line in changed_gate.splitlines()
                if line.startswith(marker)
            )
            normalized = changed_gate.replace(marker + old_hash, marker + "__SELF__", 1)
            new_hash = hashlib.sha256(normalized.encode()).hexdigest()
            gate_path.write_text(changed_gate.replace(marker + old_hash, marker + new_hash, 1))
            self.assertEqual(run(checkout, "git", "add", GATE).returncode, 0)
            self.assertEqual(run(checkout, "git", "commit", "-qm", "negative gate self rehash").returncode, 0)
            self.assertIn("externally reviewed functional successor head", preflight().stderr)

            # Reproduce the original escape: a direct sibling of the first
            # reviewed gate contains a rehashed copy of this gate. The pin is
            # fixed before switching to the sibling, as the root caller must do.
            current_files = {
                path: (checkout / path).read_bytes()
                for path in (GATE, MANIFEST, "tests/tr3_lease/test_functional_successor_gate.py")
            }
            self.assertEqual(
                run(checkout, "git", "checkout", "--detach", "--quiet", REVIEWED_GATE).returncode,
                0,
            )
            for path, contents in current_files.items():
                (checkout / path).write_bytes(contents)
            sibling_gate = checkout / GATE
            sibling_text = sibling_gate.read_text() + "\n# sibling gate tamper\n"
            old_hash = next(
                line[len(marker):]
                for line in sibling_text.splitlines()
                if line.startswith(marker)
            )
            sibling_hash = hashlib.sha256(
                sibling_text.replace(marker + old_hash, marker + "__SELF__", 1).encode()
            ).hexdigest()
            sibling_gate.write_text(
                sibling_text.replace(marker + old_hash, marker + sibling_hash, 1)
            )
            self.assertEqual(run(checkout, "git", "add", GATE, MANIFEST,
                                 "tests/tr3_lease/test_functional_successor_gate.py").returncode, 0)
            self.assertEqual(run(checkout, "git", "commit", "-qm", "sibling self rehash").returncode, 0)
            sibling_head = run(checkout, "git", "rev-parse", "HEAD").stdout.strip()
            self.assertNotEqual(sibling_head, reviewed_head)
            self.assertEqual(run(checkout, "git", "rev-parse", "HEAD^").stdout.strip(), REVIEWED_GATE)
            self.assertIn("externally reviewed functional successor head", preflight().stderr)


if __name__ == "__main__":
    unittest.main()
