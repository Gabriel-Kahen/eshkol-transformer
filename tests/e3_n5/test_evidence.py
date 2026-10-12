from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tests.e3_n5.evidence import entries, seal, sha256


class N5EvidenceTests(unittest.TestCase):
    def test_gate_preflight_failure_keeps_sealed_status(self) -> None:
        project = Path(__file__).resolve().parents[2]
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project,
                                         text=True).strip()
        tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"],
                                       cwd=project, text=True).strip()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "failed-gate"
            environment = dict(os.environ)
            environment.update({
                "ESHKOL_ALLOW_UNSUPPORTED_HOST": "0",
                "N5_SOURCE_COMMIT": commit, "N5_SOURCE_TREE": tree,
                "N5_LOCK_SHA256": "0" * 64,
                "N5_PROVENANCE_SHA256": "0" * 64,
                "N5_RUNNER_SHA256": "0" * 64,
                "N5_CC_SHA256": "0" * 64,
                "N5_CXX_SHA256": "0" * 64,
                "N5_CORPUS_INVENTORY_SHA256": "0" * 64,
            })
            result = subprocess.run(["bash", str(project / "scripts/test-e3-n5-private.sh"),
                                     str(output)], cwd=project, env=environment,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((output / "stage.txt").read_text(), "pin-preflight\n")
            self.assertEqual((output / "exit.status").read_text(),
                             f"{result.returncode}\n")
            for copy in ("a", "b"):
                self.assertEqual((output / f"postflight-fixture-{copy}.status").read_text(),
                                 "1\n")
            verified = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=output,
                                      capture_output=True, text=True)
            self.assertEqual(verified.returncode, 0, verified.stderr)

    def test_failure_is_sealed_and_symlink_is_listed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "compile.stderr").write_text("failed\n")
            (root / "link").symlink_to("compile.stderr")
            self.assertEqual(seal(root, 42, "compile-a"), 42)
            self.assertEqual((root / "exit.status").read_text(), "42\n")
            self.assertIn('"type":"symlink"', (root / "INVENTORY.jsonl").read_text())
            self.assertIn('"target":"compile.stderr"', (root / "INVENTORY.jsonl").read_text())
            self.assertIn("compile.stderr", (root / "SHA256SUMS").read_text())
            verified = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=root,
                                      capture_output=True, text=True)
            self.assertEqual(verified.returncode, 0, verified.stderr)

    def test_symlink_rejects_nominal_success_and_regular_files_are_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a").write_bytes(b"a")
            (root / "nested").mkdir()
            (root / "nested/b").write_bytes(b"b")
            self.assertEqual(seal(root, 0, "complete"), 0)
            self.assertEqual((root / "exit.status").read_text(), "0\n")
            (root / "nested/link").symlink_to("../a")
            self.assertEqual(seal(root, 0, "complete"), 1)
            self.assertEqual((root / "exit.status").read_text(), "1\n")
            self.assertEqual(sha256(root / "a"), hashlib.sha256(b"a").hexdigest())
            self.assertEqual(len([row for row in entries(root) if row["type"] == "symlink"]), 1)

    def test_reserved_seal_symlink_is_replaced_without_following(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "outside").write_text("untouched\n")
            (root / "SHA256SUMS").symlink_to("outside")
            self.assertEqual(seal(root, 0, "complete"), 1)
            self.assertEqual((root / "outside").read_text(), "untouched\n")
            self.assertFalse((root / "SHA256SUMS").is_symlink())
            self.assertIn('"path":"preseal/SHA256SUMS"',
                          (root / "INVENTORY.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
