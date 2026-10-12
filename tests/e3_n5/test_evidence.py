from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tests.e3_n5.evidence import (COLLISION_FILE, COLLISION_STATUS, SEAL_FILES,
                                  entries, seal, sha256, verify_inventory)


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
            self.assertTrue(verify_inventory(output))

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

    def test_successful_first_seal_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "nested/result.txt").write_text("pass\n")
            (root / "nested.txt").write_text("sibling\n")
            self.assertEqual(seal(root, 0, "complete"), 0)
            self.assertEqual((root / "exit.status").read_text(), "0\n")
            self.assertFalse((root / COLLISION_FILE).exists())
            verified = subprocess.run(["sha256sum", "-c", "--status", "SHA256SUMS"],
                                      cwd=root)
            self.assertEqual(verified.returncode, 0)
            self.assertTrue(verify_inventory(root))

    def test_postseal_inventory_rejects_new_file_and_symlink(self) -> None:
        for kind in ("file", "symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "result.txt").write_text("pass\n")
                self.assertEqual(seal(root, 0, "complete"), 0)
                self.assertTrue(verify_inventory(root))
                if kind == "file":
                    (root / "unlisted").write_text("new\n")
                else:
                    (root / "unlisted").symlink_to("result.txt")
                self.assertFalse(verify_inventory(root))
                verified = subprocess.run(["sha256sum", "-c", "--strict", "--status",
                                           "SHA256SUMS"], cwd=root)
                self.assertEqual(verified.returncode, 0)

    def test_strict_postseal_check_rejects_malformed_checksum_line(self) -> None:
        project = Path(__file__).resolve().parents[2]
        self.assertIn("sha256sum -c --strict --status SHA256SUMS",
                      (project / "scripts/test-e3-n5-private.sh").read_text())
        self.assertIn("python3 -m tests.e3_n5.evidence verify-inventory .",
                      (project / "scripts/test-e3-n5-private.sh").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "result.txt").write_text("pass\n")
            self.assertEqual(seal(root, 0, "complete"), 0)
            with (root / "SHA256SUMS").open("a") as stream:
                stream.write("malformed checksum row\n")
            verified = subprocess.run(["sha256sum", "-c", "--strict", "--status",
                                       "SHA256SUMS"], cwd=root, capture_output=True)
            self.assertNotEqual(verified.returncode, 0)

    def test_symlink_rejects_nominal_success_and_regular_files_are_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a").write_bytes(b"a")
            (root / "nested").mkdir()
            (root / "nested/b").write_bytes(b"b")
            (root / "nested/link").symlink_to("../a")
            self.assertEqual(seal(root, 0, "complete"), 1)
            self.assertEqual((root / "exit.status").read_text(), "1\n")
            self.assertEqual(sha256(root / "a"), hashlib.sha256(b"a").hexdigest())
            self.assertEqual(len([row for row in entries(root) if row["type"] == "symlink"]), 1)

    def test_resealing_failed_evidence_preserves_every_reserved_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "compile.stderr").write_text("failure\n")
            self.assertEqual(seal(root, 42, "compile-a"), 42)
            original = {name: (root / name).read_bytes() for name in SEAL_FILES}
            self.assertEqual(seal(root, 0, "complete"), COLLISION_STATUS)
            self.assertEqual({name: (root / name).read_bytes() for name in SEAL_FILES},
                             original)
            report = json.loads((root / COLLISION_FILE).read_text())
            self.assertEqual(report["reserved_collisions"], list(SEAL_FILES))
            self.assertEqual(report["command_status"], 0)
            self.assertEqual(report["stage"], "complete")
            self.assertEqual(seal(root, 2, "later"), COLLISION_STATUS)
            self.assertEqual(json.loads((root / COLLISION_FILE).read_text()), report)

    def test_any_reserved_regular_file_blocks_sealing(self) -> None:
        for name in SEAL_FILES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / name).write_bytes(b"preexisting evidence\n")
                self.assertEqual(seal(root, 0, "complete"), COLLISION_STATUS)
                self.assertEqual((root / name).read_bytes(), b"preexisting evidence\n")
                self.assertEqual([candidate for candidate in SEAL_FILES
                                  if os.path.lexists(root / candidate)], [name])
                report = json.loads((root / COLLISION_FILE).read_text())
                self.assertEqual(report["reserved_collisions"], [name])

    def test_reserved_symlink_and_dangling_symlink_block_sealing(self) -> None:
        for target in ("outside", "missing"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "outside").write_text("untouched\n")
                (root / "SHA256SUMS").symlink_to(target)
                self.assertEqual(seal(root, 0, "complete"), COLLISION_STATUS)
                self.assertTrue((root / "SHA256SUMS").is_symlink())
                self.assertEqual(os.readlink(root / "SHA256SUMS"), target)
                self.assertEqual((root / "outside").read_text(), "untouched\n")
                self.assertFalse((root / "INVENTORY.jsonl").exists())

    def test_existing_collision_report_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / COLLISION_FILE).write_text("original\n")
            self.assertEqual(seal(root, 0, "complete"), COLLISION_STATUS)
            self.assertEqual((root / COLLISION_FILE).read_text(), "original\n")
            self.assertFalse(any(os.path.lexists(root / name) for name in SEAL_FILES))


if __name__ == "__main__":
    unittest.main()
