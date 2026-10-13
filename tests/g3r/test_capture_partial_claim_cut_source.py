"""Source gate for the bounded current-main partial C2 claim fixture."""

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tests.c2.test_handler_order import parse
from tests.g3r import prepare_capture_partial_claim_cut as preparation
from tests.g3r.prepare_capture_partial_claim_cut import (
    ROOT, BASE, TREE, EXTRA, BYTE_FINGERPRINT, git, verify, verify_file, prepare,
)


class PartialClaimSource(unittest.TestCase):
    def test_exact_approved_closure_and_runtime_hooks(self):
        self.assertEqual(len(verify(git("rev-parse", "HEAD"))), 96)
        root = (ROOT / EXTRA[2]).read_text()
        runtime = (ROOT / EXTRA[3]).read_text()
        self.assertTrue(parse(root))
        self.assertTrue(parse(runtime))
        for marker in (
            "(g3r-partial-real-p1-claim p1 token)",
            "(g3r-save-o2-require-held-internal shell)",
            "(g3r-partial-real-p1-rollback p1 token)",
            "(g3r-partial-real-compose request)",
            "(g3r-partial-real-abandon shell)",
            "(g3r-partial-real-o2-release shell)",
            "(g3r-partial-real-p1-release p1)",
            "(g3r-partial-restore!)",
        ):
            self.assertIn(marker, root + runtime)
        self.assertIn("(eq? (vector-ref request 5) #f)", root)
        self.assertIn("(images=? moments-before (moment-images) 28)", runtime)
        self.assertIn("(bytes=? history-before (history-image))", runtime)
        self.assertIn("(c2-public-trainer-state-release!", runtime)
        for mode in ("hook-absent", "wrong-token", "wrong-shell",
                     "premature-o2", "rollback-uncertain",
                     "abandon-uncertain"):
            self.assertIn(f'"{mode}"', root + runtime)

    def test_changed_closure_and_old_source_fail_closed(self):
        with patch.dict(preparation.PINS, {preparation.CLOSURE: "0" * 64}):
            with self.assertRaisesRegex(ValueError, "pinned source changed"):
                verify(git("rev-parse", "HEAD"))
        with patch.object(preparation, "BASE",
                          "ac0ac36e5da6ec4e7bc19c4f4dc91c78cbfc9f5e"):
            with self.assertRaisesRegex(ValueError, "approved commit/tree pair"):
                verify(git("rev-parse", "HEAD"))
        with self.assertRaisesRegex(ValueError, "fixture HEAD differs"):
            verify("0" * 40)
        held = subprocess.run(
            ["bash", "scripts/test-g3r-capture-partial-claim.sh", "--run"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(held.returncode, 74)
        self.assertIn("HOLD", held.stderr)

    def test_staged_files_are_pinned_and_read_only(self):
        with tempfile.TemporaryDirectory(prefix="g3r-partial-source-") as scratch:
            output = Path(scratch) / "prepared"
            prepare(output, git("rev-parse", "HEAD"))
            contents = (output / "PREPARED.tsv").read_text()
            self.assertIn(f"approved_commit\t{BASE}\n", contents)
            self.assertIn(f"approved_tree\t{TREE}\n", contents)
            self.assertIn("compiled gate HOLD", contents)
            self.assertTrue((output / "corpus/manifest.etm").is_file())
            self.assertIn(BYTE_FINGERPRINT.encode(),
                          (output / "corpus/manifest.etm").read_bytes())
            for name in EXTRA:
                path = output / "source" / name
                self.assertEqual(path.read_bytes(), (ROOT / name).read_bytes())
                self.assertEqual(path.stat().st_mode & 0o222, 0)
            with self.assertRaises(ValueError):
                prepare(output, git("rev-parse", "HEAD"))

    def test_isolated_fixture_and_same_line_pin_drift_rejected(self):
        head = git("rev-parse", "HEAD")
        with tempfile.TemporaryDirectory(prefix="g3r-partial-drift-") as scratch:
            for name, suffix in (
                (EXTRA[3], b"# appended 34-byte fixture drift!\n"),
                ("toolchain/eshkol.lock", b"# same expected line; drift\n"),
                ("tests/tr3_lease/runtime_candidate.tsv",
                 b"# same expected line; drift\n"),
                ("tests/g3r/prepare_capture_partial_claim_cut.py",
                 b"# dirty preparer\n"),
                ("tests/g3r/test_capture_partial_claim_cut_source.py",
                 b"# dirty source test\n"),
                ("scripts/test-g3r-capture-partial-claim.sh",
                 b"# dirty launcher\n"),
            ):
                with self.subTest(name=name):
                    if name == EXTRA[3]:
                        self.assertEqual(len(suffix), 34)
                    candidate = Path(scratch) / Path(name).name
                    candidate.write_bytes((ROOT / name).read_bytes() + suffix)
                    with self.assertRaises(ValueError):
                        verify_file(name, head, candidate)

    def test_dirty_worktree_is_rejected(self):
        with tempfile.NamedTemporaryFile(
            dir=ROOT / "tests/g3r", prefix=".g3r-partial-dirty-"
        ):
            with self.assertRaisesRegex(ValueError, "worktree is not clean"):
                verify(git("rev-parse", "HEAD"))


if __name__ == "__main__":
    unittest.main()
