"""Bounded source contract for the private two-layer release path."""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class PublishedReleaseSource(unittest.TestCase):
    def test_admission_and_withdrawal_order(self):
        root = (ROOT / "native/t1_wave1_root.esk").read_text()
        seam = (ROOT / "native/t1_published_release_private.esk").read_text()
        self.assertIn("(admission (vector #f 'pending))", root)
        self.assertIn("(vector-set! admission 1 'live)", root)
        self.assertIn("(eq? (vector-ref (car remaining) 1) 'live)", root)
        self.assertLess(seam.index("(t1-wave1-find candidate"),
                        seam.index("(t1-private-shell-release-preflight candidate)"))
        self.assertLess(seam.index("(t1-private-shell-release-preflight candidate)"),
                        seam.index("(vector-set! entry 1 'releasing)"))
        self.assertLess(seam.index("(vector-set! entry 1 'releasing)"),
                        seam.index("(t1-private-shell-release candidate)"))
        self.assertLess(seam.index("(t1-private-shell-release candidate)"),
                        seam.index("(vector-set! entry 1 'dead)"))
        self.assertIn("(eq? (vector-ref entry 1) 'dead)", seam)

    def test_native_teardown_and_dead_guards(self):
        source = (ROOT / "native/t1_i64_shell.c").read_text()
        release = source[source.index("int64_t et_t1_i64_shell_private_release_v1") :]
        self.assertLess(release.index("et_i64_tensor_borrow_end_v1"),
                        release.index("et_i64_tensor_destroy_v1"))
        self.assertLess(release.index("et_i64_tensor_destroy_v1"),
                        release.index("shell->released = 1"))
        self.assertGreaterEqual(source.count("shell->release_pending || shell->released"), 6)
        self.assertIn("if (!entry->released) ++count", source)


if __name__ == "__main__":
    unittest.main()
