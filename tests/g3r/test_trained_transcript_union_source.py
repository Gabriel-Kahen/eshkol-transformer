"""Mutation guards for the pinned producer/consumer source-union preflight."""

import unittest

from tests.g3r.preflight_trained_transcript_union import (
    IMPORTS, REPLAY, REPLAY_CLOSURE, SEAM, SEAM_CLOSURE,
    check_union, git_blob, git_show, paths,
)


class TrainedTranscriptUnionSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seam = paths(git_show(SEAM, SEAM_CLOSURE))
        cls.replay = paths(git_show(REPLAY, REPLAY_CLOSURE))

    def check(self, *, seam=None, seam_blob=None, read_seam=None):
        return check_union(
            self.seam if seam is None else seam, self.replay,
            (lambda p: git_blob(SEAM, p)) if seam_blob is None else seam_blob,
            lambda p: git_blob(REPLAY, p),
            (lambda p: git_show(SEAM, p)) if read_seam is None else read_seam,
            lambda p: git_show(REPLAY, p),
        )

    def test_exact_union(self):
        consumer = self.check()
        self.assertEqual(consumer[-2:], IMPORTS)
        self.assertEqual(len(consumer), 103)

    def test_duplicate_closure_rejected(self):
        with self.assertRaises(AssertionError):
            paths("native/a.esk\nnative/a.esk\n")

    def test_shared_blob_mutation_rejected(self):
        target = "native/cli3_generate_root.esk"
        with self.assertRaisesRegex(AssertionError, "blob drift"):
            self.check(seam_blob=lambda p: "mutated" if p == target
                       else git_blob(SEAM, p))

    def test_second_root_rejected(self):
        with self.assertRaisesRegex(AssertionError, "closure count"):
            self.check(seam=self.seam + ("native/g3r_trained_replay_root.esk",))

    def test_root_load_mutation_rejected(self):
        target = self.seam[0]
        with self.assertRaisesRegex(AssertionError, "root load order"):
            self.check(read_seam=lambda p: git_show(SEAM, p) +
                       '\n(load "g3r_trained_replay_root.esk")'
                       if p == target else git_show(SEAM, p))

    def test_duplicate_transitive_load_rejected(self):
        target = "native/g3r_trained_pair_producer_root.esk"
        with self.assertRaisesRegex(AssertionError, "duplicate transitive"):
            self.check(read_seam=lambda p: git_show(SEAM, p) +
                       '\n(load "cli3_generate_root.esk")'
                       if p == target else git_show(SEAM, p))


if __name__ == "__main__":
    unittest.main()
