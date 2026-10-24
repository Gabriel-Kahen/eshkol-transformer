"""Source custody for the private FILE/C2-to-TR3 restore candidate."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class PairLiveRestoreSource(unittest.TestCase):
    def test_one_registry_and_no_second_c2_root(self):
        source = (ROOT / "native/g3r_c2_pair_live_restore_root.esk").read_text()
        self.assertEqual(source.count('(load "cli3_generate_root.esk")'), 1)
        self.assertNotIn('(load "g3r_c2_sha_receipt_root.esk")', source)
        self.assertLess(source.index('(load "cli3_generate_root.esk")'),
                        source.index('(load "g3r_c2_sha_stage_extension.esk")'))
        self.assertLess(source.index('(load "g3r_c2_pair_tokenizer_bind_extension.esk")'),
                        source.index('(load "g3r_c2_pair_live_restore_extension.esk")'))

    def test_pair_checks_and_guard_precede_restore(self):
        source = (ROOT / "native/g3r_c2_pair_live_restore_extension.esk").read_text()
        ordered = (
            "g3r-c2-pair-file-verify-c2-tokenizer-internal!",
            "t2-private-tokenizer-vocab-size",
            "t2-private-tokenizer-fingerprint",
            "tr3-lease-record-for-shell-internal",
            "(guard (cleanup-defect",
            "(guard (caught",
            "(vector-set! receipt-record 1 'restoring)",
            "(trainer-load-state! trainer owner)",
        )
        offsets = [source.index(fragment) for fragment in ordered]
        self.assertEqual(offsets, sorted(offsets))
        self.assertIn("(c2-load-exact-live-owner? owner)", source)
        self.assertIn("(eq? (vector-ref trainer-record 3) 'idle)", source)
        self.assertIn("(g3r-c2-sha-fail-stop 134)", source)


if __name__ == "__main__":
    unittest.main()
