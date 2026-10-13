"""Source-only contract checks for feature-private P2/G2 holder wiring."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import parse


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "native/g3c4_p2g2_terminal_held_holder_extension.esk").read_text()
ROOT_SOURCE = (ROOT / "native/g3c4_p2g2_terminal_held_holder_root.esk").read_text()
ORDINARY = (ROOT / "native/g3c4_p2g2_terminal_composer_extension.esk").read_text()


def ordered(source, *fragments):
    at = -1
    for fragment in fragments:
        found = source.find(fragment, at + 1)
        if found < 0:
            return False
        at = found
    return True


class HeldHolderSource(unittest.TestCase):
    def test_closed_feature_and_exact_abi(self):
        self.assertTrue(parse(SOURCE))
        self.assertTrue(parse(ROOT_SOURCE))
        self.assertIn("(define (g3c4-p2g2-terminal-generate-held-internal generator input holder)", SOURCE)
        self.assertIn("(define (g3c4-p2g2-terminal-generate-internal generator input)", ORDINARY)
        self.assertNotIn("p2g2-holder-new", ORDINARY)
        self.assertIn('(load "g3c4_p2g2_terminal_held_holder_extension.esk")', ROOT_SOURCE)
        self.assertNotIn('(load "g3c4_p2g2_terminal_composer_extension.esk")', ROOT_SOURCE)
        closure = (ROOT / "native/g3c4_p2g2_terminal_held_holder_source_closure.txt").read_text().splitlines()
        self.assertEqual(len(closure), len(set(closure)))
        self.assertTrue(all((ROOT / entry).is_file() for entry in closure))
        self.assertNotIn("native/g3c4_p2g2_terminal_composer_extension.esk", closure)
        self.assertIn("(or (= kind 1) (= kind 2))", SOURCE)
        self.assertIn("(or (= phase 2) (= phase 3))", SOURCE)
        self.assertIn("(eq? (car (vector-ref g3c4-p2g2-holder-roots 0)) original)", SOURCE)
        self.assertIn("(= (g3c4-p2g2-holder-generation (vector-ref shell 1))", SOURCE)
        header = (ROOT / "src/eshkol_transformer/g3c4_p2g2_holder_private.h").read_text()
        for symbol in ("new", "generation", "bind", "begin_capture", "witness",
                       "normal_close", "abort_attempt", "end_after_busy", "dispose"):
            self.assertIn(f"et_g3c4_private_p2g2_holder_{symbol}_v1", header)
            self.assertIn(f"et_g3c4_private_p2g2_holder_{symbol}_v1", SOURCE)
        native_fixture = (ROOT / "tests/g3c4/test_p2_g2_held_holder.c").read_text()
        self.assertIn("holder_case(owner, phase, kind, 0)", native_fixture)
        self.assertIn("holder_case(owner, phase, kind, 1)", native_fixture)

    def test_transfer_and_both_capture_positions(self):
        self.assertTrue(ordered(SOURCE,
            "(holder-original (g3c4-p2g2-holder-available-check holder operation))",
            "(guard (caught", "(g3c4-p2g2-holder-transfer! holder)",
            "(g3c4-native-call-acquire native 2 2)",
            "(output (g3c4-p2g2-protected-reserve call))",
            "(g3c4-p2g2-holder-bind!", "(g3c4-p2g2-native-eos-first",
            "(g3c4-p2g2-holder-begin!", "(t1-private-g3-decode-raw-into!",
            "(g3c4-p2g2-holder-close!", "(let ((count",
            "(g3c4-p2g2-native-second", "(g3c4-p2g2-holder-begin!",
            "(t1-private-g3-decode-raw-into!", "(g3c4-p2g2-holder-close!",
            "(g3c4-p2g2-native-terminal"))
        self.assertIn("P3 holder cannot follow first-frame EOS", SOURCE)
        self.assertIn("(if (= (vector-ref holder 4) 2)", SOURCE)
        self.assertIn("(if (= (vector-ref holder 4) 3)", SOURCE)

    def test_typed_busy_retry_and_no_postcommit_holder_use(self):
        abort = SOURCE[SOURCE.index("(define (g3c4-p2g2-holder-abort-held!"):
                       SOURCE.index("(define (g3c4-p2g2-terminal-abort-clean!")]
        self.assertTrue(ordered(abort, "context output 1", "(if (= first 0) (exit 134))",
            "(g3c4-p2g2-holder-end-busy", "(g3c4-p2g2-terminal-active-check",
            "(g3c4-p2g2-holder-witness", "context output 2",
            "(g3c4-p2g2-held-protected-postzero!",
            "(g3c4-p2g2-holder-dispose-root!"))
        self.assertIn("(if (not (= (g3c4-p2g2-holder-abort", abort)
        self.assertEqual(abort.count("context output 2"), 1)
        self.assertIn("(if (vector-ref barrier 0) (exit 134))", SOURCE)
        self.assertIn("(if (not (= (g3c4-p2g2-holder-abort", SOURCE)
        tail = SOURCE[SOURCE.index("(vector-set! barrier 0 #t)"):
                      SOURCE.index("(define (g3c4-p2g2-terminal-entry-live")]
        self.assertNotIn("(g3c4-p2g2-holder-", tail)

    def test_wrong_handle_and_postzero_negative_guards(self):
        check = SOURCE[SOURCE.index("(define (g3c4-p2g2-holder-original"):
                       SOURCE.index("(define (g3c4-p2g2-holder-transfer!")]
        self.assertIn("(eq? (vector-ref shell 0) g3c4-p2g2-holder-brand)", check)
        self.assertIn("(eq? (vector-ref original 1) (vector-ref shell 1))", check)
        self.assertIn("(= (g3c4-p2g2-holder-generation (vector-ref shell 1))", check)
        self.assertIn("(eq? (car (vector-ref g3c4-p2g2-holder-roots 0)) original)", check)
        self.assertFalse(ordered(check.replace("g3c4-p2g2-holder-brand", "wrong-brand"),
                                 "(eq? (vector-ref shell 0) g3c4-p2g2-holder-brand)"))
        self.assertTrue(ordered(SOURCE, "(g3c4-p2g2-holder-close!",
                                "(g3c4-p2g2-native-terminal",
                                "(vector-set! barrier 0 #t)"))

    def test_rejects_missing_phase_or_retry_gates(self):
        self.assertFalse(ordered(SOURCE.replace("(g3c4-p2g2-holder-end-busy", "(missing-end-busy"),
                                 "context output 1", "(g3c4-p2g2-holder-end-busy",
                                 "context output 2"))
        self.assertFalse(ordered(SOURCE.replace("P3 holder cannot follow first-frame EOS", ""),
                                 "P3 holder cannot follow first-frame EOS"))


if __name__ == "__main__":
    unittest.main()
