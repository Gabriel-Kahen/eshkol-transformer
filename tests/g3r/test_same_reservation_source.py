"""Bounded source contract for the private G3-R receipt/TR3 bridge."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "native/g3r_same_reservation_extension.esk").read_text()
CLOSURE = (ROOT / "native/g3r_same_reservation_source_closure.txt").read_text().splitlines()


def function(name):
    start = SOURCE.index(f"(define ({name} ")
    end = SOURCE.find("\n(define (", start + 1)
    return SOURCE[start:] if end < 0 else SOURCE[start:end]


class SameReservationSourceTest(unittest.TestCase):
    def test_private_predecessor_inventory(self):
        self.assertEqual(len(CLOSURE), len(set(CLOSURE)))
        self.assertEqual({path for path in CLOSURE if not (ROOT / path).is_file()},
                         {"native/d2_semantic_core.esk"})
        self.assertIn("native/g3r_checked_pair_extension.esk", CLOSURE)
        self.assertIn("native/tr3_c_joint_restore_extension.esk", CLOSURE)
        self.assertIn("native/g3r_same_reservation_extension.esk", CLOSURE)
        self.assertLess(CLOSURE.index("native/g3r_checked_pair_extension.esk"),
                        CLOSURE.index("native/g3r_same_reservation_extension.esk"))
        self.assertLess(CLOSURE.index("native/tr3_c_joint_restore_extension.esk"),
                        CLOSURE.index("native/g3r_same_reservation_extension.esk"))

    def test_candidate_uses_registered_exact_tuple(self):
        enroll = function("g3r-same-reservation-enroll-internal")
        recheck = function("g3r-same-reservation-trainer!")
        self.assertIn("tr3-lease-record-for-shell-internal", enroll)
        self.assertIn("g3r-checked-pair-recheck-internal", enroll)
        self.assertIn("g3r-same-reservation-overlap?", enroll)
        for name in ("d2-open-dataset-state", "m3t-entry", "o2-require-optimizer",
                     "tr3-lease-e3-byte-profile?", "d2-tokenizer-identity"):
            self.assertIn(name, recheck)

    def test_validation_keeps_one_receipt_borrow(self):
        gate = function("g3r-same-reservation-restore-internal!")
        verify = function("g3r-same-reservation-verify!")
        self.assertEqual(gate.count("g3r-c2-receipt-borrow-begin-internal"), 1)
        self.assertIn("g3r-c2-receipt-borrow-c2-internal", verify)
        self.assertIn("g3r-c2-receipt-borrow-sha-internal", verify)
        self.assertIn("c2-training-state-borrow-projection-internal", verify)
        self.assertIn("c2-training-state-controls-copy", verify)
        self.assertIn("c2-d2-cursor-pair-internal", verify)
        self.assertIn("d2-core-state-identity=?", verify)
        self.assertIn("tr3-c-joint-fixed-source!", verify)
        self.assertIn("o2-require-c2-owned-state", verify)
        self.assertIn("g3r-same-reservation-source-bits!", verify)
        self.assertLess(gate.index("g3r-same-reservation-verify!"),
                        gate.index("g3r-c2-receipt-borrow-end-internal! borrow"))
        self.assertLess(gate.index("g3r-c2-receipt-borrow-end-internal! borrow"),
                        gate.index("(vector-set! receipt 1 'restoring)"))
        self.assertLess(gate.index("(vector-set! receipt 1 'restoring)"),
                        gate.index("tr3-c-joint-restore-internal!"))

    def test_uncertain_cuts_fail_stop(self):
        gate = function("g3r-same-reservation-restore-internal!")
        before = function("g3r-same-reservation-before-tr3!")
        self.assertIn("(guard (end-defect (#t (g3r-c2-sha-fail-stop 134)))", gate)
        self.assertIn("(guard (tr3-error (#t (g3r-c2-sha-fail-stop 134)))", gate)
        self.assertIn("(if (eq? (vector-ref row 1) 'restoring)", gate)
        self.assertIn("g3r-same-reservation-before-tr3!", gate)
        self.assertIn("(eq? (vector-ref receipt 2) owner)", before)
        self.assertIn("(eq? (vector-ref state 7) 'live)", before)
        self.assertLess(before.index("(vector-set! receipt 1 'live)"),
                        before.index("(raise caught)"))
        self.assertNotIn("g3r-c2-receipt-release-internal!", SOURCE)
        for forbidden in ("(open ", "(read ", "checkpoint-save", "generator-generate!",
                          "(provide "):
            self.assertNotIn(forbidden, SOURCE)


if __name__ == "__main__":
    unittest.main()
