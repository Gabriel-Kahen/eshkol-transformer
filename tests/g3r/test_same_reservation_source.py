"""Bounded source contract for the private G3-R receipt/TR3 bridge."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "native/g3r_same_reservation_extension.esk").read_text()
CLOSURE = (ROOT / "native/g3r_same_reservation_source_closure.txt").read_text().splitlines()
CHECKED = (ROOT / "native/g3r_checked_pair_source_closure.txt").read_text().splitlines()
TR3 = (ROOT / "native/tr3_c_joint_restore_source_closure.txt").read_text().splitlines()
D2_BASE = (ROOT / "native/tr3_c_d2_restore_source_closure.txt").read_text().splitlines()
DEPENDENCY = re.compile(r'^\s*\((load|require)\s+(?:"([^"]+)"|([^\s)]+))\)',
                        re.MULTILINE)
INCLUDE_DIRS = ("internal/p1/lib", "internal/c1/lib", "internal/t2/lib",
                "internal/t1/lib", "internal/d2/lib", "src", "lib", "native")


def function(name):
    start = SOURCE.index(f"(define ({name} ")
    end = SOURCE.find("\n(define (", start + 1)
    return SOURCE[start:] if end < 0 else SOURCE[start:end]


class SameReservationSourceTest(unittest.TestCase):
    def test_private_predecessor_inventory(self):
        self.assertEqual(len(CLOSURE), len(set(CLOSURE)))
        self.assertEqual([path for path in CLOSURE if not (ROOT / path).is_file()], [])
        self.assertEqual(TR3[:len(D2_BASE) - 1], D2_BASE[:-1])
        new = {"native/g3r_same_reservation_extension.esk",
               "native/g3r_same_reservation_source_closure.txt",
               "tests/g3r/test_same_reservation_source.py",
               "docs/g3/G3_R_SAME_RESERVATION_SOURCE_GATE.md"}
        self.assertEqual(set(CLOSURE), set(CHECKED) | set(TR3) | new)
        for parent in (CHECKED, TR3):
            self.assertEqual([path for path in CLOSURE if path in parent], parent)
        self.assertIn("internal/d2/lib/d2_semantic_core.esk", CLOSURE)
        self.assertNotIn("native/d2_semantic_core.esk", CLOSURE)
        self.assertIn("native/g3r_checked_pair_extension.esk", CLOSURE)
        self.assertIn("native/tr3_c_joint_restore_extension.esk", CLOSURE)
        self.assertIn("native/g3r_same_reservation_extension.esk", CLOSURE)
        self.assertLess(CLOSURE.index("native/g3r_checked_pair_extension.esk"),
                        CLOSURE.index("native/g3r_same_reservation_extension.esk"))
        self.assertLess(CLOSURE.index("native/tr3_c_joint_restore_extension.esk"),
                        CLOSURE.index("native/g3r_same_reservation_extension.esk"))

    def test_transitive_root_loads_are_in_closure(self):
        roots = ("native/g3r_checked_pair_root.esk",
                 "native/tr3_c_joint_restore_root.esk")
        pending, seen = list(roots), set()
        while pending:
            source = pending.pop()
            if source in seen:
                continue
            seen.add(source)
            for kind, load, module in DEPENDENCY.findall((ROOT / source).read_text()):
                name = load if kind == "load" else module.replace(".", "/") + ".esk"
                matches = [f"{directory}/{name}" for directory in INCLUDE_DIRS
                           if (ROOT / directory / name).is_file()]
                self.assertTrue(matches, (source, kind, name))
                target = matches[0]
                self.assertIn(target, CLOSURE, (source, kind, name))
                pending.append(target)
        self.assertIn("internal/c1/lib/c1_sha256.esk", seen)
        self.assertIn("internal/p1/lib/transformer/module.esk", seen)
        self.assertIn("internal/t2/lib/transformer/tokenizer_internal.esk", seen)
        self.assertIn("internal/d2/lib/d2_semantic_core.esk", seen)

        expected_loads = {
            roots[0]: ("g3r_c2_sha_receipt_root.esk", "g3r_record_codec_extension.esk",
                       "g3r_checked_pair_extension.esk"),
            "native/d2_wave2_root.esk":
                ("t2_wave2_root.esk", "i2_wave2_extension.esk",
                 "d2_semantic_core.esk", "d2_dataset.esk"),
            "native/t2_wave2_root.esk":
                ("eshkol_transformer/token_shard.esk",
                 "transformer/tokenizer_internal.esk", "t2_d1_bridge.esk"),
            roots[1]:
                ("d2_wave2_root.esk", "c2_d2_cursor_pair_extension.esk",
                 "o2_wave2_extension.esk", "c2_o2_reconstruct_extension.esk",
                 "c2_x1_canonical_extension.esk", "c2_training_state_extension.esk",
                 "m3t_transport_extension.esk", "m3_schedule.esk",
                 "m3_model_extension.esk", "tr3_p1_fixed_set_extension.esk",
                 "tr3_lease_core_extension.esk", "tr3_lease_public_gates_extension.esk",
                 "tr3_c_d2_restore_extension.esk",
                 "tr3_c_restore_bindings_extension.esk",
                 "tr3_c_joint_restore_extension.esk")}
        for source, expected in expected_loads.items():
            actual = [load for kind, load, _ in
                      DEPENDENCY.findall((ROOT / source).read_text())
                      if kind == "load"]
            self.assertEqual(tuple(actual), expected, source)

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
