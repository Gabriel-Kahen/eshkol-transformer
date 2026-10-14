"""Mutation-sensitive source gate for the private pre-eval handoff fence."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = (ROOT / "native/tr3_lease_core_extension.esk").read_text()
CAPSULE = (ROOT / "native/tr3_c_handoff_capsule_extension.esk").read_text()
AGGREGATE = (ROOT / "native/tr3_c_handoff_capsule_root.esk").read_text()


def body(source: str, name: str) -> str:
    start = source.index(f"(define ({name}")
    end = source.find("\n(define ", start + 1)
    return source[start : end if end >= 0 else len(source)]


def boundary_has_required_cuts(source: str) -> bool:
    check = body(source, "tr3-handoff-capsule-check!")
    capture = body(source, "tr3-handoff-capture-internal!")
    return all(marker in check for marker in (
        "(eq? (vector-ref capsule 1) capsule)",
        "(eq? (vector-ref record 4) capsule)",
        "(vector-ref (vector-ref record 14) 1)",
        "(vector-ref (vector-ref record 16) 1)",
        "(eq? (car (cdr t1-entry))",
        "(not (equal? (vector-ref capsule 7)",
    )) and "(tr3-c-snapshot-p1-release! (vector-ref p1-cell 0))" in capture


class HandoffCapsuleContract(unittest.TestCase):
    def test_private_fixed_shape_and_aggregate(self):
        self.assertIn('(load "tr3_c_snapshot_root.esk")', AGGREGATE)
        self.assertIn('(load "tr3_c_handoff_capsule_extension.esk")', AGGREGATE)
        self.assertIn("(define tr3-handoff-capsule-length 10)", CAPSULE)
        self.assertEqual(CAPSULE.count("(define (tr3-handoff-capture-internal!"), 1)
        for directory in ("include", "lib", "src"):
            for path in (ROOT / directory).rglob("*"):
                if path.is_file() and path.suffix in {".h", ".c", ".cpp", ".esk"}:
                    self.assertNotIn(
                        "tr3-handoff-capture-internal!",
                        path.read_text(errors="replace"), str(path))

    def test_forged_copied_stale_capsule_and_phase_are_rejected(self):
        check = body(CAPSULE, "tr3-handoff-capsule-check!")
        abort = body(CAPSULE, "tr3-handoff-abort-before-eval-internal!")
        for marker in (
            "tr3-lease-authenticate-minimal record",
            "(= (vector-length capsule) tr3-handoff-capsule-length)",
            "(eq? (vector-ref capsule 0) tr3-handoff-capsule-tag)",
            "(eq? (vector-ref capsule 1) capsule)",
            "(eq? (vector-ref capsule 2) record)",
            "(eq? (vector-ref capsule 3) (vector-ref record 5))",
            "(eq? (vector-ref capsule 4) (tr3-handoff-node record))",
            "(eq? (vector-ref record 4) capsule)",
            "(eq? (vector-ref record 3) 'handoff)",
        ):
            self.assertIn(marker, check)
        self.assertIn("(eq? (vector-ref record 4) capsule)", abort)
        self.assertIn("tr3-handoff-capsule-check! capsule record 'fenced", abort)

    def test_swapped_identity_and_changed_controls_are_rejected(self):
        check = body(CAPSULE, "tr3-handoff-capsule-check!")
        for slot in range(20):
            self.assertIn(f"(vector-ref owners {slot})", check)
        for owner, record, width in ((5, 10, 17), (6, 11, 14), (7, 12, 14)):
            self.assertIn(
                f"(vector-ref owners {owner}) (vector-ref record {record}) {width}",
                check)
        for marker in (
            "tr3-lease-t2-witness-valid?",
            "(t2-private-entry (vector-ref owners 18))",
            "(eq? (car (cdr t1-entry))",
            "(vector-ref (vector-ref record 14) 1)",
            "(vector-ref (vector-ref record 16) 1)",
            "(not (equal? (vector-ref capsule 7) (tr3-handoff-controls record)))",
            "tr3-lease-config-witness!",
            "tr3-lease-tokenizer-dataset!",
            "tr3-lease-restore-enter-admission!",
        ):
            self.assertIn(marker, check)

    def test_busy_reentry_and_allocation_failure_cleanup(self):
        capture = body(CAPSULE, "tr3-handoff-capture-internal!")
        for marker in (
            "tr3-handoff-capture-active?", "tr3-acquisition-active?",
            "tr3-lease-unenroll-active?", "(vector-ref tr3-trainers 1)",
            "(guard (caught", "(tr3-c-snapshot-p1-release! (vector-ref p1-cell 0))",
            "(set! tr3-handoff-capture-active? #f)", "(raise caught)",
        ):
            self.assertIn(marker, capture)
        for name in (
            "tr3-lease-create-internal", "tr3-lease-record-for-shell-internal",
            "tr3-lease-unenroll-internal!", "tr3-lease-phase!",
        ):
            self.assertIn("tr3-handoff-capture-active?", body(CORE, name))
        order = (
            "(tr3-lease-original-t2-core-internal",
            "(set! tr3-handoff-capture-active? #t)",
            "(tr3-handoff-controls record)",
            "(tr3-c-snapshot-p1-call",
            "(tr3-handoff-capsule-check! capsule record 'prepared)",
            "(vector-set! record 4 capsule)",
            "(vector-set! record 3 'handoff)",
        )
        self.assertEqual([capture.index(x) for x in order],
                         sorted(capture.index(x) for x in order))

    def test_pre_eval_abort_revalidates_then_releases_before_unlink(self):
        abort = body(CAPSULE, "tr3-handoff-abort-before-eval-internal!")
        for marker in (
            "(tr3-lease-authenticate record",
            "tr3-handoff-capsule-check! capsule record 'fenced",
            "(tr3-c-snapshot-p1-release! (vector-ref p1-cell 0))",
            "(tr3-lease-fail-stop 134)",
            "(vector-set! capsule 5 'dead)",
            "(vector-set! record 4 #f)",
            "(vector-set! record 3 'idle)",
        ):
            self.assertIn(marker, abort)
        self.assertLess(abort.index("(tr3-c-snapshot-p1-release!"),
                        abort.index("(vector-set! record 4 #f)"))
        self.assertLess(abort.index("(vector-set! capsule 5 'dead)"),
                        abort.index("(vector-set! record 4 #f)"))

    def test_negative_source_mutants(self):
        # The guarded identity and cleanup cuts must each remain independently
        # present; a byte-identical foreign owner cannot replace any one.
        self.assertTrue(boundary_has_required_cuts(CAPSULE))
        markers = {
            "copy": "(eq? (vector-ref capsule 1) capsule)",
            "stale": "(eq? (vector-ref record 4) capsule)",
            "O2 pointer": "(vector-ref (vector-ref record 14) 1)",
            "D2 pointer": "(vector-ref (vector-ref record 16) 1)",
            "T1 core": "(eq? (car (cdr t1-entry))",
            "cursor/counters": "(not (equal? (vector-ref capsule 7)",
            "P1 cleanup": "(tr3-c-snapshot-p1-release! (vector-ref p1-cell 0))",
        }
        for name, marker in markers.items():
            with self.subTest(name=name):
                function = ("tr3-handoff-capture-internal!" if name == "P1 cleanup"
                            else "tr3-handoff-capsule-check!")
                original = body(CAPSULE, function)
                self.assertIn(marker, original)
                mutant = CAPSULE.replace(original, original.replace(marker, "", 1), 1)
                self.assertFalse(
                    boundary_has_required_cuts(mutant))


if __name__ == "__main__":
    unittest.main()
