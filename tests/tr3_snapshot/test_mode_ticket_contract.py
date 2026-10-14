"""Causal source contract for the private snapshot-owned eval ticket."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "native/tr3_lease_core_extension.esk"
SNAPSHOT = ROOT / "native/tr3_c_snapshot_extension.esk"
FIXTURE = ROOT / "tests/tr3_snapshot/runtime_smoke.esk"


def body(source: str, name: str, successor: str) -> str:
    start = source.index(f"(define ({name}")
    return source[start:source.index(f"(define ({successor}", start)]


def ordered(source: str, *anchors: str) -> bool:
    try:
        offsets = [source.index(anchor) for anchor in anchors]
    except ValueError:
        return False
    return offsets == sorted(offsets)


def contract(core: str, snapshot: str) -> None:
    assert "(define tr3-lease-snapshot-parent-length 10)" in core
    assert "(define tr3-lease-snapshot-mode-root (vector #f #f))" in core
    abort_core = body(core, "tr3-lease-snapshot-abort-internal!",
                      "tr3-lease-snapshot-finish-tail-internal!")
    finish = body(core, "tr3-lease-snapshot-finish-tail-internal!",
                  "tr3-lease-restore-parent-header!")
    abort = body(snapshot, "tr3-c-snapshot-abort!", "tr3-c-snapshot-compose!")
    entry = snapshot[snapshot.index("(define (tr3-c-trainer-state-internal"):]
    for text, first_write in (
        (abort_core, "(vector-set! parent 3 'dead)"),
        (finish, "(vector-set! record 4 #f)"),
        (abort, "(vector-set! parent 3 'aborting)"),
    ):
        assert ordered(text, "(eq? (vector-ref tr3-lease-snapshot-mode-root 0) parent)",
                       "(tr3-lease-fail-stop 134)", first_write)
    assert ordered(entry,
                   "(eq? (vector-ref tr3-lease-snapshot-mode-root 0) parent)",
                   "(tr3-lease-fail-stop 134)",
                   "(vector-set! result-cell 0 owner)",
                   "(vector-set! parent 3 'dead)")
    drain = body(snapshot, "tr3-c-snapshot-mode-drain!",
                 "tr3-c-snapshot-with-eval-internal!")
    for identity in (
        "(vector-ref ticket 0)",
        "(vector-ref ticket 1) ticket",
        "(vector-ref tr3-lease-snapshot-mode-root 0) parent",
        "(vector-ref tr3-lease-snapshot-mode-root 1) ticket",
        "(vector-ref ticket 5) token",
        "(vector-ref record 4) parent",
        "(vector-ref parent 1) parent",
        "(vector-ref parent 2) (vector-ref record 2)",
    ):
        assert identity in drain
    assert ordered(drain,
                   "(vector-ref p1-trusted-surface 67)",
                   "(tr3-c-snapshot-mode-p1-recheck",
                   "(vector-ref p1-trusted-surface 68)",
                   "(vector-set! ticket 6 'dead)",
                   "(vector-set! tr3-lease-snapshot-mode-root 1 #f)",
                   "(vector-set! tr3-lease-snapshot-mode-root 0 #f)")
    assert "(tr3-lease-fail-stop 134)" in drain
    transition = body(snapshot, "tr3-c-snapshot-with-eval-internal!",
                      "tr3-c-snapshot-abort!")
    assert ordered(transition,
                   "(tr3-lease-snapshot-recheck-internal record parent)",
                   "(vector-set! tr3-lease-snapshot-mode-root 0 parent)",
                   "(vector-ref p1-trusted-surface 64)",
                   "(vector-ref p1-trusted-surface 65)",
                   "(vector-ref p1-trusted-surface 66)",
                   "(tr3-c-snapshot-mode-p1-recheck record #t)",
                   "(let ((result (thunk)))")
    assert transition.count("(tr3-c-snapshot-mode-p1-recheck record #t)") == 2
    assert transition.count("(tr3-c-snapshot-mode-drain! ticket)") == 2
    assert "(if (vector-ref tr3-lease-snapshot-mode-root 0)" in transition
    assert "(vector tr3-c-snapshot-mode-ticket-tag #f record parent" in transition
    assert "(tr3-c-snapshot-rethrow caught)" in transition
    assert "(tr3-lease-snapshot-recheck-internal record parent)\n        result" in transition
    assert "module-eval!" not in transition
    assert "module-train!" not in transition


class SnapshotModeTicketContract(unittest.TestCase):
    def test_exact_contract_and_causal_mutants(self):
        core, snapshot = CORE.read_text(), SNAPSHOT.read_text()
        contract(core, snapshot)
        mutations = (
            (snapshot, "(vector-ref p1-trusted-surface 66)",
             "(vector-ref p1-trusted-surface 65)"),
            (snapshot, "(vector-ref p1-trusted-surface 67)",
             "(vector-ref p1-trusted-surface 66)"),
            (snapshot, "(vector-ref p1-trusted-surface 68)",
             "(vector-ref p1-trusted-surface 67)"),
            (snapshot, "(tr3-c-snapshot-mode-p1-recheck record #t)",
             "(tr3-c-snapshot-mode-p1-recheck record #f)"),
            (snapshot, "(vector-ref tr3-lease-snapshot-mode-root 1) ticket",
             "(vector-ref tr3-lease-snapshot-mode-root 1) parent"),
            (core, "(define tr3-lease-snapshot-mode-root (vector #f #f))",
             "(define tr3-lease-snapshot-mode-root (vector #f))"),
        )
        for original, before, after in mutations:
            with self.subTest(before=before):
                self.assertIn(before, original)
                changed = original.replace(before, after, 1)
                with self.assertRaises(AssertionError):
                    contract(changed if original == core else core,
                             changed if original == snapshot else snapshot)
        for original, header, early_write in (
            (core, "(define (tr3-lease-snapshot-abort-internal! record parent)\n",
             "(vector-set! parent 3 'dead)"),
            (snapshot, "(define (tr3-c-snapshot-abort! record parent)\n",
             "(vector-set! parent 3 'aborting)"),
            (snapshot, "(let ((owner (tr3-c-snapshot-compose! record parent)))\n",
             "(vector-set! result-cell 0 owner)"),
        ):
            with self.subTest(early_write=early_write):
                poisoned = original.replace(header, header + "  " + early_write + "\n", 1)
                self.assertNotEqual(poisoned, original)
                with self.assertRaises(AssertionError):
                    contract(poisoned if original == core else core,
                             poisoned if original == snapshot else snapshot)

    def test_genuine_fixture_stages_success_copy_failure_and_rollback(self):
        fixture = FIXTURE.read_text()
        for label in (
            "mode ticket admits exact original eval set",
            "train recheck rejects entered eval",
            "mode ticket restores train before returning",
            "mode ticket rejects copied snapshot parent",
            "overlapping mode ticket is rejected",
            "mode ticket rolls back callback failure",
            "failed mode ticket leaves original train and live parent",
            "mode ticket abort occurs after train restoration",
            "stale mode parent cannot reenter",
        ):
            self.assertIn(label, fixture)


if __name__ == "__main__":
    unittest.main()
