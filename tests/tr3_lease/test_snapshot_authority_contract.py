"""Source contract for the private TR3-C snapshot lease authority."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORE = (ROOT / "native" / "tr3_lease_core_extension.esk").read_text()


def definition(name: str, next_name: str) -> str:
    start = CORE.index(f"(define ({name}")
    return CORE[start : CORE.index(f"(define ({next_name}", start)]


class SnapshotAuthorityContract(unittest.TestCase):
    def test_parent_is_fixed_self_bound_and_owns_the_required_cells(self) -> None:
        self.assertIn("(define tr3-lease-snapshot-parent-length 10)", CORE)
        header = definition(
            "tr3-lease-snapshot-parent-header!",
            "tr3-lease-snapshot-parent-empty!",
        )
        for required in (
            "tr3-lease-snapshot-parent-length",
            "tr3-lease-snapshot-parent-tag",
            "(eq? (vector-ref parent 1) parent)",
            "(eq? (vector-ref parent 2) trainer)",
        ):
            self.assertIn(required, header)
        empty = definition(
            "tr3-lease-snapshot-parent-empty!",
            "tr3-lease-snapshot-enter-internal!",
        )
        for slot in range(4, 10):
            self.assertIn(f"(vector-ref parent {slot})", empty)
        self.assertIn("(= (vector-length controls) 5)", empty)
        self.assertIn("(= (vector-length cleanup) 2)", empty)
        self.assertGreaterEqual(empty.count("(eq?"), 15)

    def test_enter_promotes_parent_before_phase_publication(self) -> None:
        enter = definition(
            "tr3-lease-snapshot-enter-internal!",
            "tr3-lease-snapshot-recheck-internal",
        )
        admission = enter.index("(tr3-lease-restore-enter-admission!")
        parent = enter.index("(vector-set! record 4 parent)")
        phase = enter.index("(vector-set! record 3 'snapshotting)")
        self.assertLess(admission, parent)
        self.assertLess(parent, phase)
        for forbidden in ("(if ", "(guard ", "m3t-fail", "o2-native-"):
            self.assertNotIn(forbidden, enter[parent:])

    def test_recheck_binds_exact_parent_and_revalidates_components(self) -> None:
        recheck = definition(
            "tr3-lease-snapshot-recheck-internal",
            "tr3-lease-snapshot-abort-internal!",
        )
        for required in (
            "tr3-lease-authenticate-minimal",
            "tr3-lease-authenticate",
            "(eq? (vector-ref enrolled 3) 'snapshotting)",
            "(eq? (vector-ref enrolled 4) parent)",
            "tr3-lease-snapshot-parent-header!",
            "tr3-lease-restore-enter-admission!",
        ):
            self.assertIn(required, recheck)

    def test_abort_authenticates_then_uses_assignment_only_unlink(self) -> None:
        abort = definition(
            "tr3-lease-snapshot-abort-internal!",
            "tr3-lease-snapshot-finish-tail-internal!",
        )
        self.assertIn("tr3-lease-authenticate-minimal", abort)
        first_store = abort.index("(vector-set! parent 3 'dead)")
        self.assertLess(first_store, abort.index("(vector-set! enrolled 4 #f)"))
        self.assertLess(
            abort.index("(vector-set! enrolled 4 #f)"),
            abort.index("(vector-set! enrolled 3 'idle)"),
        )
        for forbidden in (
            "tr3-lease-authenticate",
            "tr3-lease-restore-enter-admission!",
            "m3t-fail",
            "o2-native-",
            "d2-core-",
            "(guard ",
        ):
            self.assertNotIn(forbidden, abort[first_store:])

    def test_finish_is_fail_stop_checked_and_assignment_only(self) -> None:
        finish = definition(
            "tr3-lease-snapshot-finish-tail-internal!",
            "tr3-lease-restore-parent-header!",
        )
        self.assertIn("tr3-lease-fail-stop 134", finish)
        self.assertIn("(eq? (vector-ref parent 3) 'dead)", finish)
        first_store = finish.index("(vector-set! record 4 #f)")
        self.assertLess(first_store, finish.index("(vector-set! record 3 'idle)"))
        for forbidden in (
            "tr3-lease-authenticate",
            "tr3-lease-operation-fail",
            "m3t-fail",
            "o2-native-",
            "d2-core-",
            "(guard ",
        ):
            self.assertNotIn(forbidden, finish)

    def test_fixed_snapshot_admission_requires_exact_active_parent(self) -> None:
        snapshot = definition(
            "tr3-lease-snapshot-internal",
            "tr3-lease-restore-internal",
        )
        self.assertIn("'snapshotting snapshot-parent", snapshot)
        self.assertIn("tr3-lease-snapshot-parent-header!", snapshot)
        self.assertNotIn("'snapshotting #f", snapshot)

    def test_surface_is_private_and_does_not_add_an_authority_root(self) -> None:
        self.assertEqual(CORE.count("(define tr3-trainers "), 1)
        names = (
            "tr3-lease-snapshot-enter-internal!",
            "tr3-lease-snapshot-recheck-internal",
            "tr3-lease-snapshot-abort-internal!",
            "tr3-lease-snapshot-finish-tail-internal!",
        )
        for name in names:
            self.assertEqual(CORE.count(f"(define ({name}"), 1)
            for directory in (ROOT / "include", ROOT / "lib", ROOT / "src"):
                for path in directory.rglob("*"):
                    if not path.is_file():
                        continue
                    try:
                        public = path.read_text()
                    except UnicodeDecodeError:
                        continue
                    self.assertNotIn(name, public, str(path))

    def test_restore_authority_remains_separate(self) -> None:
        for name in (
            "tr3-lease-restore-enter-internal!",
            "tr3-lease-restore-recheck-internal",
            "tr3-lease-restore-abort-internal!",
            "tr3-lease-restore-commit-controls-tail-internal!",
            "tr3-lease-restore-finish-tail-internal!",
        ):
            self.assertEqual(CORE.count(f"(define ({name}"), 1)


if __name__ == "__main__":
    unittest.main()
