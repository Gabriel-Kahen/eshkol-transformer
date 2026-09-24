"""Private TR3-C joint composer source and transaction contract."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ESK = (ROOT / "native/tr3_c_joint_restore_extension.esk").read_text()
ROOT_ESK = (ROOT / "native/tr3_c_joint_restore_root.esk").read_text()


def definition(name: str, next_name: str | None = None) -> str:
    start = ESK.index(f"(define ({name}")
    if next_name is None:
        return ESK[start:]
    return ESK[start : ESK.index(f"(define ({next_name}", start)]


class JointRestoreContract(unittest.TestCase):
    def test_exact_parent_shape_and_lease_owned_slots(self) -> None:
        parent = definition("tr3-c-joint-parent", "tr3-c-joint-controls-copy")
        self.assertIn("(make-vector 14 #f)", parent)
        self.assertIn("(make-vector 5 #f)", parent)
        self.assertIn("(make-vector 6 #f)", parent)
        self.assertIn("tr3-lease-restore-parent-tag", parent)
        self.assertRegex(parent, r"trainer 'constructing #f\s+p1-cells")
        self.assertIn("(vector #f) (vector #f) controls #f #f cleanup", parent)
        self.assertIn("(vector-set! parent 1 parent)", parent)

    def test_detached_source_closes_before_any_receiver_plan(self) -> None:
        stage = definition(
            "tr3-c-joint-stage-source!", "tr3-c-joint-checked-product"
        )
        self.assertIn("c2-training-state-borrow-begin-internal", stage)
        self.assertIn("state-dict-c2-owned-projection-internal", stage)
        self.assertEqual(stage.count("tr3-c-joint-install-clone!"), 1)
        self.assertEqual(stage.count("tr3-c-joint-install-moment!"), 2)
        staged = stage.index("(vector-set! parent 3 'source-staged)")
        ended = stage.index("(c2-training-state-borrow-end-internal borrow)")
        closed = stage.index("(vector-set! parent 3 'source-closed)")
        self.assertLess(staged, ended)
        self.assertLess(ended, closed)
        for receiver_call in (
            "tr3-d2-restore-prepare-internal",
            "tr3-c-restore-o2-prepare-internal",
            "tr3-c-restore-i2-create-internal",
        ):
            self.assertNotIn(receiver_call, stage)

    def test_fixed_14_prefix_and_exact_o2_authority(self) -> None:
        prepare = definition(
            "tr3-c-joint-prepare-receiver!", "tr3-c-joint-release-cell!"
        )
        order = [
            "tr3-d2-restore-prepare-internal",
            "tr3-c-restore-o2-prepare-internal",
            "tr3-c-restore-i2-create-internal",
            "tr3-c-restore-i2-append-parameter-internal!",
            "tr3-c-restore-o2-append-internal!",
            "tr3-lease-restore-recheck-internal",
            "tr3-d2-restore-check-internal",
            "tr3-c-restore-o2-check-internal",
            "tr3-c-restore-i2-prepare-internal!",
            "(vector-set! parent 3 'sealed)",
        ]
        positions = [prepare.index(item) for item in order]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("(if (= index 14)", prepare)
        self.assertIn("(vector-ref o2-ledger 2)", prepare)
        self.assertIn("(vector-set! (vector-ref parent 7) 0 native-plan)", prepare)
        self.assertIn("(if (vector-ref request 6)", prepare)
        self.assertLess(
            prepare.index("(if (vector-ref request 6)"),
            prepare.index("(if (not (vector-ref outcome 0))"),
        )

    def test_cross_state_validation_is_complete_and_checked(self) -> None:
        validate = definition(
            "tr3-c-joint-validate-controls!", "tr3-c-joint-prepare-receiver!"
        )
        for required in (
            "c2-d2-cursor-pair-internal",
            "x1-canonical-inspect-internal",
            "tr3-c-joint-checked-product",
            "tr3-c-joint-checked-add",
            "(vector-ref cursor-state 7)",
            "(vector-ref cursor-state 8)",
            "(vector-ref cursor-state 9)",
            "(vector-ref x1 14)",
            "(not (= update-contributions reachable))",
            "(> update-contributions tokens)",
            "(vector-ref o2-projection 3)",
        ):
            self.assertIn(required, validate)

    def test_abort_order_preserves_phase_dependent_authority(self) -> None:
        abort = definition("tr3-c-joint-abort!", "tr3-c-joint-commit-tail!")
        order = [
            "(vector-set! parent 3 'aborting)",
            "c2-training-state-borrow-end-internal",
            "tr3-c-restore-i2-abort-internal!",
            "tr3-d2-restore-abort-internal!",
            "tr3-c-restore-o2-abort-internal!",
            "tr3-c-joint-release-cell!",
            "(vector-set! parent 3 'dead)",
            "tr3-lease-restore-abort-internal!",
        ]
        positions = [abort.index(item) for item in order]
        self.assertEqual(positions, sorted(positions))
        self.assertLess(
            abort.index("state-dict-c2-owned-tensor-borrow-end-internal"),
            abort.index("c2-training-state-borrow-end-internal"),
        )
        self.assertGreaterEqual(abort.count("tr3-c-joint-access-clear?"), 3)
        self.assertGreaterEqual(abort.count("tr3-lease-fail-stop 134"), 3)
        self.assertNotIn("'committing", abort)

    def test_irreversible_tail_is_fixed_and_transitively_infallible(self) -> None:
        tail = definition(
            "tr3-c-joint-commit-tail!", "tr3-c-joint-restore-internal!"
        )
        self.assertEqual(
            tail.count("tr3-c-restore-owned-release-tail-internal!"), 14
        )
        expected = [
            "(vector-set! parent 3 'committing)",
            "tr3-c-restore-i2-commit-tail-internal!",
            "tr3-c-restore-o2-commit-tail-internal!",
            "tr3-c-restore-o2-finalize-tail-internal!",
            "tr3-d2-restore-commit-internal!",
            "tr3-lease-restore-commit-controls-tail-internal!",
            "(vector-set! parent 3 'committed)",
            "(vector-set! parent 3 'dead)",
            "tr3-lease-restore-finish-tail-internal!",
        ]
        positions = [tail.index(item) for item in expected]
        self.assertEqual(positions, sorted(positions))
        for forbidden in (
            "guard",
            "raise",
            "m3t-fail",
            "o2-fail",
            "tr3-c-joint-fail",
            "tr3-lease-restore-recheck-internal",
            "tr3-c-restore-require-",
            "tr3-c-restore-find-identity",
            "i2-carrier-preflight",
            "state-dict-",
            "c2-training-state-",
        ):
            self.assertNotIn(forbidden, tail)
        self.assertNotRegex(tail, r"\(let\s+loop|\(let\s+[a-zA-Z0-9-]+\s*\(")

    def test_only_one_private_entry_and_no_public_surface(self) -> None:
        entries = re.findall(r"\(define \((tr3-c-joint-[^\s()]+)", ESK)
        self.assertIn("tr3-c-joint-restore-internal!", entries)
        self.assertNotIn("(provide", ESK)
        self.assertNotRegex(ESK, r"\(define \(trainer-load-state!")
        self.assertNotRegex(ESK, r"\bet_tr3_c_(?!private)")

    def test_recovery_handler_precedes_lease_publication(self) -> None:
        entry = definition("tr3-c-joint-restore-internal!")
        self.assertLess(
            entry.index("(guard (caught"),
            entry.index("tr3-lease-restore-enter-internal!"),
        )
        self.assertIn("(record-cell (vector #f))", entry)
        self.assertIn("(tr3-lease-fail-stop 134)", entry)
        self.assertIn("(transformer-error-category", entry)
        self.assertIn("'internal", entry)

    def test_root_uses_existing_components_once(self) -> None:
        for source in (
            "d2_wave2_root.esk",
            "c2_training_state_extension.esk",
            "tr3_lease_core_extension.esk",
            "tr3_c_d2_restore_extension.esk",
            "tr3_c_restore_bindings_extension.esk",
            "tr3_c_joint_restore_extension.esk",
        ):
            self.assertEqual(ROOT_ESK.count(source), 1)
        self.assertNotIn("c2_wave2_root.esk", ROOT_ESK)
        self.assertNotIn("tr3_lease_root.esk", ROOT_ESK)


class FailurePrefixModel(unittest.TestCase):
    """Exhaust the bounded publication model independently of source syntax."""

    STEPS = 14 + 28 + 1 + 1 + 1 + 14 + 1 + 1 + 1

    def run_prefix(self, fail_at: int | None) -> tuple[str, int, int, int]:
        phase = "constructing"
        p1 = moments = live = 0
        for ordinal in range(self.STEPS):
            if ordinal == fail_at:
                p1 = moments = live = 0
                return "dead", p1, moments, live
            if ordinal < 14:
                p1 += 1
            elif ordinal < 42:
                moments += 1
            elif ordinal == 42:
                phase = "source-staged"
            elif ordinal == 43:
                phase = "source-closed"
            elif ordinal == 44:
                phase = "receiver-prepared"
            elif ordinal < 59:
                pass
            elif ordinal == 59:
                phase = "sealed"
            elif ordinal == 60:
                live = 42
                phase = "committing"
            else:
                phase = "dead"
        return phase, p1, moments, live

    def test_every_precommit_prefix_rolls_back_without_publication(self) -> None:
        for fail_at in range(60):
            self.assertEqual(self.run_prefix(fail_at), ("dead", 0, 0, 0))

    def test_retention_horizons_have_no_live_authority_and_known_tombstones(self) -> None:
        for horizon in (1024, 8192):
            live = 0
            for ordinal in range(horizon):
                _, p1, moments, writes = self.run_prefix(ordinal % 60)
                live += p1 + moments + writes
            self.assertEqual(live, 0)
            # Accepted I2/O2 terminal control retention is linear: 64 + 1464.
            self.assertEqual(horizon * 1528, {1024: 1_564_672, 8192: 12_517_376}[horizon])


if __name__ == "__main__":
    unittest.main()
