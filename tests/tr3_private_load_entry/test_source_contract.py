"""Accepted private result-cell entry contract and genuine gate coverage."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
ENTRY = (ROOT / "native/tr3_c_trainer_load_state_extension.esk").read_text()
AGGREGATE = (ROOT / "native/tr3_c_trainer_load_state_root.esk").read_text()
RUNTIME = (ROOT / "tests/tr3_private_load_entry/runtime_suffix.esk").read_text()
GATE = (ROOT / "scripts/test-tr3-c-trainer-load-state.sh").read_text()


class TrainerLoadStateEntryContract(unittest.TestCase):
    def test_exact_one_slot_admission_before_joint_restore(self):
        for clause in (
            "(not (vector? result-cell))",
            "(not (= (vector-length result-cell) 1))",
            "(vector-ref result-cell 0)",
            "'invalid-argument 'trainer-load-state!",
        ):
            self.assertIn(clause, ENTRY)
        self.assertLess(
            ENTRY.index("restore result cell must be empty and exact"),
            ENTRY.index("(tr3-c-joint-restore-internal! trainer state)"),
        )

    def test_success_publishes_only_after_authentic_joint_transaction(self):
        self.assertLess(
            ENTRY.index("(tr3-c-joint-restore-internal! trainer state)"),
            ENTRY.index("(vector-set! result-cell 0 #t)"),
        )
        self.assertEqual(AGGREGATE.count('(load "tr3_c_joint_restore_root.esk")'), 1)
        self.assertEqual(
            AGGREGATE.count('(load "tr3_c_trainer_load_state_extension.esk")'), 1
        )
        self.assertNotIn("define-public", ENTRY + AGGREGATE)

    def test_genuine_runtime_covers_admission_rollback_and_exact_image(self):
        for label in (
            "nonvector result cell is rejected before restore",
            "occupied result cell is rejected before restore",
            "forged detached source rejects",
            "busy detached source rejects",
            "rollback preserves all receiver tensor/control bytes",
            "rollback leaves detached source live and unchanged",
            "success restores all 42 tensor bytes and controls",
        ):
            self.assertIn(label, RUNTIME)
        self.assertIn("test-tr3-c-joint-runtime.sh", GATE)
        self.assertIn("tr3_c_trainer_load_state_extension.esk", GATE)


if __name__ == "__main__":
    unittest.main()
