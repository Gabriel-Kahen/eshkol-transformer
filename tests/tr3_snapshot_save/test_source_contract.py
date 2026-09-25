"""Structural checks for the private snapshot-to-C2 SAVE ownership seam."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SEAM = (ROOT / "native/tr3_c_snapshot_save_extension.esk").read_text()
AGGREGATE = (ROOT / "native/tr3_c_snapshot_save_root.esk").read_text()
FIXTURE = (ROOT / "tests/tr3_snapshot_save/runtime_suffix.esk").read_text()
GATE = (ROOT / "scripts/test-tr3-c-snapshot-save.sh").read_text()


class SnapshotSaveSourceContract(unittest.TestCase):
    def test_exact_accepted_dependencies_and_private_root(self):
        self.assertIn("tr3-c-snapshot-save-call c2-checkpoint-save-internal!", SEAM)
        self.assertIn(
            "tr3-c-snapshot-save-release! c2-training-state-release-internal!", SEAM
        )
        for name in (
            "tr3_c_snapshot_root.esk",
            "c2_o2_encode_extension.esk",
            "c2_model_encode_extension.esk",
            "c2_persistence_policy_extension.esk",
            "c2_checkpoint_save_extension.esk",
            "tr3_c_snapshot_save_extension.esk",
        ):
            self.assertEqual(AGGREGATE.count(f'(load "{name}")'), 1)
        self.assertNotIn("define-public", SEAM + AGGREGATE)

    def test_cell_is_authority_across_exception_and_release(self):
        self.assertLess(
            SEAM.index("(tr3-c-trainer-state-internal trainer owner-cell)"),
            SEAM.index("(tr3-c-snapshot-save-call owner path policy options)"),
        )
        self.assertLess(
            SEAM.index("(vector-set! owner-cell 0 #f)"),
            SEAM.index("(tr3-c-snapshot-save-release!"),
        )
        self.assertIn("(vector-set! failure 1 caught)", SEAM)
        self.assertIn("(vector-set! cleanup 1 caught)", SEAM)
        self.assertIn("(raise (vector-ref failure 1))", SEAM)

    def test_runtime_uses_real_writer_and_injected_write_failures(self):
        for text in (
            "byte-identical C2 checkpoints",
            "malformed path releases snapshot owner once",
            "malformed options release snapshot owner once",
            "failed precommit publishes no checkpoint",
            "postcommit fault leaves a complete checkpoint",
            "same trainer retries after writer failures",
        ):
            self.assertIn(text, FIXTURE)
        self.assertIn("-DET_CHECKPOINT_IO_TESTING", GATE)
        self.assertIn("c2_checkpoint_save_bridge.c", GATE)


if __name__ == "__main__":
    unittest.main()
