"""Private C2 LOAD to joint live restore source contract."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SEAM = (ROOT / "native/tr3_c_checkpoint_restore_extension.esk").read_text()
AGGREGATE = (ROOT / "native/tr3_c_checkpoint_restore_root.esk").read_text()
RUNTIME = (ROOT / "tests/tr3_checkpoint_restore/runtime_suffix.esk").read_text()
GATE = (ROOT / "scripts/test-tr3-c-checkpoint-restore.sh").read_text()


class CheckpointRestoreSourceContract(unittest.TestCase):
    def test_genuine_c2_load_and_joint_restore(self):
        self.assertIn("tr3-c-checkpoint-restore-load c2-public-checkpoint-load", SEAM)
        self.assertIn("tr3-c-checkpoint-restore-apply! tr3-c-joint-restore-internal!", SEAM)
        self.assertIn("tr3-c-checkpoint-restore-release! c2-training-state-release-internal!", SEAM)
        self.assertLess(
            SEAM.index("(tr3-c-checkpoint-restore-load path policy report)"),
            SEAM.index("(tr3-c-checkpoint-restore-apply! trainer owner)"),
        )

    def test_release_drains_exact_loaded_owner_on_all_returns(self):
        self.assertLess(
            SEAM.index("(vector-set! owner-cell 0 #f)"),
            SEAM.index("(tr3-c-checkpoint-restore-release!"),
        )
        self.assertIn("(vector-set! failure 1 caught)", SEAM)
        self.assertIn("(vector-set! cleanup 1 caught)", SEAM)
        self.assertIn("(raise (vector-ref failure 1))", SEAM)

    def test_aggregate_one_identity_universe(self):
        for name in (
            "d2_wave2_root.esk",
            "tr3_c_snapshot_extension.esk",
            "k2_wave2_extension.esk",
            "c2_checkpoint_load_extension.esk",
            "c2_public_extension.esk",
            "tr3_c_joint_restore_extension.esk",
            "tr3_c_checkpoint_restore_extension.esk",
        ):
            self.assertEqual(AGGREGATE.count(f'(load "{name}")'), 1)
        self.assertNotIn("define-public", AGGREGATE + SEAM)

    def test_compiled_fixture_covers_corruption_and_exact_restore(self):
        for label in (
            "checksum rejection preserves receiver",
            "torn checkpoint preserves receiver",
            "X1 compatibility rejection preserves receiver",
            "forged trainer releases loaded owner once",
            "forged K2 report preserves source trainer",
            "successful restore releases loaded owner once",
            "all 42 restored tensor bytes and controls match saved source",
        ):
            self.assertIn(label, RUNTIME)
        for source in (
            "c2_checkpoint_load_bridge.c",
            "c2_checkpoint_reader.c",
            "k2_capabilities.c",
        ):
            self.assertIn(source, GATE)


if __name__ == "__main__":
    unittest.main()
