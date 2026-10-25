"""Source boundary for the private C4 workspace candidate."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "native/g3r_c4_workspace_ownership_extension.esk").read_text()
NATIVE = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
FIXTURE = (ROOT / "tests/g3c4/workspace_ownership_source_fixture.esk").read_text()


class WorkspaceOwnershipSource(unittest.TestCase):
    def test_create_uses_train_authority_and_rechecks_after_stage(self):
        self.assertIn("g3r-c4-trainable-owner-association-status-internal", SOURCE)
        self.assertIn("g3r-c4-workspace-native-no-context", SOURCE)
        self.assertIn("g3r-c4-workspace-native-no-lease", SOURCE)
        self.assertIn("(g3r-c4-workspace-create-preflight model operation)", SOURCE)
        self.assertIn("g3r-c4-workspace-native-prepare owner native nonce", SOURCE)
        self.assertIn("g3r-c4-workspace-native-commit owner native nonce", SOURCE)

    def test_live_path_uses_lifetime_tree_and_original_bindings(self):
        self.assertIn("module-c4-lifetime-topology-status-internal", SOURCE)
        self.assertIn("module-c4-train-topology-status-internal", SOURCE)
        self.assertIn("(eq? carrier (vector-ref carriers index))", SOURCE)
        self.assertIn("g3r-c4-workspace-native-binding", SOURCE)
        self.assertIn("(eq? (vector-ref entry 10) expected)", SOURCE)

    def test_native_edge_is_private_and_releases_allocation(self):
        self.assertIn("#ifdef ET_G3R_C4_WORKSPACE_PRIVATE", NATIVE)
        self.assertIn("owner->active = workspace", NATIVE)
        self.assertIn("owner->active = NULL;\n  *link = workspace->next;\n  free(workspace)", NATIVE)
        self.assertIn("et_g3c4_workspace_admit_live", NATIVE)

    def test_negative_fixture_covers_fault_and_eval_drift(self):
        for phrase in (
            "after-stage rollback", "after-shell rollback",
            "before-commit rollback", "native edge commit failure",
            "after-allocation changed binding recheck",
            "changed live carrier", "swapped current handle",
            "eval reset", "eval release",
            "eval lifetime original-handle rejection",
            "eval lifetime original-carrier rejection",
            "second workspace", "forged closure shell",
            "dead shell idempotent", "independent model coexistence",
        ):
            self.assertIn(phrase, FIXTURE)


if __name__ == "__main__":
    unittest.main()
