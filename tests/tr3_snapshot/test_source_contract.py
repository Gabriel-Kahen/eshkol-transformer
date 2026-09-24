"""Private TR3-C snapshot composer source contract."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ESK = (ROOT / "native/tr3_c_snapshot_extension.esk").read_text()
ROOT_ESK = (ROOT / "native/tr3_c_snapshot_root.esk").read_text()
RUNTIME = (ROOT / "tests/tr3_snapshot/runtime_smoke.esk").read_text()
GATE = (ROOT / "scripts/test-tr3-c-snapshot-composer.sh").read_text()


def definition(name: str, next_name: str | None = None) -> str:
    start = ESK.index(f"(define ({name}")
    return ESK[start:] if next_name is None else ESK[start:ESK.index(f"(define ({next_name}", start)]


class SnapshotComposerContract(unittest.TestCase):
    def test_parent_is_exact_preallocated_lease_graph(self) -> None:
        parent = definition("tr3-c-snapshot-parent", "tr3-c-snapshot-byte-copy")
        self.assertIn("tr3-lease-snapshot-parent-tag", parent)
        self.assertIn("trainer 'constructing", parent)
        self.assertEqual(parent.count("(vector #f)"), 3)
        self.assertIn("(make-vector 5 #f)", parent)
        self.assertIn("(vector #f #f) result-cell", parent)
        self.assertIn("(vector-set! parent 1 parent)", parent)

    def test_real_p1_o2_d2_c2_seams_are_the_defaults(self) -> None:
        for binding in (
            "tr3-c-snapshot-p1-call module-state-dict",
            "tr3-c-snapshot-o2-call o2-public-optimizer-state",
            "tr3-c-snapshot-cursor-call d2-token-dataset-cursor",
            "tr3-c-snapshot-compose-call c2-training-state-compose-internal",
            "tr3-c-snapshot-p1-release! state-dict-release!",
            "tr3-c-snapshot-o2-release! o2-public-optimizer-state-release!",
            "tr3-c-snapshot-c2-release! c2-training-state-release-internal!",
        ):
            self.assertIn(binding, ESK)
        self.assertNotIn("define-public", ESK)
        self.assertNotIn("install-public", ESK)

    def test_recheck_precedes_component_snapshot_and_immediate_rooting(self) -> None:
        compose = definition("tr3-c-snapshot-compose!", "tr3-c-trainer-state-internal")
        ordered = (
            "tr3-lease-snapshot-recheck-internal",
            "tr3-c-snapshot-p1-call",
            "(vector-set! (vector-ref parent 4) 0 p1)",
            "tr3-c-snapshot-o2-call",
            "(vector-set! (vector-ref parent 5) 0 o2)",
            "tr3-c-snapshot-controls!",
            "tr3-c-snapshot-compose-call",
            "tr3-c-snapshot-adopt-transfer!",
        )
        positions = [compose.index(item) for item in ordered]
        self.assertEqual(positions, sorted(positions))

    def test_transfer_uses_request_and_exact_live_registry_owner(self) -> None:
        adopt = definition("tr3-c-snapshot-adopt-transfer!", "tr3-c-snapshot-cleanup-call!")
        self.assertIn("(not p1) (not o2) output", adopt)
        self.assertIn("c2-training-state-find output", adopt)
        self.assertIn("(vector-ref state 7) 'live", adopt)
        self.assertIn("(vector-ref state 1)", adopt)
        self.assertIn("(vector-ref state 2)", adopt)
        self.assertLess(adopt.index("parent 4) 0 #f"), adopt.index("parent 6) 0 output"))
        self.assertLess(adopt.index("parent 5) 0 #f"), adopt.index("parent 6) 0 output"))

    def test_controls_are_fresh_and_cross_validated_before_compose(self) -> None:
        controls = definition("tr3-c-snapshot-controls!", "tr3-c-snapshot-adopt-transfer!")
        for required in (
            "tr3-c-snapshot-byte-copy",
            "utf8->string",
            "x1-private-config-canonical",
            "x1-private-config-fingerprint",
            "c2-d2-cursor-pair-internal",
            "x1-canonical-inspect-internal",
            "(vector-ref cursor-state 7)",
            "(vector-ref cursor-state 8)",
            "(vector-ref cursor-state 9)",
            "(vector-ref x1 14)",
            "update-contributions",
            "epoch-contributions",
        ):
            self.assertIn(required, controls)

    def test_abort_clears_cells_in_c2_o2_p1_order_then_unlinks(self) -> None:
        abort = definition("tr3-c-snapshot-abort!", "tr3-c-snapshot-compose!")
        ordered = (
            "(vector-set! parent 3 'aborting)",
            "(vector-ref parent 6) tr3-c-snapshot-c2-release!",
            "(vector-ref parent 5) tr3-c-snapshot-o2-release!",
            "(vector-ref parent 4) tr3-c-snapshot-p1-release!",
            "tr3-lease-snapshot-abort-internal!",
        )
        positions = [abort.index(item) for item in ordered]
        self.assertEqual(positions, sorted(positions))
        release = definition("tr3-c-snapshot-release-cell!", "tr3-c-snapshot-abort!")
        self.assertLess(release.index("(vector-set! cell 0 #f)"), release.index("release owner"))

    def test_success_publishes_before_assignment_only_finish(self) -> None:
        entry = definition("tr3-c-trainer-state-internal")
        ordered = (
            "tr3-lease-snapshot-enter-internal!",
            "(vector-set! record-cell 0 record)",
            "tr3-c-snapshot-compose!",
            "(vector-set! result-cell 0 owner)",
            "(vector-set! (vector-ref parent 6) 0 #f)",
            "(vector-set! parent 3 'dead)",
            "tr3-lease-snapshot-finish-tail-internal!",
        )
        positions = [entry.index(item) for item in ordered]
        self.assertEqual(positions, sorted(positions))

    def test_runtime_covers_normal_negative_ownership_and_retry(self) -> None:
        for witness in (
            "snapshot preserves all 42 tensor payloads and controls",
            "same-thread snapshot reentry is rejected",
            "pre-compose failure consumes O2 and P1 ownership once",
            "pre-compose failure does not publish a result",
            "post-transfer C2 owner is dead after cleanup",
            "post-transfer P1 subowner is released",
            "post-transfer O2 subowner is dead",
            "post-transfer failure does not publish a result",
            "source remains snapshot-ready after cleanup",
        ):
            self.assertIn(witness, RUNTIME)

    def test_private_root_has_one_identity_universe(self) -> None:
        self.assertEqual(ROOT_ESK.count('(load "d2_wave2_root.esk")'), 1)
        self.assertEqual(ROOT_ESK.count('(load "c2_training_state_extension.esk")'), 1)
        self.assertEqual(ROOT_ESK.count('(load "tr3_lease_core_extension.esk")'), 1)
        self.assertEqual(ROOT_ESK.count('(load "tr3_c_snapshot_extension.esk")'), 1)
        self.assertNotIn("tr3_c_joint_restore", ROOT_ESK)

    def test_supported_gate_pins_existing_native_recipe_and_focused_fixture(self) -> None:
        self.assertIn("test-tr3-c-joint-runtime.sh", GATE)
        self.assertIn("tests/tr3_snapshot/runtime_smoke.esk", GATE)
        self.assertIn("tr3_snapshot_runtime", GATE)
        self.assertIn("tests.tr3_snapshot.test_source_contract", GATE)
        self.assertIn("tests.tr3_lease.test_snapshot_authority_contract", GATE)
        self.assertIn("composer_gate_sha256", GATE)
        self.assertIn("inherited_native_gate_sha256", GATE)


if __name__ == "__main__":
    unittest.main()
