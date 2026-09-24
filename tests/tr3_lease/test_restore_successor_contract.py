"""Source contract for the private TR3-C lease restore successor."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "native" / "tr3_lease_core_extension.esk"


class RestoreSuccessorContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.core = CORE.read_text()

    def definition(self, name: str, next_name: str) -> str:
        start = self.core.index(f"(define ({name}")
        end = self.core.index(f"(define ({next_name}", start)
        return self.core[start:end]

    def test_record_appends_exact_control_authority(self) -> None:
        self.assertIn("(define tr3-lease-record-length 22)", self.core)
        self.assertIn("'tr3-lease-record-v2", self.core)
        record = re.search(
            r"\(vector tr3-lease-record-tag #f shell 'idle #f token resolved"
            r"[\s\S]*?initial-cursor-bytes initial-rng 0 0 0\)\)",
            self.core,
        )
        self.assertIsNotNone(record)
        self.assertIn("(not (tr3-lease-controls-valid? record))", self.core)

    def test_creation_owns_a_nonterminal_current_cursor_and_seeded_rng(self) -> None:
        initial = self.definition(
            "tr3-lease-initial-cursor", "tr3-lease-cursor-position!"
        )
        self.assertIn("d2-core-cursor-snapshot", initial)
        self.assertIn("d2-core-cursor-decode", initial)
        self.assertIn("(>= current total)", initial)
        self.assertIn("'trainer-create", initial)
        create = self.definition(
            "tr3-lease-create-internal", "tr3-lease-authenticate"
        )
        self.assertIn("(x1-private-config-ref resolved 'run.seed)", create)
        self.assertIn("(vector 'philox4x32-10 1", create)
        self.assertLess(
            create.index("(tr3-lease-initial-cursor"),
            create.index("(vector-set! tr3-trainers 1 staging)"),
        )
        self.assertIn("initial-total initial-current", create)

    def test_restore_operations_are_narrow_and_private(self) -> None:
        names = (
            "tr3-lease-restore-enter-internal!",
            "tr3-lease-restore-recheck-internal",
            "tr3-lease-restore-abort-internal!",
            "tr3-lease-restore-commit-controls-tail-internal!",
            "tr3-lease-restore-finish-tail-internal!",
        )
        for name in names:
            self.assertEqual(self.core.count(f"(define ({name}"), 1, name)
            for directory in (ROOT / "include", ROOT / "lib", ROOT / "src"):
                for path in directory.rglob("*"):
                    if path.is_file():
                        try:
                            public = path.read_text()
                        except UnicodeDecodeError:
                            continue
                        self.assertNotIn(name, public, str(path))

    def test_enter_admits_before_assignment_only_publication(self) -> None:
        enter = self.definition(
            "tr3-lease-restore-enter-internal!",
            "tr3-lease-restore-recheck-internal",
        )
        admission = enter.index("(tr3-lease-restore-enter-admission!")
        active = enter.index("(vector-set! record 4 parent)")
        phase = enter.index("(vector-set! record 3 'restoring)")
        self.assertLess(admission, active)
        self.assertLess(active, phase)
        tail = enter[active:]
        for forbidden in ("(if ", "(guard ", "m3t-fail", "o2-native-"):
            self.assertNotIn(forbidden, tail)

    def test_restore_o2_admission_uses_trainer_count_and_shared_busy_probe(self) -> None:
        restore_o2 = self.definition(
            "tr3-lease-o2-restore-idle!", "tr3-lease-controls-valid?"
        )
        self.assertIn("(vector-ref record 20)", restore_o2)
        self.assertIn("o2-native-optimizer-completed-updates", restore_o2)
        self.assertIn("o2-native-optimizer-require-absent-gradients", restore_o2)
        self.assertNotIn("tr3-lease-o2-idle!", restore_o2)

    def test_parent_is_exact_private_self_bound_authority_graph(self) -> None:
        self.assertIn("(define tr3-lease-restore-parent-length 13)", self.core)
        enter = self.definition(
            "tr3-lease-restore-enter-internal!",
            "tr3-lease-restore-recheck-internal",
        )
        self.assertIn("tr3-lease-restore-parent-empty! parent trainer operation", enter)
        parent = self.definition(
            "tr3-lease-restore-parent-header!",
            "tr3-lease-restore-parent-empty!",
        )
        for witness in (
            "tr3-lease-restore-parent-length",
            "tr3-lease-restore-parent-tag",
            "(eq? (vector-ref parent 1) parent)",
            "(eq? (vector-ref parent 2) trainer)",
            "(eq? (vector-ref parent 3) phase)",
        ):
            self.assertIn(witness, parent)

    def test_seal_binds_prepared_o2_and_parent_controls_without_idle_probe(self) -> None:
        recheck = self.definition(
            "tr3-lease-restore-recheck-internal",
            "tr3-lease-restore-abort-internal!",
        )
        self.assertIn("expected-o2-plan", recheck)
        self.assertIn("tr3-lease-restore-parent-seal!", recheck)
        self.assertIn("tr3-lease-m3-idle!", recheck)
        self.assertIn("tr3-lease-d2-idle!", recheck)
        self.assertNotIn("tr3-lease-o2-restore-idle!", recheck)
        self.assertNotIn("tr3-lease-restore-enter-admission!", recheck)
        seal = self.definition(
            "tr3-lease-restore-parent-seal!",
            "tr3-lease-restore-enter-internal!",
        )
        self.assertIn("'receiver-prepared", seal)
        self.assertIn("(vector-ref parent 7)", seal)
        self.assertIn("tr3-lease-copied-controls-valid? controls record", seal)

    def test_abort_uses_minimal_authentication_and_unlinks_after_checks(self) -> None:
        abort = self.definition(
            "tr3-lease-restore-abort-internal!",
            "tr3-lease-restore-commit-controls-tail-internal!",
        )
        self.assertIn("tr3-lease-authenticate-minimal", abort)
        self.assertNotIn("tr3-lease-authenticate record", abort)
        for phase in (
            "'constructing",
            "'source-staged",
            "'source-closed",
            "'receiver-prepared",
            "'sealed",
            "'aborting",
            "'dead",
        ):
            self.assertIn(phase, abort)
        self.assertNotIn("'committing", abort)
        self.assertNotIn("'committed", abort)
        check = abort.index("(if (or")
        active = abort.index("(vector-set! enrolled 4 #f)")
        phase = abort.index("(vector-set! enrolled 3 'idle)")
        self.assertLess(check, active)
        self.assertLess(active, phase)

    def test_commit_and_finish_are_split_assignment_only_tails(self) -> None:
        commit = self.definition(
            "tr3-lease-restore-commit-controls-tail-internal!",
            "tr3-lease-restore-finish-tail-internal!",
        )
        self.assertEqual(
            commit.splitlines()[0],
            "(define (tr3-lease-restore-commit-controls-tail-internal! record parent)",
        )
        self.assertIn("(let* ((copied-controls (vector-ref parent 8))", commit)
        stores = [commit.index(f"(vector-set! record {slot}") for slot in range(17, 22)]
        self.assertEqual(stores, sorted(stores))
        self.assertNotIn("(vector-set! record 4 #f)", commit)
        self.assertLess(commit.index("tr3-lease-fail-stop"), stores[0])
        finish = self.definition(
            "tr3-lease-restore-finish-tail-internal!", "tr3-lease-phase!"
        )
        self.assertLess(
            finish.index("(vector-set! record 4 #f)"),
            finish.index("(vector-set! record 3 'idle)"),
        )
        for tail in (commit, finish):
            for forbidden in (
                "tr3-lease-authenticate",
                "tr3-lease-restore-enter-admission!",
                "m3t-fail",
                "o2-native-",
                "d2-core-",
                "(guard ",
            ):
                self.assertNotIn(forbidden, tail)
            self.assertIn("tr3-lease-fail-stop", tail)


if __name__ == "__main__":
    unittest.main()
