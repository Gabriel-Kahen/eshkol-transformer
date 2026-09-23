"""Fail-closed source and package checks for the TR3 lease aggregate."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "native"
CORE = NATIVE / "tr3_lease_core_extension.esk"
AGGREGATE = NATIVE / "tr3_lease_root.esk"
GATES = NATIVE / "tr3_lease_public_gates_extension.esk"


class LeaseSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.core = CORE.read_text()
        cls.aggregate = AGGREGATE.read_text()
        cls.gates = GATES.read_text()

    def test_single_private_registry_and_create_entry(self) -> None:
        self.assertEqual(
            len(re.findall(r"\(define\s+tr3-trainers(?:\s|\))", self.core)), 1
        )
        self.assertEqual(
            len(re.findall(r"\(define\s+\(tr3-lease-create-internal\b", self.core)),
            1,
        )
        self.assertIn("tr3-acquisition-active?", self.core)
        self.assertIn("tr3-p1-fixed-capture-internal", self.core)
        self.assertIn("tr3-p1-fixed-recheck-internal", self.core)

    def test_root_composes_one_lineage_and_installs_o2_hook(self) -> None:
        loads = re.findall(r'^\(load\s+"([^"]+)"\)', self.aggregate, re.MULTILINE)
        self.assertEqual(loads.count("tr3_lease_core_extension.esk"), 1)
        self.assertEqual(loads.count("tr3_lease_public_gates_extension.esk"), 1)
        self.assertEqual(len(loads), len(set(loads)), "aggregate repeats a source load")
        self.assertLess(
            loads.index("tr3_lease_core_extension.esk"),
            loads.index("tr3_lease_public_gates_extension.esk"),
        )
        self.assertNotIn("tr3_c_d2_restore_extension.esk", loads)
        self.assertIn("o2-install-tr3-create-overlap-check-internal!", self.aggregate)
        self.assertIn("tr3-o2-create-overlap-check-internal", self.aggregate)

    def test_core_binds_the_direct_d2_idle_probe(self) -> None:
        self.assertIn("et_tr3_c_d2_dataset_idle_preflight_v1", self.core)

    def test_runtime_destroys_the_successful_acquisition_region(self) -> None:
        runtime = (ROOT / "tests" / "tr3_lease" / "lease_runtime.esk").read_text()
        self.assertIn("(with-region ('tr3-lease-acquisition", runtime)
        self.assertIn("ESHKOL_ARENA_POISON=1", (ROOT / "scripts" / "test-tr3-lease.sh").read_text())

    def test_acquisition_has_initial_and_final_overlap_checks(self) -> None:
        create = self.core[self.core.index("(define (tr3-lease-create-internal") :]
        recheck = self.core[
            self.core.index("(define (tr3-lease-recheck!") :
            self.core.index("(define (tr3-lease-create-internal")
        ]
        for allocating_helper in (
            "tr3-lease-config!",
            "tr3-lease-tokenizer-dataset!",
            "x1-private-config-canonical",
            "x1-private-config-fingerprint",
            "d2-tokenizer-identity",
        ):
            self.assertNotIn(allocating_helper, recheck)
        self.assertIn("(tr3-lease-tuple-overlap?", create)
        self.assertIn("(tr3-lease-tuple-overlap?", recheck)
        self.assertIn("tr3-p1-fixed-recheck-internal", recheck)
        final_head = recheck.rindex("(vector-ref tr3-trainers 0)")
        for observation in (
            "tr3-p1-fixed-recheck-internal",
            "tr3-lease-m3-idle!",
            "tr3-lease-o2-idle!",
            "tr3-lease-d2-idle!",
        ):
            self.assertLess(recheck.index(observation), final_head, observation)
        final_recheck = create.index("(tr3-lease-recheck!")
        self.assertLess(create.index("tr3-p1-fixed-capture-internal"), final_recheck)
        commit = re.search(
            r"\(vector-set!\s+tr3-trainers\s+0\s+canonical-successor\)", create
        )
        self.assertIsNotNone(commit)
        self.assertLess(final_recheck, commit.start())
        staging_clear = create.rindex("(vector-set! tr3-trainers 1 #f)")
        self.assertLess(final_recheck, staging_clear)
        self.assertLess(staging_clear, commit.start())

    def test_phase_vocabulary_and_fixed_admissions_are_present(self) -> None:
        for phase in (
            "idle",
            "step",
            "commit",
            "evaluate",
            "snapshotting",
            "restoring",
        ):
            self.assertRegex(self.core, rf"'{phase}\b")
        for name in (
            "tr3-lease-step-model-internal",
            "tr3-lease-step-optimizer-internal",
            "tr3-lease-step-dataset-internal",
            "tr3-lease-commit-optimizer-internal",
            "tr3-lease-evaluation-modes-internal",
            "tr3-lease-snapshot-internal",
            "tr3-lease-restore-internal",
        ):
            self.assertIn(f"(define ({name}", self.core)

    def test_gate_wrappers_are_lexically_prefixed(self) -> None:
        definitions = re.findall(r"\(define\s+\(([^\s()]+)", self.gates)
        self.assertTrue(definitions)
        for name in definitions:
            self.assertTrue(name.startswith("tr3-public-"), name)

    def test_lease_authority_is_not_publicly_packaged(self) -> None:
        private_spellings = (
            "tr3-trainers",
            "tr3-lease-token",
            "tr3-lease-create-internal",
            "tr3-lease-step-model-internal",
            "tr3-lease-restore-internal",
        )
        public_roots = (
            ROOT / "include",
            ROOT / "lib",
            ROOT / "src" / "eshkol_transformer",
        )
        for directory in public_roots:
            for path in directory.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    text = path.read_text()
                except UnicodeDecodeError:
                    continue
                for spelling in private_spellings:
                    self.assertNotIn(spelling, text, str(path))


if __name__ == "__main__":
    unittest.main()
