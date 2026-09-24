"""Fail-closed source and package checks for the TR3 lease aggregate."""

from __future__ import annotations

import hashlib
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

    def test_failure_witness_uses_real_post_staging_census(self) -> None:
        witness = (
            ROOT / "tests" / "tr3_lease_failure" / "lease_failure_runtime.esk"
        ).read_text()
        shim = (
            ROOT / "tests" / "tr3_lease_failure" / "lease_failure_shim.cpp"
        ).read_text()
        self.assertNotIn("tr3-failure-after-recheck", witness)
        self.assertNotIn("(define (tr3-p1-fixed-recheck-internal", witness)
        self.assertIn("lf-poststage-arm", witness)
        self.assertIn("lf-poststage-finish", witness)
        self.assertIn("(guard (raised (#t raised))", witness)
        self.assertNotIn("(define (caught?", witness)
        self.assertIn('(string=? case-name "object")', witness)
        self.assertIn('(string=? case-name "all")', witness)
        self.assertIn("(+ 1 (handler-scope tuple (+ depth 1)))", witness)
        self.assertIn("real_region_allocate_quiet", shim)
        self.assertIn("__wrap_arena_allocate_ad_node_with_header", shim)
        self.assertIn("real bounded object allocation did not fail", shim)
        self.assertIn("transformer-error-category raised-value", witness)
        self.assertIn("transformer-error-operation raised-value", witness)
        self.assertIn("'config-fingerprint", witness)
        self.assertIn("post-staging interval attempted an allocation", shim)
        gate = (ROOT / "scripts" / "test-tr3-lease-failures.sh").read_text()
        self.assertIn("/out/case-status.tsv", gate)
        self.assertIn('${evidence_dir}/inputs', gate)
        self.assertIn("container_image_id", gate)
        self.assertIn("--allocation-class", gate)
        self.assertIn('"${container_id}" bash -lc', gate)
        self.assertIn('git -C "${PROJECT_ROOT}" diff --quiet', gate)
        self.assertIn("trainer_source_commit", gate)
        self.assertIn("trainer_source_tree", gate)
        self.assertIn("--production-base", gate)
        self.assertIn("successor checkout must be clean", gate)
        self.assertIn("internal/d2/lib/d2_dataset.esk", gate)
        self.assertIn("tests/d2/public_errors_runtime.esk", gate)
        self.assertIn("artifact_manifest_relative_path", gate)
        self.assertIn("llvm-config-21 --version", gate)
        self.assertIn("O2 diagnostics require reviewed byte-count and SHA pins", gate)
        self.assertIn("--diagnostic-only", gate)
        self.assertIn('test ! -s "${evidence_dir}/compile.stderr"', gate)
        match = re.search(
            r"^expected_runner_self_sha256=([0-9a-f]{64})$", gate, re.M
        )
        self.assertIsNotNone(match)
        normalized = gate.replace(
            match.group(0), "expected_runner_self_sha256=__SELF__", 1
        )
        self.assertEqual(
            hashlib.sha256(normalized.encode()).hexdigest(), match.group(1)
        )
        def read_manifest(name: str) -> dict[str, str]:
            lines = (ROOT / "tests" / "tr3_lease" / name).read_text().splitlines()
            entries = [line.split("\t", 1) for line in lines]
            self.assertTrue(all(len(entry) == 2 for entry in entries))
            self.assertEqual(len({key for key, _ in entries}), len(entries))
            return dict(entries)

        functional = read_manifest("runtime_candidate.tsv")
        failure = read_manifest("failure_runtime_candidate.tsv")
        self.assertEqual(
            functional["trainer_source_commit"],
            "3f98cd3f316afa2ab44574e0a85cdbf8b651e31c",
        )
        self.assertEqual(
            functional["trainer_source_tree"],
            "7a64228b447fbfd3f31a7ac42b6d0e588aa9eb34",
        )
        self.assertEqual(
            functional["trainer_source_commit"], failure["trainer_source_commit"]
        )
        self.assertEqual(
            functional["trainer_source_tree"], failure["trainer_source_tree"]
        )
        self.assertEqual(
            functional["runner_sha256"],
            "d5c23f1a59bf8fa96fd54fe4f1e47ce9ae903db2dd0f93408f41345ab34014fa",
        )
        self.assertEqual(
            functional["lost_runner_sha256"],
            "4a0e6303f7b85ed06fb753b52b62155235a3a77bca6c32aeb17241a28ed80be1",
        )
        self.assertEqual(
            functional["runner_role"], "functional_recovered_final81298"
        )
        self.assertEqual(functional["lost_runner_status"], "missing")
        self.assertNotEqual(
            functional["runner_sha256"], functional["lost_runner_sha256"]
        )
        self.assertEqual(
            failure["runner_sha256"],
            "5b6c5cae8872330cf59f47f84535fb9acf0242ece001eb544112ac133280b918",
        )
        self.assertEqual(
            failure["runner_role"], "allocation_prefix_production_off"
        )
        self.assertNotEqual(functional["runner_sha256"], failure["runner_sha256"])
        self.assertEqual(failure["o2_compile_stderr_bytes"], "1200")
        self.assertEqual(
            failure["o2_compile_stderr_sha256"],
            "13f1ab2db5ac94d4a382aa6d03a801e4607ee585e5f34bcc3a1c3b8acfffd0d5",
        )
        source_test_bytes = Path(__file__).read_bytes()
        source_test_hash = hashlib.sha256(source_test_bytes).hexdigest()
        self.assertEqual(source_test_hash, functional["source_contract_sha256"])
        self.assertEqual(source_test_hash, failure["source_contract_sha256"])
        self.assertNotEqual(
            hashlib.sha256(source_test_bytes + b"\n").hexdigest(),
            source_test_hash,
        )
        self.assertIn('inputs/ROADMAP.md', gate)


if __name__ == "__main__":
    unittest.main()
