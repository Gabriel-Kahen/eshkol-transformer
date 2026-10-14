"""Fail-closed source checks for the separate five-class capsule allocator recipe."""
from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "capsule_allocator_recipe",
    ROOT / "scripts/prepare-tr3-handoff-capsule-allocator-gate.py",
)
assert SPEC and SPEC.loader
RECIPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECIPE)
MANIFEST = (ROOT / "tests/tr3_handoff_capsule/allocator_prefix_candidate.tsv").read_text()
RUNNER = (ROOT / "tests/tr3_handoff_capsule/allocator_prefix_runtime.esk").read_text()
SHIM = (ROOT / "tests/tr3_handoff_capsule/capsule_allocator_shim.cpp").read_text()
GATE = (ROOT / "scripts/test-tr3-handoff-capsule-allocator-gate.sh").read_text()


def has_causal_contract(runner: str, shim: str, gate: str) -> bool:
    runner_cuts = (
        "(tr3-handoff-capture-internal! trainer tokenizer)",
        "(tr3-handoff-abort-before-eval-internal! record capsule)",
        "(= publications releases)",
        "(= (p1-live-count) original-live)",
        "(ca-verify-parameter i (vector-ref carrier 2))",
        "(ca-verify-optimizer",
        "(same-vector? (vector-ref record 10) original-modules 17)",
        "(same-vector? (vector-ref record 11) original-handles 14)",
        "(same-vector? (vector-ref record 12) original-carriers 14)",
        "(eq? (tr3-handoff-node record) original-node)",
        "(eq? (vector-ref (vector-ref record 14) 1)",
        "(eq? (vector-ref (vector-ref record 16) 1)",
        "(eq? (vector-ref (vector-ref record 22) 4)",
        "(eq? (module-mode-internal model) 'train)",
        "(arm-class! class ordinal)",
        "(ca-handler-arm)",
    )
    shim_cuts = (
        "bounded_object_failure",
        "mode == Mode::promotion || mode == Mode::target",
        "__real_eshkol_region_write_barrier_checked_v1(",
        "real_region_allocate_quiet(arena, bytes, alignment)",
        "throw std::bad_alloc()",
        "bytes == sizeof(eshkol_exception_handler_t) && in_push",
        "__real_eshkol_get_raised_value(value)",
        "require(caught == 1 && observed",
    )
    gate_cuts = (
        "for class in object all promotion target; do",
        "while (( ordinal < 2048 )); do",
        "run_case handler 0",
        "-DET_P1_TRUSTED_BUILD=1 -DET_P1_TEST_HOOKS=1",
        "ordinal > 0 && ordinal < 2048",
        "TR3-CAPSULE-ALLOCATOR-PASS classes=5",
        "/out/capsule_allocator /out/corpus",
        "cmp /out/expected-wrap-symbols.txt /out/linked-wrap-symbols.txt",
    )
    return (all(cut in runner for cut in runner_cuts)
            and all(cut in shim for cut in shim_cuts)
            and all(cut in gate for cut in gate_cuts)
            and runner.index("(arm-class! class ordinal)")
            < runner.index("(tr3-handoff-capture-internal! trainer tokenizer)",
                           runner.index("(arm-class! class ordinal)"))
            and gate.index("run_case handler 0")
            > gate.index("for class in object all promotion target; do"))


class CapsuleAllocatorSource(unittest.TestCase):
    def test_strict_manifest_and_repaired_runtime_pin(self) -> None:
        rows = RECIPE.parse_manifest(MANIFEST)
        self.assertEqual(rows["source_basis_commit"],
                         "3ea8f80ba330d5574975cfb4e5ca4187db094b9b")
        self.assertEqual(rows["runner_role"], "allocation_prefix_production_off")
        self.assertEqual(rows["admission"], "unadmitted_source_recipe")
        self.assertEqual(rows["allocation_classes"],
                         "object,all,promotion,target,handler")

    def test_manifest_mutations_fail_closed(self) -> None:
        for key in ("fixture_sha256", "shim_sha256", "gate_sha256",
                    "capsule_source_sha256", "runtime_source_commit",
                    "runner_sha256", "admission", "allocation_classes"):
            with self.subTest(key=key):
                mutated = []
                for line in MANIFEST.splitlines():
                    k, value = line.split("\t")
                    mutated.append(f"{k}\t{'0' * len(value) if k == key else value}")
                with self.assertRaises(ValueError):
                    RECIPE.parse_manifest("\n".join(mutated) + "\n")
        with self.assertRaises(ValueError):
            RECIPE.parse_manifest(MANIFEST + "admission\tadmitted\n")

    def test_causal_proof_contract_and_negative_cuts(self) -> None:
        self.assertTrue(has_causal_contract(RUNNER, SHIM, GATE))
        for source, markers in (
            (RUNNER, ("(= publications releases)",
                      "(= (p1-live-count) original-live)",
                      "(ca-verify-optimizer",
                      "(eq? (vector-ref (vector-ref record 16) 1)",
                      "(ca-handler-arm)")),
            (SHIM, ("bounded_object_failure",
                    "throw std::bad_alloc()",
                    "__real_eshkol_get_raised_value(value)")),
            (GATE, ("while (( ordinal < 2048 )); do",
                    "run_case handler 0",
                    "cmp /out/expected-wrap-symbols.txt /out/linked-wrap-symbols.txt")),
        ):
            for marker in markers:
                with self.subTest(marker=marker):
                    self.assertIn(marker, source)
                    arguments = [RUNNER, SHIM, GATE]
                    arguments[(RUNNER, SHIM, GATE).index(source)] = source.replace(marker, "")
                    self.assertFalse(has_causal_contract(*arguments))

    def test_gate_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(ROOT /
                       "scripts/test-tr3-handoff-capsule-allocator-gate.sh")],
                       check=True)


if __name__ == "__main__":
    unittest.main()
