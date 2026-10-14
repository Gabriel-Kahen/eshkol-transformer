"""Source-only checks for the unadmitted current-tree capsule runtime recipe."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "handoff_recipe", ROOT / "scripts/prepare-tr3-handoff-capsule-gate.py"
)
assert SPEC and SPEC.loader
RECIPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECIPE)
MANIFEST = (ROOT / "tests/tr3_handoff_capsule/current_tree_candidate.tsv").read_text()
FIXTURE = (ROOT / "tests/tr3_handoff_capsule/runtime_candidate.esk").read_text()


class CurrentTreeCandidate(unittest.TestCase):
    def test_exact_source_and_historical_evidence_pins(self) -> None:
        rows = RECIPE.parse_manifest(MANIFEST)
        self.assertEqual(rows["admission"], "unadmitted_source_recipe")
        self.assertEqual(rows["source_basis_commit"],
                         "793befcea413a1c76efeb374765aaf8484f59578")
        self.assertEqual(rows["lease_core_sha256"],
                         "28b1d5bddf68331a877169080ed67de0493a9697d9bc37db9ae64fb03471d75b")
        self.assertEqual(rows["allocation_classes"],
                         "object,all,promotion,target,handler")
        self.assertEqual(rows["runner_role"], "functional_recovered_final81298")

    def test_manifest_tampering_fails_closed(self) -> None:
        rows = dict(line.split("\t", 1) for line in MANIFEST.splitlines())
        for key, value in (
            ("admission", "admitted"),
            ("source_basis_tree", "0" * 40),
            ("lease_core_sha256", "0" * 64),
            ("fixture_sha256", "0" * 64),
            ("historical_functional_manifest_sha256", "0" * 64),
            ("historical_failure_manifest_sha256", "0" * 64),
            ("allocation_classes", "object,all"),
            ("runner_sha256", "0" * 64),
            ("runner_role", "wrong_role"),
        ):
            with self.subTest(key=key):
                mutant = dict(rows, **{key: value})
                with self.assertRaises(ValueError):
                    RECIPE.parse_manifest("".join(
                        f"{name}\t{item}\n" for name, item in mutant.items()
                    ))
        with self.assertRaises(ValueError):
            RECIPE.parse_manifest(MANIFEST + "admission\tadmitted\n")
        with self.assertRaises(ValueError):
            RECIPE.parse_manifest(MANIFEST.replace("\t", " ", 1))
        with self.assertRaises(ValueError):
            RECIPE.parse_manifest("\n".join(
                line for line in MANIFEST.splitlines()
                if not line.startswith("runner_role\t")) + "\n")

    def test_held_recipe_replaces_fixture_and_preflights_current_source(self) -> None:
        rendered = RECIPE.render(RECIPE.parse_manifest(MANIFEST))
        self.assertIn("tests/tr3_handoff_capsule/runtime_candidate.esk", rendered)
        self.assertNotIn("tests/tr3_joint_restore/runtime_smoke.esk", rendered)
        self.assertIn("prepare-tr3-handoff-capsule-gate.py --verify-only", rendered)
        marker = r"^TR3-HANDOFF-CANDIDATE-PASS checks=15$"
        self.assertIn(marker, rendered)
        self.assertNotIn("checks=[0-9]+", rendered)
        gate_marker = re.search(r"grep -E '([^']+)'", rendered)
        self.assertIsNotNone(gate_marker)
        self.assertEqual(gate_marker.group(1), marker)
        for stdout in ("TR3-HANDOFF-CANDIDATE-PASS checks=0\n",
                       "TR3-HANDOFF-CANDIDATE-PASS checks=14\n",
                       "TR3-HANDOFF-CANDIDATE-PASS checks=150\n"):
            with self.subTest(stdout=stdout):
                self.assertNotEqual(subprocess.run(
                    ["grep", "-E", gate_marker.group(1)], input=stdout,
                    text=True, capture_output=True).returncode, 0)
        self.assertEqual(subprocess.run(
            ["grep", "-E", gate_marker.group(1)],
            input="TR3-HANDOFF-CANDIDATE-PASS checks=15\n",
            text=True, capture_output=True).returncode, 0)
        self.assertNotIn("TR3-C JOINT RUNTIME PASS", rendered)
        with self.assertRaises(ValueError):
            RECIPE.replace_once("missing marker", "expected marker", "new")

    def test_rendered_role_expression_reports_inherited_value(self) -> None:
        rendered = RECIPE.render(RECIPE.parse_manifest(MANIFEST))
        role_lines = [line.strip() for line in rendered.splitlines()
                      if "printf 'runner_role\\t%s\\n'" in line]
        self.assertEqual(len(role_lines), 1)
        script = (f"source '{ROOT}/scripts/common.sh'\n"
                  f"candidate_manifest='{RECIPE.MANIFEST}'\n"
                  f"{role_lines[0]}\n")
        result = subprocess.run(["bash", "-c", script], capture_output=True,
                                text=True, check=True)
        self.assertEqual(result.stdout,
                         "runner_role\tfunctional_recovered_final81298\n")
        self.assertEqual(result.stderr, "")

    def test_fixture_check_count_matches_pinned_marker(self) -> None:
        self.assertEqual(len(re.findall(r"(?m)^\(check\s", FIXTURE)), 15)

    def test_fixture_covers_causal_private_cuts(self) -> None:
        for marker in (
            '(load "tr3_c_handoff_capsule_root.esk")',
            "tr3-handoff-capture-internal! trainer tokenizer",
            "tr3-handoff-abort-before-eval-internal! record capsule",
            "(vector-copy capsule)",
            "changed copied controls are rejected",
            "changed live controls are rejected",
            "(tr3-public-module-eval! model)",
            "(tr3-lease-unenroll-internal! trainer)",
            "post-P1 cut releases once and leaves no fence",
            "stale capsule cannot reenter",
            "double abort cannot release twice",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, FIXTURE)
        self.assertNotIn("module-eval! model)\n(define capsule", FIXTURE)


    def test_fixture_has_balanced_eshkol_forms(self) -> None:
        depth = 0
        string = False
        escaped = False
        comment = False
        for char in FIXTURE:
            if char == "\n":
                comment = False
                continue
            if comment:
                continue
            if string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    string = False
                continue
            if char == ";":
                comment = True
            elif char == '"':
                string = True
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                self.assertGreaterEqual(depth, 0)
        self.assertFalse(string)
        self.assertEqual(depth, 0)


if __name__ == "__main__":
    unittest.main()
