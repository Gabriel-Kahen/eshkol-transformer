#!/usr/bin/env python3
"""Source-only controls for the noninstalled sealed-M3 fixture producer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/generate-e3-sealed-fixture-source.py"
SPEC = importlib.util.spec_from_file_location("sealed_fixture", GENERATOR)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FixtureSource(unittest.TestCase):
    @staticmethod
    def balanced(source: str) -> bool:
        depth = 0
        quoted = False
        escaped = False
        comment = False
        for char in source:
            if char == "\n":
                comment = False
                continue
            if comment:
                continue
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == ";":
                comment = True
            elif char == '"':
                quoted = True
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    return False
        return depth == 0 and not quoted

    def test_exact_generated_inputs_and_private_outputs(self) -> None:
        outputs = MODULE.outputs()
        self.assertEqual(set(outputs), {
            "transformer/module.esk", "i2_wave2_root.esk",
            "m3t_transport_extension.esk"})
        p1 = outputs["transformer/module.esk"].decode()
        i2 = outputs["i2_wave2_root.esk"].decode()
        m3t = outputs["m3t_transport_extension.esk"].decode()
        for source in outputs.values():
            self.assertTrue(self.balanced(source.decode()))
        self.assertIn("et_p1_private_candidate_construction_begin_v1", p1)
        self.assertIn("et_p1_private_candidate_graph_preflight_v1", p1)
        self.assertIn("et_p1_private_candidate_graph_revoke_v1", p1)
        self.assertIn("module-candidate-construction-revoke-internal!", p1)
        self.assertIn("(if candidate? 'm3t-candidate 'm3t)", i2)
        self.assertIn("(i2-candidate-construction-unlink! i2-record)", m3t)
        self.assertIn("(module-candidate-construction-revoke-internal! construction)", m3t)
        self.assertIn("(m3t-zero-test-tie-target (vector-ref entry 8))", m3t)
        self.assertLess(
            m3t.index("m3t-native-candidate-initializer-retire-preflight\n"),
            m3t.index("m3t-native-candidate-retire-preflight\n"))
        self.assertLess(m3t.index("m3t-native-candidate-retire-preflight\n"),
                        m3t.index("module-candidate-construction-revoke-internal! construction"))
        self.assertNotIn("candidate", (ROOT / "native/m3t_transport_extension.esk").read_text())

    def test_generator_detects_wrong_source_and_missing_anchor(self) -> None:
        original = MODULE.INPUTS["p1"]
        try:
            MODULE.INPUTS["p1"] = (original[0], "0" * 64)
            with self.assertRaisesRegex(ValueError, "predecessor changed"):
                MODULE.outputs()
        finally:
            MODULE.INPUTS["p1"] = original
        with self.assertRaisesRegex(ValueError, "one source anchor"):
            MODULE.once("x", "unavailable", "y")

    def test_written_tuple_detects_changed_member(self) -> None:
        expected = MODULE.outputs()
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent) / "source"
            root.mkdir()
            for name, content in expected.items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            self.assertEqual({name: (root / name).read_bytes()
                              for name in expected}, expected)
            command = [sys.executable, "-I", "-B", str(GENERATOR),
                       "--output-dir", str(root), "--check"]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            target = root / "i2_wave2_root.esk"
            target.write_bytes(target.read_bytes() + b"\n")
            self.assertNotEqual(target.read_bytes(), expected["i2_wave2_root.esk"])
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_fixture_uses_real_sealed_owner_and_typed_stale_checks(self) -> None:
        fixture = (ROOT / "tests/e3_private/sealed_fixture_driver.esk").read_text()
        self.assertTrue(self.balanced(fixture))
        for required in (
            "(m3t-candidate-model-create config initial)",
            "(m3t-zero-test-wrong-tie-arm!)",
            "(source-message \"diagnostic tie schedule differs\")",
            "(m3t-candidate-retire! model)",
            "(m3t-public-workspace-create model)",
            "(m3t-public-workspace-release! workspace)",
            "(parameter-handle-tensor-internal head)",
            "(m3t-candidate-retire! retry)",
            "(transformer-error-operation caught)",
        ):
            self.assertIn(required, fixture)

    def test_p1_revoke_uses_only_preflight_captured_roots(self) -> None:
        source = MODULE.outputs()["transformer/module.esk"].decode()
        preflight = source.split("(define (g3r-construction-preflight", 1)[1].split(
            "(define (g3r-construction-revoke!", 1)[0]
        tail = source.split("(define (g3r-construction-revoke!", 1)[1].split(
            "(define p1-trusted-surface", 1)[0]
        for captured in ("record", "(shell-registry-current)", "identity"):
            self.assertIn("g3r-candidate-ready-root", preflight)
            self.assertIn(captured, preflight)
        self.assertIn("(eq? (vector-ref g3r-candidate-ready-root 0) identity)", tail)
        self.assertIn("(vector-set! g3r-candidate-ready-root 0 #f)", tail)
        self.assertIn("(p1-candidate-invariant-fail)", tail)

        def assert_no_late_lookup(candidate: str) -> None:
            for forbidden in ("construction-record", "shell-registry-current"):
                self.assertNotIn(forbidden, candidate)

        assert_no_late_lookup(tail)
        with self.assertRaises(AssertionError):
            assert_no_late_lookup(tail + "(construction-record operation identity)")
        with self.assertRaises(AssertionError):
            assert_no_late_lookup(tail + "(shell-registry-current)")


if __name__ == "__main__":
    unittest.main()
