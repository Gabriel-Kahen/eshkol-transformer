"""Source-only controls for the exact noninstalled zero wrong-tie fixture."""

import importlib.util
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/generate-e3-zero-wrong-tie-source.py"
SPEC = importlib.util.spec_from_file_location("zero_wrong_tie", GENERATOR)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ZeroWrongTieSource(unittest.TestCase):
    @staticmethod
    def balanced(source: str) -> bool:
        depth = 0
        quoted = escaped = comment = False
        for char in source:
            if char == "\n":
                comment = False
            if comment:
                continue
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == ";":
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

    def test_exact_generated_bytes_and_one_shot(self):
        original = MODULE.CANONICAL.read_bytes()
        variant = MODULE.compose(original)
        self.assertEqual(MODULE.sha(variant), MODULE.VARIANT_SHA)
        self.assertNotEqual(variant, original)
        self.assertIn(b"(vector-ref handles 9)", variant)
        self.assertIn(b"(vector-ref handles 10)", variant)
        self.assertIn(b"(vector-set! m3t-zero-test-wrong-tie-cell 0 #f)", variant)
        self.assertTrue(self.balanced(variant.decode()))
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "m3t_transport_extension.esk"
            subprocess.run([sys.executable, "-B", str(GENERATOR), "--output",
                            str(target)], check=True, capture_output=True)
            self.assertEqual(target.read_bytes(), MODULE.compose(original))
            subprocess.run([sys.executable, "-B", str(GENERATOR), "--output",
                            str(target), "--check"], check=True, capture_output=True)
            target.write_bytes(variant + b";changed\n")
            bad = subprocess.run([sys.executable, "-B", str(GENERATOR),
                                  "--output", str(target), "--check"],
                                 capture_output=True)
            self.assertNotEqual(bad.returncode, 0)

    def test_changed_source_and_missing_or_duplicate_anchor_denied(self):
        original = MODULE.CANONICAL.read_bytes()
        for changed in (
            original.replace(b"m3t-profile", b"m3t-profilE", 1),
            original.replace(MODULE.ANCHOR.encode(), b"", 1),
            original + MODULE.ANCHOR.encode(),
        ):
            with self.assertRaises(ValueError):
                MODULE.compose(changed)

    def test_tuple_only_build_and_closed_dependency(self):
        build = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        policy = (ROOT / "scripts/e3-private-package-policy.sh").read_text()
        closure = (ROOT / "tests/e3_private/zero_control_package_source_closure.txt").read_text()
        gate = (ROOT / "scripts/test-e3-zero-control.sh").read_text()
        self.assertIn('if [[ "${e3_tuple_kind}" == zero ]]; then\n    python3 -B "${PROJECT_ROOT}/scripts/generate-e3-zero-wrong-tie-source.py"', build)
        self.assertIn('"${e3_tuple_kind}" == zero ||\n        "${e3_tuple_kind}" == retention', build)
        self.assertIn('"${e3_tuple_kind}" == zero &&', policy)
        self.assertIn("@BUILD@/source/m3t_transport_extension.esk", closure)
        self.assertNotIn("native/m3t_transport_extension.esk", closure)
        self.assertIn('wrong-tie-owner missing-path', gate)

    def test_first_typed_error_and_same_process_retry(self):
        driver = (ROOT / "tests/e3_private/zero_control_negative_driver.esk").read_text()
        self.assertTrue(self.balanced(driver))
        self.assertIn("(m3t-zero-test-wrong-tie-arm!)", driver)
        self.assertIn("(eq? (transformer-error-category caught) 'internal)", driver)
        self.assertIn("(source-message \"diagnostic tie schedule differs\")", driver)
        self.assertIn("(= p1-before (native-live-count 'e3-zero-negative))", driver)
        self.assertLess(driver.index("(wrong-tie-rejected?"),
                        driver.index("(model (g3t-checked-m3t-model-create config initial))"))
        self.assertIn("(wrong-tie? wrong-tie-rejected?)", driver)
        self.assertIn("(e3-zero-negative-same-parameters?", driver)
        self.assertIn("(e3-zero-negative-same-destinations?", driver)
        self.assertIn("(e3-zero-negative-bytes=?", driver)
        p1 = (ROOT / "internal/p1/lib/transformer/module.esk").read_text()
        start = p1.index("(define (module-register-tied-parameter-internal! module name handle)")
        tied = p1[start:p1.index("(define (module-register-buffer-internal!", start)]
        self.assertIn("(module-root (vector-ref handle 1))", tied)
        self.assertNotIn("parameter-handle-shape", tied)
        canonical = MODULE.CANONICAL.read_text()
        self.assertIn("(eq? (vector-ref (car rows) 1) (vector-ref handles 10))", canonical)


if __name__ == "__main__":
    unittest.main()
