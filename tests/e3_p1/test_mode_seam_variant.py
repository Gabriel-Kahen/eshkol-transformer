"""Exact source-composition and ownership limits of the P1-only test variant."""

from pathlib import Path
import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/generate-e3-p1-mode-test-variant.py"
SPEC = importlib.util.spec_from_file_location("e3_p1_variant", GENERATOR)
assert SPEC and SPEC.loader
variant = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(variant)


class ModeSeamVariantTest(unittest.TestCase):
    def test_runtime_runner_resolves_only_generated_p1(self) -> None:
        runner = (ROOT / "scripts/test-e3-p1-mode-seam.sh").read_text()
        self.assertLess(runner.index('-I "${out}/source"'),
                        runner.index('-I "${PROJECT_ROOT}/internal/p1/lib"'))
        self.assertIn('--compile-only --emit-depfile "${out}/deps-${cache}.d"',
                      runner)
        self.assertIn('-o "${out}/object-${cache}.o"', runner)
        self.assertIn('XDG_CACHE_HOME="${out}/cache-object-${cache}"', runner)
        self.assertIn('XDG_CACHE_HOME="${out}/cache-link-${cache}"', runner)
        self.assertLess(runner.index('--compile-only --emit-depfile'),
                        runner.index('-o "${out}/runtime-${cache}"'))
        self.assertIn('did not resolve the generated P1 variant', runner)
        self.assertIn('also resolved canonical P1', runner)
        self.assertIn('cmp "${out}/object-a.o" "${out}/object-b.o"', runner)
        self.assertIn('cmp "${out}/deps-a.normalized.d" "${out}/deps-b.normalized.d"',
                      runner)
        self.assertIn('cmp "${out}/runtime-a" "${out}/runtime-b"', runner)
        self.assertIn('cmp "${out}/runtime-a.stdout" "${out}/runtime-b.stdout"',
                      runner)

    def test_existing_e3_tuples_keep_canonical_p1(self) -> None:
        for manifest in (
            "native/e3_private_package_source_closure.txt",
            "native/e3_diagnostic_private_package_source_closure.txt",
            "native/e3_diagnostic_public_package_source_closure.txt",
        ):
            rows = (ROOT / manifest).read_text().splitlines()
            self.assertEqual(rows.count("internal/p1/lib/transformer/module.esk"),
                             1, manifest)
            self.assertFalse(any("mode-test-variant" in row for row in rows),
                             manifest)

    def test_exact_regeneration_and_near_matches(self) -> None:
        canonical = variant.CANONICAL.read_text()
        self.assertEqual(hashlib.sha256(canonical.encode()).hexdigest(),
                         variant.CANONICAL_SHA256)
        generated = variant.compose(canonical)
        self.assertEqual(hashlib.sha256(generated.encode()).hexdigest(),
                         variant.VARIANT_SHA256)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "transformer/module.esk"
            for extra in ([], ["--check"]):
                result = subprocess.run(
                    [sys.executable, "-B", str(GENERATOR), "--output",
                     str(output), *extra], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_text(), generated)
            output.write_text(generated + "; poisoned\n")
            result = subprocess.run(
                [sys.executable, "-B", str(GENERATOR), "--output",
                 str(output), "--check"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("bytes differ", result.stderr)
        with self.assertRaisesRegex(ValueError, "digest differs"):
            variant.compose(canonical.replace("e3-mode-eval 'eval",
                                              "e3-mode-eval 'train", 1))

    def test_trusted_surface_stays_73_and_raw_nodes_do_not_escape(self) -> None:
        canonical = variant.CANONICAL.read_text()
        generated = variant.compose(canonical)
        self.assertNotIn("p1-e3-test-seam", canonical)
        forms = variant.spans(generated)
        trusted_start, trusted_end = variant.one_form(
            generated, forms, "(define p1-trusted-surface", 0)
        vector = [(start, end) for start, end, depth in forms
                  if depth == 2 and trusted_start < start < end < trusted_end
                  and generated.startswith("(vector\n  (lambda (module)", start)]
        self.assertEqual(len(vector), 1)
        start, end = vector[0]
        canonical_forms = variant.spans(canonical)
        canonical_trusted_start, canonical_trusted_end = variant.one_form(
            canonical, canonical_forms, "(define p1-trusted-surface", 0)
        canonical_vector = [(begin, finish) for begin, finish, depth
                            in canonical_forms
                            if depth == 2 and canonical_trusted_start < begin
                            < finish < canonical_trusted_end
                            and canonical.startswith(
                                "(vector\n  (lambda (module)", begin)]
        self.assertEqual(len(canonical_vector), 1)
        original_start, original_end = canonical_vector[0]
        self.assertEqual(generated[start:end],
                         canonical[original_start:original_end])
        self.assertIn("p1-e3-test-fixed-mixed! p1-e3-test-mode-mask",
                      generated)
        self.assertIn("(e3-mode-record token)", generated)
        self.assertIn("(e3-mode-check record)", generated)
        self.assertIn("(eq? token (vector-ref record 0))", generated)
        self.assertNotIn("(vector-ref (m3t-entry", generated)
        self.assertEqual(
            generated.count("(vector-set! (vector-ref nodes "),
            canonical.count("(vector-set! (vector-ref nodes ") + 17,
        )
        for index in range(17):
            self.assertIn(
                f"(vector-set! (vector-ref nodes {index}) 1 "
                f"e3-mode-{'train' if index % 2 == 0 else 'eval'})",
                generated,
            )
            self.assertIn(
                f"(if (eq? (vector-ref (vector-ref nodes {index}) 1) "
                f"e3-mode-eval) {1 << index} 0)", generated,
            )
        expected_mixed_mask = sum(1 << index for index in range(17)
                                  if index % 2 == 1)
        fixture = (ROOT / "tests/e3_p1/mode_seam_runtime.esk").read_text()
        self.assertEqual(expected_mixed_mask, 43690)
        for label in ("mixed17-mask", "exact-mixed17-restore",
                      "retry-mixed17-restore"):
            self.assertIn(
                f"(check '{label} (p1-e3-test-mode-mask token) "
                f"{expected_mixed_mask})", fixture,
            )

    def test_duplicate_parsed_anchors_are_rejected(self) -> None:
        canonical = variant.CANONICAL.read_text()
        duplicated = canonical + "\n(define p1-trusted-surface #f)\n"
        previous = variant.CANONICAL_SHA256
        try:
            variant.CANONICAL_SHA256 = hashlib.sha256(
                duplicated.encode()).hexdigest()
            with self.assertRaisesRegex(ValueError, "expected one parsed"):
                variant.compose(duplicated)
        finally:
            variant.CANONICAL_SHA256 = previous


if __name__ == "__main__":
    unittest.main()
