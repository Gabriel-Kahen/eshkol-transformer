"""Check that T2 boundary phases preserve the full gate's evidence."""

from pathlib import Path
import subprocess
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "test-t2-boundary.sh"


class T2BoundaryPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = SCRIPT.read_text(encoding="utf-8")

    def run_selector(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        prefix = self.script.split("\nsource ", 1)[0]
        harness = (
            prefix
            + """
for candidate in production d1-test public-caller; do
  if t2_boundary_phase_enabled "${candidate}"; then
    printf '%s\n' "${candidate}"
  fi
done
"""
        )
        return subprocess.run(
            ["bash", "-c", harness, str(SCRIPT), *arguments],
            cwd=PROJECT_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_zero_args_and_explicit_all_select_every_phase(self) -> None:
        expected = "production\nd1-test\npublic-caller\n"
        for arguments in ((), ("--phase", "all")):
            with self.subTest(arguments=arguments):
                result = self.run_selector(*arguments)
                self.assertEqual((result.returncode, result.stdout), (0, expected))

    def test_each_explicit_phase_selects_only_itself(self) -> None:
        for phase in ("production", "d1-test", "public-caller"):
            with self.subTest(phase=phase):
                result = self.run_selector("--phase", phase)
                self.assertEqual((result.returncode, result.stdout), (0, f"{phase}\n"))

    def test_malformed_requests_fail_before_toolchain_checks(self) -> None:
        for arguments in (
            ("production",),
            ("--phase",),
            ("--phase", "unknown"),
            ("--phase", "d1-test", "extra"),
            ("--wrong", "public-caller"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_selector(*arguments)
                self.assertEqual(result.returncode, 2)
                self.assertIn("usage:", result.stderr)

    def test_all_keeps_interleaved_same_path_build_pairs(self) -> None:
        build_pair = self.script.split("for repetition in 1 2; do", 1)[1].split(
            "\nif t2_boundary_phase_enabled production; then\nproduction_one=", 1
        )[0]
        ordered = (
            'scripts/build-t2.sh"',
            'build_d1_test "${working_dir}/d1/wave2-d1-test.o"',
            'cp "${working_dir}/production/wave2.o"',
            'cp "${working_dir}/d1/wave2-d1-test.o"',
        )
        positions = tuple(build_pair.index(fragment) for fragment in ordered)
        self.assertEqual(positions, tuple(sorted(positions)))
        self.assertEqual(
            build_pair.count("if t2_boundary_phase_enabled production; then"), 2
        )
        self.assertEqual(
            build_pair.count("if t2_boundary_phase_enabled d1-test; then"), 2
        )
        self.assertEqual(build_pair.count('scripts/build-t2.sh"'), 1)
        self.assertEqual(
            build_pair.count(
                'build_d1_test "${working_dir}/d1/wave2-d1-test.o"'
            ),
            1,
        )
        self.assertIn(
            'cp -a "${working_dir}/production/wave2.o.evidence"', build_pair
        )
        self.assertIn(
            'cp -a "${working_dir}/d1/wave2-d1-test.o.evidence"', build_pair
        )

    def test_full_gate_keeps_original_assertion_order(self) -> None:
        ordered = (
            'cmp "${production_one}/wave2.o"',
            'cmp "${d1_one}/wave2-d1-test.o"',
            "Wave 2 production aggregate must expose exactly 47 globals",
            "Wave 2 D1 test aggregate must expose exactly 48 globals",
            'check_source_closure "${production_evidence}/private.d"',
            'check_source_closure "${d1_evidence}/private.d"',
            'check_localized_archive "${production_object}"',
            'check_localized_archive "${d1_object}"',
            "reject_builder_input copied-root",
            "reject_builder_input copied-d1-root",
            "reject_builder_input copied-bridge",
            "reject_builder_input mismatched-d1-bridge",
            "reject_builder_input wrong-include-order",
            "private_source_bindings=(",
            "caller_source=",
            "# Independent Wave 1 and Wave 2 packages",
        )
        positions = tuple(self.script.index(fragment) for fragment in ordered)
        self.assertEqual(positions, tuple(sorted(positions)))

    def test_production_owns_exact_negative_families(self) -> None:
        self.assertIn(
            "if t2_boundary_phase_enabled production; then\n"
            "private_source_bindings=(",
            self.script,
        )
        private_bindings = self.script.split("private_source_bindings=(", 1)[1].split(
            "\n)", 1
        )[0]
        guessed_symbols = self.script.split("guessed_symbols=(", 1)[1].split(
            "\n)", 1
        )[0]
        self.assertEqual(len(private_bindings.split()), 13)
        self.assertEqual(len(guessed_symbols.split()), 13)
        for label in (
            "copied-root",
            "prelocalized-root",
            "copied-bridge",
            "mismatched-bridge",
            "mismatched-renames",
            "mismatched-exports",
            "wrong-include-order",
            "hostile-include",
        ):
            self.assertEqual(self.script.count(f"reject_builder_input {label} "), 1)
        standalone = self.script.split("for standalone_root in", 1)[1].split(
            "\ndone", 1
        )[0]
        self.assertEqual(standalone.count('"${PROJECT_ROOT}/internal/t2/'), 3)
        self.assertIn(
            "if t2_boundary_phase_enabled production; then\n"
            "# Independent Wave 1 and Wave 2 packages",
            self.script,
        )

    def test_d1_phase_owns_pair_evidence_and_two_negatives(self) -> None:
        self.assertIn(
            "if t2_boundary_phase_enabled d1-test; then\n"
            'd1_one="${t2_boundary_tmp}/repeat-1/d1"',
            self.script,
        )
        self.assertEqual(self.script.count("reject_builder_input copied-d1-root "), 1)
        self.assertEqual(
            self.script.count("reject_builder_input mismatched-d1-bridge "), 1
        )
        self.assertIn(
            'diff -ru "${d1_one}/wave2-d1-test.o.evidence"', self.script
        )
        self.assertIn(
            'check_localized_archive "${d1_object}" "${d1_archive}"', self.script
        )

    def test_public_caller_owns_only_canonical_d2_dependency(self) -> None:
        public = self.script.split(
            "if t2_boundary_phase_enabled public-caller; then\n"
            "# A public caller",
            1,
        )[1].split(
            "\nfi\n\nif t2_boundary_phase_enabled production; then\n"
            "# Independent Wave 1",
            1,
        )[0]
        self.assertEqual(self.script.count('$(project_build_dir)/d2'), 1)
        self.assertIn('$(project_build_dir)/d2', public)
        self.assertEqual(public.count("for repetition in 1 2; do"), 1)
        for cache_family in ("caller-object", "caller-link"):
            self.assertIn(f'run_compiler "{cache_family}-${{repetition}}"', public)
        ordered = (
            'run_compiler "caller-object-${repetition}"',
            'run_compiler "caller-link-${repetition}"',
            '"${t2_boundary_tmp}/caller" \\',
            'cp "${t2_boundary_tmp}/caller.o"',
            'cp "${t2_boundary_tmp}/caller"',
            'cp "${t2_boundary_tmp}/caller.d"',
        )
        positions = tuple(public.index(fragment) for fragment in ordered)
        self.assertEqual(positions, tuple(sorted(positions)))
        for comparison in (
            'cmp "${t2_boundary_tmp}/caller-1.o" "${t2_boundary_tmp}/caller-2.o"',
            'cmp "${t2_boundary_tmp}/caller-1" "${t2_boundary_tmp}/caller-2"',
            'cmp "${t2_boundary_tmp}/caller-1.d" "${t2_boundary_tmp}/caller-2.d"',
            'cmp "${t2_boundary_tmp}/caller-1.stdout" \\',
        ):
            self.assertEqual(public.count(comparison), 1)
        run_compiler = self.script.split("run_compiler() {", 1)[1].split(
            "\n}", 1
        )[0]
        self.assertIn("ESHKOL_JIT_CACHE=0", run_compiler)
        self.assertIn(
            'XDG_CACHE_HOME="${t2_boundary_tmp}/cache/${cache_name}"',
            run_compiler,
        )
        public_sources = public.split(
            "for public_source in config ", 1
        )[1].split("; do", 1)[0]
        self.assertEqual(
            tuple(public_sources.replace("\\\n", " ").split()),
            ("data", "error_consumer", "error_public", "module", "persistence", "tokenizer"),
        )
        private_markers = public.split(
            "for private_marker in ", 1
        )[1].split("; do", 1)[0]
        self.assertEqual(
            len(private_markers.replace("\\\n", " ").split()), 5
        )
        for assertion in (
            "caller-1.o",
            "caller-1.d",
            "t1-import-reverse:v1",
            "public-source-closure-repeat.txt",
            "caller-wrapper-refs.txt",
            "links a Python or Torch runtime",
        ):
            self.assertIn(assertion, public)

    def test_phase_outputs_do_not_claim_unrun_checks(self) -> None:
        self.assertEqual(self.script.count("T2 BOUNDARY PASS:"), 1)
        self.assertEqual(self.script.count("T2 BOUNDARY PRODUCTION PASS:"), 1)
        self.assertEqual(self.script.count("T2 BOUNDARY D1-TEST PASS:"), 1)
        self.assertEqual(self.script.count("T2 BOUNDARY PUBLIC-CALLER PASS:"), 1)


if __name__ == "__main__":
    unittest.main()
