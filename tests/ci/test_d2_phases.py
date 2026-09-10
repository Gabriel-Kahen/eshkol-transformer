"""Verify the D2 CI phase selector and assertion partition."""

from pathlib import Path
import subprocess
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "test-d2.sh"


class D2PhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = SCRIPT.read_text()

    def run_selector(self, *arguments):
        prefix = self.script.split("\nverify_toolchain\n", 1)[0]
        prefix = "\n".join(
            line for line in prefix.splitlines() if not line.startswith("source ")
        )
        command = f'{prefix}\nprintf "%s\\n" "$d2_phase"\n'
        return subprocess.run(
            ["bash", "-c", command, str(SCRIPT), *arguments],
            capture_output=True,
            text=True,
        )

    def test_selector_defaults_to_all_and_accepts_each_phase(self):
        cases = {
            (): "all",
            ("--phase", "all"): "all",
            ("--phase", "semantics"): "semantics",
            ("--phase", "resources"): "resources",
            ("--phase", "packaging"): "packaging",
        }
        for arguments, expected in cases.items():
            with self.subTest(arguments=arguments):
                result = self.run_selector(*arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, f"{expected}\n")

    def test_selector_rejects_malformed_requests_before_toolchain_checks(self):
        for arguments in (
            ("semantics",),
            ("--phase",),
            ("--phase", "unknown"),
            ("--phase", "resources", "extra"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_selector(*arguments)
                self.assertEqual(result.returncode, 2)
                self.assertIn("usage:", result.stderr)

    def test_phase_timing_helpers_support_paused_work_segments(self):
        prefix = self.script.split("\nverify_toolchain\n", 1)[0]
        prefix = "\n".join(
            line for line in prefix.splitlines() if not line.startswith("source ")
        )
        command = (
            f"{prefix}\n"
            "d2_phase_begin semantics\n"
            "d2_phase_pause semantics\n"
            "d2_phase_resume semantics\n"
            "d2_phase_end semantics\n"
        )
        result = subprocess.run(
            ["bash", "-c", command, str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(
            result.stdout,
            r"^D2 PHASE START: semantics\nD2 PHASE PASS: semantics elapsed=\d+s\n$",
        )

    def test_phase_dispatch_keeps_determinism_pairs_and_resource_probes(self):
        semantics = self.script.split(
            "if d2_phase_enabled semantics; then\n  d2_phase_begin semantics\n", 1
        )[1].split("\nd2_phase_pause semantics\nfi", 1)[0]
        self.assertIn("compile_semantic_core a\ncompile_semantic_core b", semantics)
        self.assertEqual(semantics.count("compile_semantic_core a"), 1)
        self.assertEqual(semantics.count("compile_semantic_core b"), 1)
        fixture_generation = semantics.split(
            "python3 -m unittest discover", 1
        )[1].split('cmp "${d2_tmp}/fixture-1.json"', 1)[0]
        self.assertEqual(fixture_generation.count("for repetition in 1 2; do"), 1)
        self.assertIn("tests.d2.generate_batch_fixture", fixture_generation)
        semantic_compile = semantics.split("compile_semantic_core() {", 1)[1].split(
            "\n}", 1
        )[0]
        self.assertIn("ESHKOL_JIT_CACHE=0", semantic_compile)
        self.assertIn('XDG_CACHE_HOME="${d2_tmp}/cache-${label}"', semantic_compile)
        native_runs = semantics.split(
            '"${d2_tmp}/test-d2-native"', 1
        )[1].split('cmp "${d2_tmp}/native-1.stdout"', 1)[0]
        self.assertIn("for repetition in 1 2; do", native_runs)
        self.assertIn("test-d2-native-sanitized", semantics)

        shared_generation = self.script.split(
            "if d2_phase_enabled semantics || d2_phase_enabled resources; then\n", 1
        )[1].split("\nfi", 1)[0]
        self.assertEqual(shared_generation.count("for repetition in 1 2; do"), 1)
        self.assertIn("tests.d2.prepare_public_resources", shared_generation)
        self.assertIn("tests.t2.generate_adversarial_fixtures", shared_generation)

        instrumented = self.script.split(
            "# only the optimized public resource executable links this test artifact.",
            1,
        )[1].split("\nd2_phase_pause resources\nfi", 1)[0]
        self.assertIn(
            'd2_resource_runtime_dir="${d2_tmp}/resource-test-runtime"', instrumented
        )
        self.assertIn(
            'd2_resource_runtime_dir_b="${d2_tmp}/resource-test-runtime-b"',
            instrumented,
        )
        self.assertIn(
            'cmp "${d2_resource_runtime_dir}/d2_wave2_test.o"', instrumented
        )
        deterministic_builds = instrumented.split("forbidden_test_object=", 1)[0]
        self.assertEqual(deterministic_builds.count("build-e1b-consumer.sh"), 2)
        self.assertIn("for test_evidence in global-defined.txt", instrumented)
        self.assertIn("forbidden_test_object=", instrumented)

        resource_probes = self.script.split(
            "if d2_phase_enabled resources; then\nd2_phase_resume resources\nrun_resource_probe()",
            1,
        )[1].split("\nd2_phase_pause resources\nfi", 1)[0]
        for evidence in (
            "D2 RESOURCE RSS/FD:",
            "D2 RESOURCE NATIVE COUNTERS PASS:",
            "D2 RESOURCE NATIVE FD PASS:",
            "D2 RESOURCE SHARD ARENA MEASURED:",
            "D2 RESOURCE REOPEN PASS:",
            "D2 RESOURCE ACCESSORS PASS:",
        ):
            self.assertIn(evidence, resource_probes)
        self.assertEqual(resource_probes.count("run_resource_probe "), 4)
        self.assertEqual(resource_probes.count("for horizon in short long; do"), 1)
        self.assertEqual(
            resource_probes.count("for topology in one-shards many-shard; do"), 1
        )
        self.assertEqual(resource_probes.count("for probe in open packed; do"), 1)

    def test_full_dispatch_preserves_all_compiled_and_audited_candidates(self):
        self.assertIn(
            "d2_public_sources+=(public_runtime public_errors_runtime)", self.script
        )
        self.assertIn(
            "d2_public_sources+=(resource_runtime resource_instrumented_runtime)",
            self.script,
        )
        compile_loop = self.script.split(
            'for public_source in "${d2_public_sources[@]}"; do', 1
        )[1].split("\ndone", 1)[0]
        self.assertEqual(compile_loop.count("compile_public_d2"), 2)
        self.assertIn('"public-a-${public_source}"', compile_loop)
        self.assertIn('"public-b-${public_source}"', compile_loop)
        public_compile = self.script.split("compile_public_d2() {", 1)[1].split(
            "\n}", 1
        )[0]
        self.assertIn("ESHKOL_JIT_CACHE=0", public_compile)
        self.assertIn('XDG_CACHE_HOME="${d2_tmp}/cache-${label}"', public_compile)
        self.assertEqual(self.script.count("compile_private_view private-a"), 1)
        self.assertEqual(self.script.count("compile_private_view private-b"), 1)
        private_compile = self.script.split("compile_private_view() {", 1)[1].split(
            "\n}", 1
        )[0]
        self.assertIn("ESHKOL_JIT_CACHE=0", private_compile)
        self.assertIn('XDG_CACHE_HOME="${d2_tmp}/cache-${label}"', private_compile)
        semantics_runtime = self.script.split(
            "if d2_phase_enabled semantics; then\nd2_phase_resume semantics\n", 1
        )[1].split("\nd2_phase_pause semantics\nfi", 1)[0]
        self.assertEqual(semantics_runtime.count("for repetition in a b; do"), 2)

        audit_lines = []
        in_assignment = False
        for line in self.script.splitlines():
            if line.startswith(
                ("d2_delivered_candidates=(", "  d2_delivered_candidates+=(")
            ):
                in_assignment = True
            elif in_assignment and line.strip() == ")":
                in_assignment = False
            elif in_assignment:
                audit_lines.append(line)
        expected_candidates = (
            '"${d2_dir}/d2_native.o"',
            '"${d2_dir}/d2_wave2.o"',
            '"${d2_tmp}/a/d2-semantic"',
            '"${d2_tmp}/b/d2-semantic"',
            '"${d2_tmp}/public-a-public_runtime/public_runtime"',
            '"${d2_tmp}/public-b-public_runtime/public_runtime"',
            '"${d2_tmp}/public-a-public_errors_runtime/public_errors_runtime"',
            '"${d2_tmp}/public-b-public_errors_runtime/public_errors_runtime"',
            '"${d2_tmp}/public-a-resource_runtime/resource_runtime"',
            '"${d2_tmp}/public-b-resource_runtime/resource_runtime"',
            '"${d2_resource_runtime_dir}/d2_wave2_test.o"',
            '"${d2_tmp}/public-a-resource_instrumented_runtime/resource_instrumented_runtime"',
            '"${d2_tmp}/public-b-resource_instrumented_runtime/resource_instrumented_runtime"',
            '"${d2_tmp}/private-a/private-view"',
            '"${d2_tmp}/private-b/private-view"',
        )
        self.assertEqual(
            tuple(line.strip() for line in audit_lines), expected_candidates
        )

        self.assertIn("strings -a", self.script)
        self.assertIn('nm -a "${delivered}"', self.script)
        self.assertIn('ldd "${delivered}"', self.script)

    def test_packaging_phase_owns_rebuild_and_negative_boundaries(self):
        packaging = self.script.split(
            "if d2_phase_enabled packaging; then\n  d2_phase_begin packaging\n", 1
        )[1].split("\nd2_phase_pause packaging\nfi", 1)[0]
        for assertion in (
            "D2 aggregate must have exactly 58 globals",
            '"${d2_tmp}/hostile-build"',
            "reject_d2_builder_input()",
            "linker admitted duplicate D2 aggregate authority",
            "D2 aggregate exposed private binding",
            "D2 production candidate references a Python runtime",
        ):
            self.assertIn(assertion, packaging)
        self.assertIn(
            'for rejected_root in "${d2_tmp}/copied-d2-root.esk" \\\n'
            '    "${d2_dir}/d2_wave2.o"; do',
            packaging,
        )
        self.assertIn("for copied_component in bridge renames exports; do", packaging)
        private_bindings = packaging.split(
            "for private_binding in d2-token-dataset-open ", 1
        )[1].split("; do", 1)[0]
        self.assertEqual(len(private_bindings.split()), 4)


if __name__ == "__main__":
    unittest.main()
