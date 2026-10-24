#!/usr/bin/env python3
"""Regression tests for the Step 4a source projection."""
import importlib.util
import subprocess
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "check_g3c4_context_cache", ROOT / "scripts/check-g3c4-context-cache.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)

BASE = """#ifdef ET_G3C4_CONTEXT_PRIVATE
int cache;
#ifdef ET_G3C4_ACTIVE_CALL_PRIVATE
int et_g3c4_model_pins;
#endif
#ifdef ET_G3C4_GENERATOR_PRIVATE
int generator_rng;
#endif
#ifdef ET_G3C4_P2_G2_TERMINAL_PRIVATE
#ifdef ET_G3C4_P2_G2_TERMINAL_TESTING
int generator_rng;
#endif
int generator_policy;
#endif
#endif
"""


class Step4ProjectionTests(unittest.TestCase):
    def test_later_feature_blocks_are_excluded(self):
        CHECKER.require_no_deferred_surface(BASE)
        source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
        block = source[source.index("#ifdef ET_G3C4_CONTEXT_PRIVATE"):]
        self.assertIn(
            "struct et_g3c4_context_internal *registry_next",
            CHECKER.step4_projection(block),
        )
        CHECKER.require_no_deferred_surface(block)

    def test_later_feature_includes_require_their_guards(self):
        for flag, include in (
            ("ET_CPU4_CANDIDATE_PRIVATE", "cpu4_candidate.inc"),
            ("ET_CPU4_C4_HISTORY_PRIVATE", "cpu4_c4_capture.inc"),
            ("ET_CPUV1_RESOURCE_TESTING", "cpuv1_resource_c4.inc"),
        ):
            guarded = (f'#ifdef ET_G3C4_CONTEXT_PRIVATE\n#ifdef {flag}\n'
                       f'#include "{include}"\n#endif\n#endif\n')
            with self.subTest(flag=flag):
                CHECKER.require_no_deferred_surface(guarded)
                unguarded = guarded.replace(f'#ifdef {flag}\n', '', 1)
                with self.assertRaisesRegex(ValueError, 'unsupported Step 4a include'):
                    CHECKER.require_no_deferred_surface(unguarded)

    def test_history_include_is_entirely_feature_guarded(self):
        included = (ROOT / 'src/eshkol_transformer/cpu4_c4_history.inc').read_text()
        CHECKER.require_history_include_guarded(included)
        with self.assertRaisesRegex(ValueError, 'history include reaches Step 4a'):
            CHECKER.require_history_include_guarded(included + '\nint unguarded_history;\n')

    def test_false_arm_leak_is_rejected(self):
        mutant = BASE.replace(
            "#ifdef ET_G3C4_GENERATOR_PRIVATE\nint generator_rng;\n#endif",
            "#ifdef ET_G3C4_GENERATOR_PRIVATE\nint generator_rng;\n"
            "#else\nint generator_seed;\n#endif",
        )
        self.assertNotEqual(mutant, BASE)
        with self.assertRaisesRegex(ValueError, "generator_seed"):
            CHECKER.require_no_deferred_surface(mutant)

    def test_reachable_conditional_arms_are_rejected(self):
        guards = (
            "#if !defined(ET_G3C4_GENERATOR_PRIVATE)",
            "#if defined(ET_G3C4_GENERATOR_PRIVATE) || "
            + chr(92) + "\n    defined(ET_G3C4_CONTEXT_PRIVATE)",
            "#ifdef ET_G3C4_GENERATOR_PRIVATE_EXTRA",
        )
        for guard in guards:
            with self.subTest(guard=guard):
                mutant = BASE.replace(
                    "int cache;", f"int cache;\n{guard}\nint generator_rng;\n#endif")
                with self.assertRaisesRegex(ValueError, "generator_rng"):
                    CHECKER.require_no_deferred_surface(mutant)

    def test_commented_guard_and_local_define_cannot_hide_leak(self):
        cases = {
            "commented guard": (
                "#ifdef ET_G3C4_CONTEXT_PRIVATE\nint cache;\n"
                "/*\n#ifdef ET_G3C4_GENERATOR_PRIVATE\n*/\n"
                "int generator_rng;\n/*\n#endif\n*/\n#endif\n",
                "generator_rng",
            ),
            "locally defined feature": (
                "#ifdef ET_G3C4_CONTEXT_PRIVATE\nint cache;\n"
                "#define ET_G3C4_GENERATOR_PRIVATE 1\n"
                "#ifdef ET_G3C4_GENERATOR_PRIVATE\n"
                "int generator_rng;\n#endif\n#endif\n",
                "feature redefined",
            ),
            "continued comment opener": (
                "#ifdef ET_G3C4_CONTEXT_PRIVATE\nint cache;\n"
                "/" + chr(92) + "\n*\n"
                "#ifdef ET_G3C4_GENERATOR_PRIVATE\n*/\n"
                "int generator_rng;\n/*\n#endif\n*/\n#endif\n",
                "generator_rng",
            ),
        }
        for label, (source, error) in cases.items():
            with self.subTest(label=label):
                processed = subprocess.run(
                    ["cc", "-E", "-P", "-x", "c",
                     "-DET_G3C4_CONTEXT_PRIVATE", "-"],
                    input=source, text=True, capture_output=True, check=True)
                self.assertIn("int generator_rng;", processed.stdout)
                with self.assertRaisesRegex(ValueError, error):
                    CHECKER.require_no_deferred_surface(source)

    def test_unknown_include_fails_closed(self):
        source = (
            "#ifdef ET_G3C4_CONTEXT_PRIVATE\n"
            '#include "feature_that_defines_generator.h"\n'
            "#ifdef ET_G3C4_GENERATOR_PRIVATE\n"
            "int generator_rng;\n#endif\n#endif\n"
        )
        with self.assertRaisesRegex(ValueError, "unsupported Step 4a include"):
            CHECKER.require_no_deferred_surface(source)

    def test_operator_macro_expansion_cannot_hide_leak(self):
        cases = (
            ("#define and ||\n",
             "defined(ET_G3C4_GENERATOR_PRIVATE) and "
             "defined(ET_G3C4_CONTEXT_PRIVATE)"),
            ("#define ENABLE_OR 1 || 1\n",
             "defined(ET_G3C4_GENERATOR_PRIVATE) && ENABLE_OR"),
        )
        for definition, condition in cases:
            with self.subTest(condition=condition):
                source = (
                    f"{definition}#ifdef ET_G3C4_CONTEXT_PRIVATE\n"
                    f"#if {condition}\nint generator_rng;\n#endif\n#endif\n"
                )
                processed = subprocess.run(
                    ["cc", "-E", "-P", "-x", "c", "-std=c11",
                     "-pedantic-errors", "-DET_G3C4_CONTEXT_PRIVATE", "-"],
                    input=source, text=True, capture_output=True, check=True)
                self.assertIn("int generator_rng;", processed.stdout)
                CHECKER.require_no_feature_redefinitions(source)
                with self.assertRaisesRegex(ValueError, "unsupported preprocessor macro expansion"):
                    CHECKER.require_no_deferred_surface(source)

    def test_unsupported_condition_fails_closed(self):
        mutant = BASE.replace(
            "int cache;",
            "int cache;\n#if defined(ET_G3C4_GENERATOR_PRIVATE) ^ 1\n"
            "int generator_rng;\n#endif",
        )
        with self.assertRaisesRegex(ValueError, "unsupported preprocessor condition"):
            CHECKER.require_no_deferred_surface(mutant)

    def test_unguarded_deferred_surfaces_are_rejected(self):
        for surface in (
            "et_g3c4_model_pins", "generator_seed", "generator_rng",
            "policy", "sampler", "forward", "owner->active =",
        ):
            with self.subTest(surface=surface):
                mutant = BASE.replace("int cache;", f"int cache;\n{surface}")
                with self.assertRaisesRegex(
                    ValueError, "Step 4a reaches deferred surface"
                ):
                    CHECKER.require_no_deferred_surface(mutant)


if __name__ == "__main__":
    unittest.main()
