#!/usr/bin/env python3
"""Regression tests for the Step 4a source projection."""
import importlib.util
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
