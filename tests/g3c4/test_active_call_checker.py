"""Regressions for the Step 5 scan's later-feature boundary."""

import runpy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = runpy.run_path(str(ROOT / "scripts/check-g3c4-active-call.py"))


class ActiveCallCheckerTests(unittest.TestCase):
    def test_later_terminal_guards_are_excluded_from_real_source(self):
        source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
        surface = CHECKER["step5_projection"](source)
        self.assertIn("#ifdef ET_G3C4_P2_G2_TERMINAL_PRIVATE", source)
        self.assertIn("#ifdef ET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE", source)
        self.assertNotIn("#ifdef ET_G3C4_P2_G2_TERMINAL_PRIVATE", surface)
        self.assertNotIn("#ifdef ET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE",
                         surface)
        CHECKER["check_deferred_surfaces"](surface)

    def test_unguarded_generator_rng_still_fails(self):
        guarded = (
            "#ifdef ET_G3C4_P2_G2_TERMINAL_PRIVATE\n"
            "generator_rng\n"
            "#endif\n"
            "#ifdef ET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE\n"
            "generator_rng\n"
            "#endif\n"
        )
        CHECKER["check_deferred_surfaces"](
            CHECKER["step5_projection"](guarded))
        with self.assertRaisesRegex(ValueError, "generator_rng"):
            CHECKER["check_deferred_surfaces"](
                CHECKER["step5_projection"](guarded + "generator_rng\n"))

    def test_disabled_and_unknown_arms_remain_visible(self):
        for macro in ("ET_G3C4_P2_G2_TERMINAL_PRIVATE",
                      "ET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE"):
            mutants = (
                f"#ifndef {macro}\ngenerator_rng\n#endif\n",
                f"#if !defined({macro})\ngenerator_rng\n#endif\n",
                f"#ifdef {macro}\nsafe\n#else\ngenerator_rng\n#endif\n",
                f"#ifdef {macro}\nsafe\n#else\n"
                "#if defined(OTHER)\ngenerator_rng\n#endif\n#endif\n",
                f"#if defined({macro}) || defined(OTHER)\n"
                "generator_rng\n#endif\n",
            )
            for mutant in mutants:
                with self.subTest(macro=macro, mutant=mutant):
                    with self.assertRaisesRegex(ValueError, "generator_rng"):
                        CHECKER["check_deferred_surfaces"](
                            CHECKER["step5_projection"](mutant))
            with self.subTest(macro=macro, mutant="unsupported elif"):
                with self.assertRaisesRegex(ValueError, "unsupported.*#elif"):
                    CHECKER["step5_projection"](
                        f"#ifdef {macro}\nsafe\n#elif defined(OTHER)\n"
                        "generator_rng\n#endif\n")


if __name__ == "__main__":
    unittest.main()
