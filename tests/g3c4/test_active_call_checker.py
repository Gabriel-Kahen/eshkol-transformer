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
        self.assertNotIn("ET_G3C4_P2_G2_TERMINAL_PRIVATE", surface)
        self.assertNotIn("ET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE", surface)
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


if __name__ == "__main__":
    unittest.main()
