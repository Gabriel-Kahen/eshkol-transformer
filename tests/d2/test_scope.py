"""Guard the carrier-neutral D2 pre-freeze scope."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "internal" / "d2" / "lib" / "d2_semantic_core.esk"


class D2ScopeTests(unittest.TestCase):
    def test_production_core_is_eshkol_only(self) -> None:
        self.assertEqual([path.suffix for path in CORE.parent.iterdir()], [".esk"])
        source = CORE.read_text(encoding="utf-8").lower()
        for forbidden in ("python", "pytorch", "system(", "popen", "shell"):
            self.assertNotIn(forbidden, source)

    def test_unresolved_public_surfaces_are_not_frozen(self) -> None:
        source = CORE.read_text(encoding="utf-8")
        for forbidden in (
            "token-dataset-open",
            "token-dataset-next-batch",
            "token-batch-release!",
        ):
            self.assertNotIn(forbidden, source)

    def test_bounded_shuffle_and_cursor_invariants_are_explicit(self) -> None:
        source = CORE.read_text(encoding="utf-8")
        for required in (
            "d2-core-unbiased-draw",
            "d2-core-permute-ordinal",
            "eshkol-d2-window-shuffle-v1\\n",
            "eshkol-token-dataset-cursor-checksum-v1\\n",
            "d2-core-cursor-encode",
            "d2-core-cursor-decode",
            "d2-core-plan-next-batch",
            "d2-core-cursor-snapshot",
            "d2-core-seek!",
            "d2-core-commit-batch!",
        ):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
