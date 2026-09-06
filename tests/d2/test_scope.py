"""Guard the accepted carrier-neutral D2 semantic-core scope."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "internal" / "d2" / "lib" / "d2_semantic_core.esk"


class D2ScopeTests(unittest.TestCase):
    def test_production_core_is_eshkol_only(self) -> None:
        self.assertTrue(all(path.suffix == ".esk" for path in CORE.parent.iterdir()))
        source = CORE.read_text(encoding="utf-8").lower()
        for forbidden in ("python", "pytorch", "system(", "popen", "shell"):
            self.assertNotIn(forbidden, source)

    def test_semantic_core_remains_private_while_supporting_public_wrappers(self) -> None:
        source = CORE.read_text(encoding="utf-8")
        self.assertNotIn("(provide", source)
        self.assertNotIn("(define (token-dataset-", source)
        self.assertNotIn("(define (token-batch-", source)
        self.assertIn("d2-core-normalize-config", source)

    def test_bounded_shuffle_and_cursor_invariants_are_explicit(self) -> None:
        source = CORE.read_text(encoding="utf-8")
        for required in (
            "d2-core-unbiased-draw",
            "d2-core-permute-ordinal",
            "eshkol-d2-window-shuffle-v1\\n",
            "eshkol-token-dataset-cursor-checksum-v1\\n",
            "ESHKDCU1",
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
