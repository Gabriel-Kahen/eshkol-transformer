"""Causal source mutants for the private P1 fixed-set eval validator."""

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-tr3-p1-fixed.py"
SPEC = importlib.util.spec_from_file_location("tr3_p1_checker", CHECKER)
assert SPEC and SPEC.loader
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class EvalRecheckSourceTest(unittest.TestCase):
    def test_exact_eval_delta_and_causal_mutants(self):
        source = (ROOT / "internal/p1/lib/transformer/module.esk").read_text()
        checker.check_eval_fidelity(source)
        mutants = (
            ("e3-mode-eval", "e3-mode-train"),
            ("(vector-ref node 7) #t", "(vector-ref node 7) #f"),
            ("(vector-ref node 6) provider", "(vector-ref node 6) #f"),
            ("(tr3-fixed-existing-shell 'module node) shell",
             "(tr3-fixed-existing-shell 'module node) node"),
            ("(if (= index 17) 0", "(if (= index 16) 0"),
            ("(eq? model (vector-ref modules 0))",
             "(equal? model (vector-ref modules 0))"),
            ("(tr3-fixed-recheck-handles\n"
             "                          modules handles carriers provider)",
             "0"),
        )
        start = source.index("(define (tr3-fixed-eval-recheck-module")
        end = source.index("\n\n(vector\n", start)
        for old, new in mutants:
            with self.subTest(mutant=old):
                leaf = source[start:end]
                self.assertIn(old, leaf)
                poisoned = source[:start] + leaf.replace(old, new, 1) + source[end:]
                with self.assertRaises(ValueError):
                    checker.check_eval_fidelity(poisoned)

    def test_runtime_fixture_contains_exact_owner_and_mode_cuts(self):
        fixed = (ROOT / "tests/tr3_p1/fixed_set_smoke.esk").read_text()
        mixed = (ROOT / "tests/e3_p1/mode_seam_runtime.esk").read_text()
        for label in (
            "eval recheck exact original fixed set",
            "eval recheck rejects copied model shell",
            "eval recheck rejects foreign model shell",
            "eval recheck rejects bit-equal distinct module",
            "eval recheck rejects bit-equal distinct handle",
            "eval recheck rejects stale foreign carrier",
            "eval recheck rejects restored train",
        ):
            self.assertIn(label, fixed)
        for label in (
            "eval-recheck-rejects-train",
            "eval-recheck-rejects-mixed",
            "eval-recheck-accepts-exact-entered-set",
            "eval-recheck-rejects-restored-mixed",
        ):
            self.assertIn(label, mixed)


if __name__ == "__main__":
    unittest.main()
