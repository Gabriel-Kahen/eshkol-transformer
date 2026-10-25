"""The retention executable uses the accepted no-stdlib argv traversal."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
PEER = ROOT / "tests/e3_private/runtime.esk"
RETENTION = ROOT / "tests/e3_private/retention_runtime.esk"
GATE = ROOT / "scripts/test-e3-complete-call-retention.sh"


def argv_helpers(source: str, end: str) -> str:
    start = source.index("(define (argument-count values)")
    return source[start:source.index(end, start)].strip()


class RetentionRuntimeImportsTests(unittest.TestCase):
    def test_reuses_authentic_no_stdlib_argument_traversal(self) -> None:
        runtime = RETENTION.read_text()
        peer = PEER.read_text()
        self.assertEqual(argv_helpers(runtime, "(define args (command-line))"),
                         argv_helpers(peer, "(define (bytes=?"))
        self.assertNotIn("(length args)", runtime)
        self.assertNotIn("(list-ref args", runtime)
        self.assertNotIn("(load ", runtime)
        self.assertEqual(re.findall(r"\(argument-ref args ([0-9]+)\)", runtime),
                         ["1", "2", "3", "3"])
        for label in ("arguments", "horizon", "gradient-option", "prepared",
                      "executed", "retained-used-bytes", "retained-total-bytes",
                      "retained-block-count", "result-shape", "cumulative-growth-shape"):
            self.assertIn(f"(check '{label}", runtime)
        self.assertNotEqual(
            argv_helpers(peer.replace("(null? remaining)", "#t", 1), "(define (bytes=?"),
            argv_helpers(runtime, "(define args (command-line))"))

    def test_both_aot_modes_keep_same_executable_and_case_matrix(self) -> None:
        gate = GATE.read_text()
        self.assertEqual(gate.count('"${PROJECT_ROOT}/tests/e3_private/retention_runtime.esk"'), 1)
        self.assertEqual(gate.count('"${runtime_source}"'), 2)
        self.assertIn('runtime_source="${PROJECT_ROOT}/tests/e3_private/retention_arena_diagnostic_runtime.esk"', gate)
        self.assertLess(gate.index('if [[ "$diagnostic" == 1 ]]; then'),
                        gate.index('for horizon in 1024 8192; do'))
        self.assertIn("--strict-types --no-stdlib -O 2", gate)
        self.assertIn("for horizon in 1024 8192; do", gate)
        self.assertIn("for gradient in absent present; do", gate)
        self.assertIn("for profile in normal-a normal-b poisoned-a poisoned-b sanitized; do", gate)
        self.assertIn('ASAN_OPTIONS=detect_leaks=1:halt_on_error=1', gate)


if __name__ == "__main__":
    unittest.main()
