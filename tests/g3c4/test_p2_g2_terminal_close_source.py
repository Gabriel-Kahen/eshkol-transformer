"""Bounded source checks for the private Eshkol P2/G2 terminal close.

These guards prevent accidental expansion of the irreversible zero tail. They
do not substitute for the compiled same-call fault and oracle gate.
"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native/g3c4_p2g2_terminal_close_extension.esk"


def balanced(source: str) -> bool:
    depth = 0
    quoted = False
    escaped = False
    for line in source.splitlines():
        for char in line.split(";;;", 1)[0]:
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    return False
    return depth == 0 and not quoted


def authority_and_retry_gate(source: str) -> bool:
    """Require distinct original authority and a repeated precommit proof."""
    retry = source.split("(let commit-retry ((attempt 0))", 1)
    if len(retry) != 2:
        return False
    before_native = retry[1].split(
        "(g3c4-p2g2-native-terminal-commit", 1
    )[0]
    live = source.split("(define (g3c4-p2g2-terminal-output-live", 1)
    release = source.split(
        "(define (g3c4-p2g2-terminal-output-release-internal!", 1
    )
    return all((
        "(let ((root '()))" in source,
        "(publication-witness (vector #f))" in source,
        "(not (vector-ref fixed-publication 0))" in source,
        "(and (vector-ref fixed-publication 0)" in source,
        "(vector-set! publication-witness 0 #t)" in source,
        "'register original-output-shell original-check" in source,
        "'remove original-output-shell" in source,
        "'check original-shell" in source,
        len(live) == 2 and "'check shell" in live[1].split(
            "(define (g3c4-p2g2-terminal-generate-probed-internal", 1
        )[0],
        len(release) == 2 and "'check shell" in release[1],
        "first-staging operation) first" in before_native,
        "second-staging operation) second" in before_native,
        "(bytevector-u8-ref text-one 0) first" in before_native,
        "(bytevector-u8-ref text-two 1)" in before_native,
        "(cdr terminal-node) original-terminals" in before_native,
        "(cdr originals-node) original-originals" in before_native,
        "(cdr anchor-node) original-anchors" in before_native,
    ))


class TerminalCloseSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text()

    def test_fixed_private_same_call_shape(self):
        source = self.source
        self.assertTrue(balanced(source))
        self.assertIn("(m3-call operation", source)
        self.assertIn("g3c4-p2g2-native-eos-first-frame", source)
        self.assertIn("g3c4-p2g2-native-prefix-commit", source)
        self.assertIn("g3c4-p2g2-native-second-frame", source)
        self.assertIn("t1-private-g3-decode-raw-into!", source)
        self.assertIn("(if (= count 1) '() second-raw)", source)
        self.assertIn("(if (= count 1) 0 9)", source)
        self.assertIn("(define (g3c4-p2g2-terminal-generate-internal", source)
        self.assertNotIn("g3c4-p2g2-prefix-input-check", source)
        self.assertNotIn("g3c4-native-call-finish", source)

    def test_native_zero_tail_has_only_bounded_writes(self):
        source = self.source
        start = source.index("(vector-set! barrier 0 #t)")
        end = source.index("original-output-shell)", start)
        tail = source[start:end]
        for forbidden in (
            "make-", "cons", "lambda", "repair-precommit", "g3c4-native-",
            "g3c4-fail", "t1-private-", "terminal-authenticate",
            "registry-find", "vector ",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, tail)
        self.assertIn("(g3c4-call-dead! call)", tail)
        self.assertIn("(g3c4-p2g2-zero-bytes! pending-raw 2)", tail)
        self.assertIn("(vector-set! g3c4-p2g2-protected-anchors 0", tail)

    def test_detached_identity_and_lease_repair_are_separate(self):
        source = self.source
        self.assertIn("g3c4-p2g2-terminal-originals", source)
        self.assertTrue(authority_and_retry_gate(source))
        self.assertIn("original-output-shell", source)
        self.assertIn("terminal-published", source)
        self.assertIn("(repair-precommit call output)", source)
        self.assertIn("(if (vector-ref barrier 0) (exit 134))", source)
        self.assertIn("g3c4-p2g2-terminal-output-release-internal!", source)

    def test_source_gate_rejects_authority_and_retry_regressions(self):
        source = self.source
        head, marker, retry = source.partition(
            "(let commit-retry ((attempt 0))"
        )
        self.assertTrue(marker)
        mutations = (
            source.replace("(let ((root '()))", "(let ((root #f))", 1),
            source.replace(
                "(vector-ref fixed-publication 0)", "#t", 1
            ),
            source.replace("'check shell", "'present shell", 1),
            head + marker + retry.replace(
                "(cdr terminal-node) original-terminals",
                "(cdr terminal-node) terminal-node", 1,
            ),
            head + marker + retry.replace(
                "first-staging operation) first",
                "first-staging operation) second", 1,
            ),
        )
        for index, mutated in enumerate(mutations):
            with self.subTest(mutation=index):
                self.assertNotEqual(source, mutated)
                self.assertFalse(authority_and_retry_gate(mutated))


if __name__ == "__main__":
    unittest.main()
