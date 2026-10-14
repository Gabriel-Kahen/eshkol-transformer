#!/usr/bin/env python3
"""Source-only dependency check for the private P2/G2 pending terminal anchor."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3c4_p2g2_t1_prefix_coordinator_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3c4_p2g2_terminal_extension.esk",
    "tests/g3c4/p2_g2_terminal_anchor_test.esk",
    "scripts/check-g3c4-p2g2-terminal-anchor.py",
    "scripts/test-g3c4-p2g2-terminal-anchor-source.sh",
    "native/g3c4_p2g2_terminal_source_closure.txt",
]
closure = (root / additions[-1]).read_text().splitlines()
assert closure == base + additions
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure)

source = (root / additions[0]).read_text()
fixture = (root / additions[1]).read_text()
runner = (root / additions[3]).read_text()


def balanced_scheme(text: str) -> bool:
    depth = 0
    string = comment = escaped = False
    for char in text:
        if comment:
            if char == "\n":
                comment = False
            continue
        if string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                string = False
            continue
        if char == ";":
            comment = True
        elif char == '"':
            string = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not string


assert balanced_scheme(source)
assert balanced_scheme(fixture)
assert source.count("(define (g3c4-p2g2-terminal-pending-") == 5
assert "(g3c4-p2g2-protected-reserve call)" in source
assert source.index("(g3c4-p2g2-protected-reserve call)") < source.index(
    "(vector-set! g3c4-p2g2-terminal-pending-anchors 0 anchor-node)"
)
assert source.index("(g3c4-p2g2-terminal-pending-check call anchor operation)",
                    source.index("(define (g3c4-p2g2-terminal-pending-abort!")) < source.index(
    "(g3c4-p2g2-protected-abort! call)",
    source.index("(define (g3c4-p2g2-terminal-pending-abort!"))
assert source.index("(g3c4-p2g2-protected-abort! call)") < source.index(
    "(vector-set! anchor 0 'dead)"
)
assert "g3c4-p2g2-native-terminal" not in source
assert "g3c4-p2g2-native-first" not in source
assert "g3c4-p2g2-native-second" not in source
for name in (
    "g3c4-p2g2-terminal-pending-reserve!",
    "g3c4-p2g2-terminal-pending-check",
    "g3c4-p2g2-terminal-pending-abort!",
):
    assert name in fixture
assert "equal-length second carrier substitution" in fixture
assert "paired extent replacement" in fixture
assert "test-g3c4-p2g2-terminal-anchor-source" not in runner
assert "check-g3c4-p2g2-terminal-anchor.py" in runner
print("G3-C4 P2/G2 terminal anchor source PASS")
