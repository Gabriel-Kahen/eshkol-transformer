#!/usr/bin/env python3
"""Source-only terminal release ordering and negative mutation checks."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "native/g3c4_p2g2_terminal_release_extension.esk").read_text()
base = (root / "native/g3c4_p2g2_terminal_snapshot_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3c4_p2g2_terminal_release_extension.esk",
    "scripts/check-g3c4-p2g2-terminal-release-source.py",
    "scripts/test-g3c4-p2g2-terminal-release-source.sh",
    "native/g3c4_p2g2_terminal_release_source_closure.txt",
]
closure = (root / additions[-1]).read_text().splitlines()
assert closure == base + additions and len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure)


def valid(text):
    code = "\n".join(line.split(";", 1)[0] for line in text.splitlines())
    depth = 0
    for char in code:
        depth += (char == "(") - (char == ")")
        if depth < 0:
            return False
    if depth != 0:
        return False
    required = (
        "(m3-call operation",
        "(g3c4-registry-find shell operation)",
        "(g3c4-dead-entry? output 'terminal-output)",
        "(g3c4-p2g2-terminal-output-entry-live",
        "(g3c4-p2g2-terminal-snapshot-live-for",
        "(= (bytevector-length text-one) 1)",
        "(= (bytevector-length text-two) 2)",
        "(not (eq? text-one text-two))",
        "(eq? native (vector-ref live 3))",
        "(barrier (vector #f))",
        "(if (vector-ref barrier 0) (exit 134))",
        "(status (g3c4-native-output-release native))",
        "(if (= status 0)",
        "(vector-set! barrier 0 #t)",
        "(bytevector-u8-set! text-one 0 0)",
        "(bytevector-u8-set! text-two 0 0)",
        "(bytevector-u8-set! text-two 1 0)",
        "(g3c4-entry-dead! entry)",
        "(vector-set! live 0 'dead)",
        "(if (< index 16)",
        "(vector-set! live index #f)",
        "(g3c4-native-check status operation)",
    )
    if any(fragment not in code for fragment in required):
        return False
    if code.count("(g3c4-p2g2-terminal-output-entry-live") != 2:
        return False
    if code.count("(g3c4-p2g2-terminal-snapshot-live-for") != 2:
        return False
    if code.count("(g3c4-native-output-release native)") != 1:
        return False
    if "g3c4-p2g1-output-release-internal!" in code:
        return False
    order = (
        "(g3c4-dead-entry? output 'terminal-output)",
        "(g3c4-p2g2-terminal-output-entry-live",
        "(status (g3c4-native-output-release native))",
        "(if (= status 0)",
        "(vector-set! barrier 0 #t)",
        "(bytevector-u8-set! text-one 0 0)",
        "(g3c4-entry-dead! entry)",
        "(vector-set! live 0 'dead)",
    )
    return list(map(code.index, order)) == sorted(map(code.index, order))


assert valid(source)
mutations = (
    ("'terminal-output)", "'output)"),
    ("(g3c4-p2g2-terminal-output-entry-live", "(g3c4-registry-find"),
    ("(g3c4-p2g2-terminal-snapshot-live-for", "(vector"),
    ("(= (bytevector-length text-one) 1)", "#t"),
    ("(= (bytevector-length text-two) 2)", "#t"),
    ("(not (eq? text-one text-two))", "#t"),
    ("(eq? native (vector-ref live 3))", "#t"),
    ("(if (vector-ref barrier 0) (exit 134))", "#t"),
    ("(if (= status 0)", "(if #t"),
    ("(vector-set! barrier 0 #t)", "#t"),
    ("(bytevector-u8-set! text-one 0 0)", "#t"),
    ("(bytevector-u8-set! text-two 1 0)", "#t"),
    ("(g3c4-entry-dead! entry)", "#t"),
    ("(vector-set! live 0 'dead)", "#t"),
    ("(g3c4-native-check status operation)", "#t"),
)
for before, after in mutations:
    assert before in source and not valid(source.replace(before, after, 1)), before
print(f"G3-C4 terminal private release source PASS: {len(mutations)} negative mutations")
