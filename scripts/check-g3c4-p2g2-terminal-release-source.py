#!/usr/bin/env python3
"""Source-only terminal release ordering and negative mutation checks."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "native/g3c4_p2g2_terminal_release_extension.esk").read_text()
authority = (root / "native/g3c4_p2g2_terminal_live_authority_extension.esk").read_text()
base = (root / "native/g3c4_p2g2_terminal_snapshot_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3c4_p2g2_terminal_release_extension.esk",
    "scripts/check-g3c4-p2g2-terminal-release-source.py",
    "scripts/test-g3c4-p2g2-terminal-release-source.sh",
    "scripts/test-g3c4-p2g2-terminal-release-dead-authority.scm",
    "native/g3c4_p2g2_terminal_release_source_closure.txt",
]
closure = (root / additions[-1]).read_text().splitlines()
assert closure == base + additions and len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure)


def valid(text, owner=authority):
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
        "(g3c4-p2g2-terminal-output-entry-dead shell output operation)",
        "(g3c4-p2g2-terminal-output-entry-live",
        "(g3c4-p2g2-terminal-snapshot-live-for",
        "(g3c4-p2g2-terminal-output-original-for",
        "(eq? original (vector-ref live 10))",
        "(= (bytevector-length text-one) 1)",
        "(= (bytevector-length text-two) 2)",
        "(not (eq? text-one text-two))",
        "(eq? native (vector-ref live 3))",
        "(barrier (vector #f))",
        "(if (vector-ref barrier 0) (exit 134))",
        "(status (original 'native-release release-request))",
        "(if (= status 0)",
        "(vector-set! barrier 0 #t)",
        "(bytevector-u8-set! text-one 0 0)",
        "(bytevector-u8-set! text-two 0 0)",
        "(bytevector-u8-set! text-two 1 0)",
        "(g3c4-entry-dead! entry)",
        "(vector-set! live 0 'dead)",
        "(original 'retire entry)",
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
    if "(g3c4-native-output-release native)" in code:
        return False
    if "g3c4-p2g1-output-release-internal!" in code:
        return False
    order = (
        "(g3c4-dead-entry? output 'terminal-output)",
        "(g3c4-p2g2-terminal-output-entry-dead shell output operation)",
        "(g3c4-p2g2-terminal-output-entry-live",
        "(status (original 'native-release release-request))",
        "(if (= status 0)",
        "(vector-set! barrier 0 #t)",
        "(bytevector-u8-set! text-one 0 0)",
        "(g3c4-entry-dead! entry)",
        "(vector-set! live 0 'dead)",
        "(original 'retire entry)",
    )
    if list(map(code.index, order)) != sorted(map(code.index, order)):
        return False
    owner_code = "\n".join(line.split(";", 1)[0] for line in owner.splitlines())
    owner_required = (
        "(native-released? #f) (retired? #f)",
        "((eq? command 'native-release)",
        "(g3c4-native-output-release native)",
        "(vector-set! (cdr candidate) 0 #t)",
        "(set! native-released? #t)",
        "((eq? command 'retire)",
        "(set! retired? #t)",
        "((eq? command 'dead)",
        "(and bound? nodes-bound? native-released? retired?\n                       (eq? candidate output)",
        "(eq? (vector-ref output 0) shell)",
        "(g3c4-dead-entry? output 'terminal-output)\n                       (let clear-entry",
        "(eq? (vector-ref live 0) 'dead)",
        "(g3c4-p2g2-terminal-output-original-for shell output operation)",
        "(if (not (original 'dead output))",
    )
    return all(fragment in owner_code for fragment in owner_required)


assert valid(source)
mutations = (
    ("'terminal-output)", "'output)"),
    ("(g3c4-p2g2-terminal-output-entry-dead shell output operation)", "#t"),
    ("(eq? original (vector-ref live 10))", "#t"),
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
    ("(original 'retire entry)", "#t"),
    ("(g3c4-native-check status operation)", "#t"),
)
for before, after in mutations:
    assert before in source and not valid(source.replace(before, after, 1)), before
owner_mutations = (
    ("(native-released? #f) (retired? #f)", "(native-released? #t) (retired? #t)"),
    ("(set! native-released? #t)", "#t"),
    ("(set! retired? #t)", "#t"),
    ("(and bound? nodes-bound? native-released? retired?", "(and bound? nodes-bound?"),
    ("(eq? candidate output)\n                       (eq? (vector-ref output 0) shell)",
     "#t\n                       (eq? (vector-ref output 0) shell)"),
    ("(g3c4-dead-entry? output 'terminal-output)\n                       (let clear-entry",
     "#t\n                       (let clear-entry"),
)
for before, after in owner_mutations:
    assert before in authority and not valid(source, authority.replace(before, after, 1)), before
print(f"G3-C4 terminal private release source PASS: {len(mutations) + len(owner_mutations)} negative mutations")
