#!/usr/bin/env python3
"""Source-only closure and negative checks for abort-only logical text."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3c4_p2g2_terminal_second_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3c4_p2g2_terminal_text_preflight_extension.esk",
    "scripts/check-g3c4-p2g2-terminal-text-preflight-source.py",
    "scripts/test-g3c4-p2g2-terminal-text-preflight-source.sh",
    "native/g3c4_p2g2_terminal_text_preflight_source_closure.txt",
]
closure = (root / additions[-1]).read_text().splitlines()
assert closure == base + additions and len(closure) == len(set(closure)) == 205
assert all((root / name).is_file() for name in closure)
source = (root / additions[0]).read_text()


def valid(text):
    code = "\n".join(line.split(";", 1)[0] for line in text.splitlines())
    depth = 0
    for char in code:
        depth += (char == "(") - (char == ")")
        if depth < 0:
            return False
    if depth:
        return False
    required = (
        "(g3c4-p2g2-terminal-pending-check call anchor operation)",
        "(vector-ref checked 7)", "(vector-ref checked 8)",
        "(vector-ref checked 9)", "(vector-ref checked 10)",
        "(vector-ref checked 13)", "(vector-ref checked 14)",
        "(vector-ref (vector-ref record 15) 6)",
        "(m3t-exact-i64? first-selected)", "(m3t-exact-i64? eos)",
        "(= first-selected eos)",
        "(= (bytevector-u8-ref staging 0) selected)",
        "(= (bytevector-u8-ref raw 0) selected)",
        "(= (bytevector-u8-ref staging index) 0)",
        "(= (bytevector-u8-ref bytes index) 0)",
        "(eq? second-selected #f)",
        "(eq? (vector-ref record 13) 'before-prefix)",
        "(eq? (vector-ref protected-anchor 13) 'before-prefix)",
        "(eq? (vector-ref record 13) 'prefix-committed)",
        "(eq? (vector-ref protected-anchor 13)\n                             'prefix-committed)",
        "(g3c4-p2g2-prefix-byte-check record first-selected operation)",
        "second-staging second-raw second-selected operation",
        "(g3c4-p2g2-terminal-text-zero? second-raw 1)",
        "(g3c4-p2g2-terminal-text-zero? second-staging 8)",
        "(g3c4-p2g2-terminal-text-zero? text-one 1)",
        "(g3c4-p2g2-terminal-text-zero? text-two 2)",
        "(bytevector-u8-set! text-one 0 first-selected)",
        "(bytevector-u8-set! text-two 0 first-selected)",
        "(bytevector-u8-set! text-two 1 second-selected)",
        "(begin (bytevector-u8-set! text-one 0 first-selected) 1)",
        "(bytevector-u8-set! text-two 1 second-selected)\n               2)",
    )
    counts = (2,) + (1,) * (len(required) - 1)
    if any(code.count(item) != count for item, count in zip(required, counts)):
        return False
    if "(let upper ((index 1))" not in code or "(if (< index 8)" not in code:
        return False
    body = code[code.index("(define (g3c4-p2g2-terminal-text-preflight!"):]
    order = (
        "(g3c4-p2g2-terminal-text-v256-check\n      first-staging first-raw first-selected operation)",
        "(if first-eos?", "(eq? second-selected #f)",
        "(g3c4-p2g2-prefix-byte-check record first-selected operation)",
        "second-staging second-raw second-selected operation",
        "(g3c4-p2g2-terminal-text-zero? text-one 1)",
        "(bytevector-u8-set! text-one 0 first-selected)",
    )
    if list(map(body.index, order)) != sorted(map(body.index, order)):
        return False
    if body.rindex(required[0]) < body.index(order[-2]):
        return False
    forbidden = (
        "g3c4-p2g2-native-terminal-commit",
        "g3c4-p2g2-native-terminal-snapshot",
        "g3c4-p2g2-native-terminal-ids-clone",
        "g3c4-entry-live!", "g3c4-call-dead!",
    )
    return not any(name in code for name in forbidden)


assert valid(source)
mutations = (
    ("(g3c4-p2g2-terminal-pending-check call anchor operation)", "#t"),
    ("(vector-ref checked 9)", "(vector-ref checked 7)"),
    ("(vector-ref checked 10)", "(vector-ref checked 8)"),
    ("(= first-selected eos)", "#t"),
    ("(m3t-exact-i64? eos)", "#t"),
    ("(= (bytevector-u8-ref raw 0) selected)", "#t"),
    ("(= (bytevector-u8-ref staging index) 0)", "#t"),
    ("(= (bytevector-u8-ref bytes index) 0)", "#t"),
    ("(eq? second-selected #f)", "#t"),
    ("(eq? (vector-ref record 13) 'before-prefix)", "#t"),
    ("(eq? (vector-ref protected-anchor 13) 'before-prefix)", "#t"),
    ("(eq? (vector-ref record 13) 'prefix-committed)", "#t"),
    ("(g3c4-p2g2-prefix-byte-check record first-selected operation)", "#t"),
    ("second-staging second-raw second-selected operation",
     "first-staging first-raw first-selected operation"),
    ("(g3c4-p2g2-terminal-text-zero? text-one 1)", "#t"),
    ("(bytevector-u8-set! text-one 0 first-selected)", "#t"),
    ("(bytevector-u8-set! text-two 1 second-selected)", "#t"),
    ("(bytevector-u8-set! text-two 1 second-selected)\n               2)",
     "(bytevector-u8-set! text-two 1 second-selected)\n               1)"),
)
for before, after in mutations:
    assert before in source and not valid(source.replace(before, after, 1)), before
assert not valid(source + "\n(g3c4-p2g2-native-terminal-commit native output)\n")
print("G3-C4 P2/G2 terminal text preflight source PASS: 19 negative mutations")
