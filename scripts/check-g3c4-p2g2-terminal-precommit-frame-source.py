#!/usr/bin/env python3
"""Source-only negative checks for the fallible terminal precommit frame."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3c4_p2g2_terminal_abort_coordinator_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3c4_p2g2_terminal_precommit_frame_extension.esk",
    "scripts/check-g3c4-p2g2-terminal-precommit-frame-source.py",
    "scripts/test-g3c4-p2g2-terminal-precommit-frame-source.sh",
    "native/g3c4_p2g2_terminal_precommit_frame_source_closure.txt",
]
closure = (root / additions[-1]).read_text().splitlines()
assert closure == base + additions and len(closure) == len(set(closure)) == 213
assert all((root / path).is_file() for path in closure)
source = (root / additions[0]).read_text()
coordinator = (root / "native/g3c4_p2g2_terminal_abort_coordinator_extension.esk").read_text()


def clean(text):
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def balanced(text):
    depth = 0
    for char in text:
        depth += (char == "(") - (char == ")")
        if depth < 0:
            return False
    return depth == 0


def valid_source(text):
    code = clean(text)
    if not balanced(code):
        return False
    required = (
        "(g3c4-p2g2-terminal-pending-check call anchor operation)",
        "(vector-ref checked 2)", "(vector-ref checked 3)",
        "(vector-ref checked 4)", "(vector-ref checked 7)",
        "(vector-ref checked 8)", "(vector-ref checked 9)",
        "(vector-ref checked 10)", "(vector-ref checked 11)",
        "(vector-ref checked 12)", "(vector-ref checked 13)",
        "(vector-ref checked 14)",
        "(m3t-exact-i64? first-selected)",
        "(m3t-exact-i64? logical-count)",
        "(= logical-count expected-count)",
        "(g3c4-p2g2-terminal-text-zero? pending-raw 2)",
        "(g3c4-p2g2-terminal-text-zero? pending-staging 16)",
        "(eq? second-selected #f)",
        "(eq? (vector-ref record 13) 'before-prefix)",
        "(eq? (vector-ref protected-anchor 13) 'before-prefix)",
        "(g3c4-p2g2-terminal-text-zero? second-raw 1)",
        "(g3c4-p2g2-terminal-text-zero? second-staging 8)",
        "(= (bytevector-u8-ref text-one 0) first-selected)",
        "(g3c4-p2g2-terminal-text-zero? text-two 2)",
        "(eq? (vector-ref record 13) 'prefix-committed)",
        "(g3c4-p2g2-prefix-byte-check record first-selected operation)",
        "second-staging second-raw second-selected operation",
        "(g3c4-p2g2-terminal-text-zero? text-one 1)",
        "(= (bytevector-u8-ref text-two 0) first-selected)",
        "(= (bytevector-u8-ref text-two 1) second-selected)",
        "(vector 'precommit call anchor output (vector-ref output 0)",
        "first-raw 9 (if first-eos? #f second-raw)",
        "(if first-eos? 0 9) expected-count",
        "(if first-eos? text-one text-two)",
        "(vector-ref anchor 17) (vector-ref anchor 18)",
        "(cdr (vector-ref g3c4-p2g2-protected-records 0))",
        "(cdr (vector-ref g3c4-p2g2-protected-anchors 0))",
        "(cdr (vector-ref g3c4-p2g2-prefix-input-anchors 0))",
    )
    counts = (2,) + (1,) * (len(required) - 1)
    if any(code.count(part) != count for part, count in zip(required, counts)):
        return False
    order = (
        "(g3c4-p2g2-terminal-pending-check call anchor operation)",
        "(g3c4-p2g2-terminal-text-v256-check\n      first-staging first-raw first-selected operation)",
        "(= logical-count expected-count)",
        "(if first-eos?\n        (if",
        "(eq? second-selected #f)",
        "(g3c4-p2g2-prefix-byte-check record first-selected operation)",
        "second-staging second-raw second-selected operation",
        "(vector 'precommit call anchor output",
    )
    if list(map(code.index, order)) != sorted(map(code.index, order)):
        return False
    if code.rindex(required[0]) > code.index("(vector 'precommit"):
        return False
    if any(name in code for name in (
        "g3c4-p2g2-native-terminal-commit",
        "g3c4-p2g2-native-terminal-snapshot",
        "g3c4-entry-live!", "g3c4-call-dead!",
    )):
        return False
    return code.rstrip().endswith("pending-raw pending-staging)))")


def valid_coordinator(text):
    code = clean(text)
    if not balanced(code):
        return False
    if code.count("(g3c4-p2g2-terminal-precommit-frame") != 2:
        return False
    if code.count("call anchor first-selected #f logical-count)") != 1:
        return False
    if code.count("call anchor first-selected second-selected\n                        logical-count)") != 1:
        return False
    first_text = code.index("call anchor first-selected #f)))")
    first_frame = code.index("call anchor first-selected #f logical-count)")
    second_text = code.index("call anchor first-selected second-selected)))")
    second_frame = code.index("call anchor first-selected second-selected\n                        logical-count)")
    abort = code.rindex("(g3c4-p2g2-terminal-pending-abort! call anchor)")
    return first_text < first_frame < second_text < second_frame < abort


assert valid_source(source)
assert valid_coordinator(coordinator)
source_mutations = (
    ("(g3c4-p2g2-terminal-pending-check call anchor operation)", "#t"),
    ("(vector-ref checked 4)", "(vector-ref checked 3)"),
    ("(vector-ref checked 9)", "(vector-ref checked 7)"),
    ("(m3t-exact-i64? logical-count)", "#t"),
    ("(= logical-count expected-count)", "#t"),
    ("(g3c4-p2g2-terminal-text-zero? pending-raw 2)", "#t"),
    ("(g3c4-p2g2-terminal-text-zero? pending-staging 16)", "#t"),
    ("(eq? second-selected #f)", "#t"),
    ("(eq? (vector-ref record 13) 'before-prefix)", "#t"),
    ("(g3c4-p2g2-terminal-text-zero? second-raw 1)", "#t"),
    ("(= (bytevector-u8-ref text-one 0) first-selected)", "#t"),
    ("(g3c4-p2g2-prefix-byte-check record first-selected operation)", "#t"),
    ("second-staging second-raw second-selected operation",
     "first-staging first-raw second-selected operation"),
    ("(= (bytevector-u8-ref text-two 1) second-selected)", "#t"),
    ("first-raw 9 (if first-eos? #f second-raw)",
     "first-raw 8 (if first-eos? #f second-raw)"),
    ("(if first-eos? 0 9) expected-count", "(if first-eos? 9 9) expected-count"),
    ("(cdr (vector-ref g3c4-p2g2-protected-records 0))", "'()"),
)
for before, after in source_mutations:
    assert before in source and not valid_source(source.replace(before, after, 1)), before
for before, after in (
    ("call anchor first-selected #f logical-count)", "#t"),
    ("call anchor first-selected second-selected\n                        logical-count)", "#t"),
):
    assert before in coordinator and not valid_coordinator(coordinator.replace(before, after, 1)), before
assert not valid_source(source + "\n(g3c4-p2g2-native-terminal-commit native output)\n")
print("G3-C4 P2/G2 precommit frame source PASS: 20 negative mutations")
