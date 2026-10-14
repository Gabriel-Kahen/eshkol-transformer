#!/usr/bin/env python3
"""Source-only checks for the abortable non-EOS terminal prefix transition."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3c4_p2g2_terminal_first_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3c4_p2g2_terminal_prefix_extension.esk",
    "scripts/check-g3c4-p2g2-terminal-prefix-source.py",
    "scripts/test-g3c4-p2g2-terminal-prefix-source.sh",
    "native/g3c4_p2g2_terminal_prefix_source_closure.txt",
]
closure = (root / additions[-1]).read_text().splitlines()
assert closure == base + additions and len(closure) == len(set(closure))
assert all((root / name).is_file() for name in closure)
source = (root / additions[0]).read_text()


def code_only(text):
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def valid(text):
    code = code_only(text)
    depth = 0
    for char in code:
        depth += (char == "(") - (char == ")")
        if depth < 0:
            return False
    if depth:
        return False
    required = (
        "(g3c4-p2g2-terminal-pending-check call anchor operation)",
        "(eq? (vector-ref record 13) 'before-prefix)",
        "(eq? (vector-ref protected-anchor 13) 'before-prefix)",
        "(eq? (vector-ref record 7) output-native)",
        "(eq? (vector-ref record 11) raw)",
        "(eq? (vector-ref record 12) staging)",
        "(g3c4-p2g2-prefix-byte-check record selected operation)",
        "(g3c4-p2g2-native-prefix-commit native output-native raw)",
        "(vector-set! protected-anchor 13 'prefix-committed)",
        "(vector-set! record 13 'prefix-committed)",
        "(vector-set! (vector-ref record 14) 3 'p2g2-prefix-committed)",
    )
    if any(code.count(item) != (2 if index == 0 else 1)
           for index, item in enumerate(required)):
        return False
    order = (required[0], required[1], required[6], required[7],
             required[8], required[9], required[10])
    if list(map(code.index, order)) != sorted(map(code.index, order)):
        return False
    if not (code.index(required[6]) < code.rindex(required[0]) <
            code.index(required[7])):
        return False
    if any(name in code for name in (
        "g3c4-p2g2-native-terminal-commit",
        "g3c4-p2g2-native-terminal-snapshot",
        "g3c4-p2g2-native-second-frame",
        "g3c4-p2g2-native-terminal-ids-clone",
    )):
        return False
    return True


assert valid(source)
mutations = (
    ("(g3c4-p2g2-terminal-pending-check call anchor operation)", "#t"),
    ("(eq? (vector-ref record 11) raw)", "(eq? (vector-ref record 9) raw)"),
    ("(eq? (vector-ref record 7) output-native)", "#t"),
    ("(g3c4-p2g2-prefix-byte-check record selected operation)", "#t"),
    ("native output-native raw", "native output-native staging"),
    ("(vector-set! protected-anchor 13 'prefix-committed)", "#t"),
    ("(vector-set! (vector-ref record 14) 3 'p2g2-prefix-committed)", "#t"),
)
for before, after in mutations:
    assert before in source and not valid(source.replace(before, after, 1)), before
assert not valid(source + "\n(g3c4-p2g2-native-terminal-commit native output)\n")
print("G3-C4 P2/G2 terminal non-EOS prefix source PASS: 8 negative mutations")
