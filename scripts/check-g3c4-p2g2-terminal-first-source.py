#!/usr/bin/env python3
"""Source-only contract and mutation checks for the abort-only first token."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3c4_p2g2_terminal_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3c4_p2g2_terminal_first_extension.esk",
    "scripts/check-g3c4-p2g2-terminal-first-source.py",
    "scripts/test-g3c4-p2g2-terminal-first-source.sh",
    "native/g3c4_p2g2_terminal_first_source_closure.txt",
]
closure = (root / additions[-1]).read_text().splitlines()
assert closure == base + additions and len(closure) == len(set(closure))
assert all((root / name).is_file() for name in closure)
source = (root / additions[0]).read_text()


def code_only(text: str) -> str:
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def balanced(text: str) -> bool:
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def valid(text: str) -> bool:
    code = code_only(text)
    if not balanced(code):
        return False
    required = [
        ":real et_g3c4_private_p2g2_eos_first_frame_carrier_v1",
        "(g3c4-p2g2-terminal-pending-check call anchor operation)",
        "(g3c4-p2g2-native-eos-first-frame native native-input staging 16)",
        "(eq? (vector-ref anchor 7) raw)",
        "(eq? (vector-ref anchor 8) staging)",
        "(eq? (vector-ref prefix 3) native-input)",
        "(eq? (vector-ref record 5) native)",
        "(eq? (vector-ref record 3) tokenizer)",
        "(t1-private-g3-decode-raw-into! tokenizer staging raw)",
        "(= (bytevector-u8-ref raw 0) selected)",
        "(= (bytevector-u8-ref staging 0) selected)",
    ]
    if any(item not in code for item in required):
        return False
    if code.count(required[1]) != 5 or code.count("(let upper ((index 1))") != 2:
        return False
    if code.count("(eq? (vector-ref anchor 7) raw)") != 2:
        return False
    if code.count("(eq? (vector-ref anchor 8) staging)") != 2:
        return False
    if any(item in code for item in (
        "g3c4-p2g2-native-first-frame",
        "g3c4-p2g2-native-prefix-commit",
        "g3c4-p2g2-native-terminal-commit",
        "g3c4-p2g2-native-terminal-snapshot",
        "g3c4-p2g2-native-terminal-ids-clone",
    )):
        return False
    order = (
        "(g3c4-p2g2-native-eos-first-frame native native-input staging 16)",
        "(let ((selected (bytevector-u8-ref staging 0)))",
        "(before-decode anchor)",
        "(t1-private-g3-decode-raw-into! tokenizer staging raw)",
        "(after-decode anchor)",
        "(= (bytevector-u8-ref raw 0) selected)",
    )
    return list(map(code.index, order)) == sorted(map(code.index, order))


assert valid(source)
mutations = (
    ("et_g3c4_private_p2g2_eos_first_frame_carrier_v1",
     "et_g3c4_private_p2g2_first_frame_carrier_v1"),
    ("native native-input staging 16", "native native-input staging 8"),
    ("(t1-private-g3-decode-raw-into! tokenizer staging raw)",
     "(bytevector-u8-set! raw 0 selected)"),
    ("(eq? (vector-ref anchor 7) raw)", "(eq? (vector-ref anchor 9) raw)"),
    ("(let upper ((index 1))", "(let upper ((index 8))"),
    ("(g3c4-p2g2-terminal-pending-check call anchor operation)", "#t"),
)
for before, after in mutations:
    assert before in source and not valid(source.replace(before, after, 1)), before
assert not valid(source + "\n(g3c4-p2g2-native-prefix-commit native output raw)\n")
print("G3-C4 P2/G2 terminal first-token source PASS: 7 negative mutations")
