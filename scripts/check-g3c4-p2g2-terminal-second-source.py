#!/usr/bin/env python3
"""Source-only closure and mutation checks for the second pending token."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3c4_p2g2_terminal_prefix_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3c4_p2g2_terminal_second_extension.esk",
    "scripts/check-g3c4-p2g2-terminal-second-source.py",
    "scripts/test-g3c4-p2g2-terminal-second-source.sh",
    "native/g3c4_p2g2_terminal_second_source_closure.txt",
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
        ":real et_g3c4_private_p2g2_second_frame_carrier_v1",
        "(g3c4-p2g2-terminal-pending-check call anchor operation)",
        "(eq? (vector-ref record 13) 'prefix-committed)",
        "(eq? (vector-ref protected-anchor 13) 'prefix-committed)",
        "(eq? (vector-ref record 7) output-native)",
        "(eq? (vector-ref anchor 9) raw)",
        "(eq? (vector-ref anchor 10) staging)",
        "(g3c4-p2g2-prefix-byte-check",
        "(g3c4-p2g2-native-second-frame native output-native staging 16)",
        "(g3c4-p2g2-terminal-second-id-check staging selected operation)",
        "(t1-private-g3-decode-raw-into! tokenizer staging raw)",
        "(= (bytevector-u8-ref raw 0) selected)",
        "(let ((selected (bytevector-u8-ref staging 0)))",
        "(= (bytevector-u8-ref staging 0) selected)",
        "(= (bytevector-u8-ref staging index) 0)",
    )
    counts = (1, 5, 1, 1, 2, 3, 3, 1, 1, 4, 1, 1, 1, 1, 1)
    if any(code.count(item) != count for item, count in zip(required, counts)):
        return False
    if code.count("(let upper ((index 1))") != 1:
        return False
    body = code[code.index("(define (g3c4-p2g2-terminal-second-sample-decode-probed!"):]
    order = (required[1], required[2], required[7], required[8],
             required[12], required[9], "(before-decode anchor)", required[10],
             "(after-decode anchor)", required[11])
    if list(map(body.index, order)) != sorted(map(body.index, order)):
        return False
    if any(name in code for name in (
        "g3c4-p2g2-native-first-frame",
        "g3c4-p2g2-native-prefix-commit",
        "g3c4-p2g2-native-terminal-commit",
        "g3c4-p2g2-native-terminal-snapshot",
        "g3c4-p2g2-native-terminal-ids-clone",
    )):
        return False
    return True


assert valid(source)
mutations = (
    ("et_g3c4_private_p2g2_second_frame_carrier_v1",
     "et_g3c4_private_p2g2_first_frame_carrier_v1"),
    ("(eq? (vector-ref record 13) 'prefix-committed)", "#t"),
    ("(eq? (vector-ref anchor 9) raw)", "(eq? (vector-ref anchor 7) raw)"),
    ("(g3c4-p2g2-prefix-byte-check", "(g3c4-p2g2-prefix-byte-skip"),
    ("native output-native staging 16", "native output-native staging 8"),
    ("(let upper ((index 1))", "(let upper ((index 8))"),
    ("(= (bytevector-u8-ref staging index) 0)", "#t"),
    ("(= (bytevector-u8-ref staging 0) selected)", "#t"),
    ("(let ((selected (bytevector-u8-ref staging 0)))",
     "(let ((selected (bytevector-u8-ref raw 0)))"),
    ("(t1-private-g3-decode-raw-into! tokenizer staging raw)",
     "(bytevector-u8-set! raw 0 selected)"),
    ("(= (bytevector-u8-ref raw 0) selected)", "#t"),
)
for before, after in mutations:
    assert before in source and not valid(source.replace(before, after, 1)), before
assert not valid(source + "\n(g3c4-p2g2-native-terminal-commit native output)\n")
print("G3-C4 P2/G2 terminal second-token source PASS: 12 negative mutations")
