#!/usr/bin/env python3
"""Source-only identity and descriptor checks for detached terminal snapshot."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "native/g3c4_p2g2_terminal_snapshot_extension.esk").read_text()
base = (root / "native/g3c4_p2g2_terminal_commit_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3c4_p2g2_terminal_snapshot_extension.esk",
    "scripts/check-g3c4-p2g2-terminal-snapshot-source.py",
    "scripts/test-g3c4-p2g2-terminal-snapshot-source.sh",
    "native/g3c4_p2g2_terminal_snapshot_source_closure.txt",
]
closure = (root / additions[-1]).read_text().splitlines()
assert closure == base + additions and len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure)


def code_only(text):
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def valid(text):
    code = code_only(text)
    depth = 0
    for char in code:
        depth += (char == "(") - (char == ")")
        if depth < 0:
            return False
    if depth != 0:
        return False
    required = (
        ":real et_g3c4_private_p2g2_terminal_snapshot_v1",
        "(m3-call operation",
        "(g3c4-p2g2-terminal-output-entry-live",
        "(g3c4-p2g2-terminal-snapshot-live-for",
        "(bytes (make-bytevector 72 0))",
        "(bytevector-u8-set! bytes 0 64)",
        "(g3c4-p2g2-native-terminal-snapshot native bytes 72)",
        "(= (bytevector-length bytes) 72)",
        "(= (bytevector-u8-ref bytes 0) 64)",
        "(g3c4-p2g2-terminal-snapshot-small-word bytes 8)",
        "(g3c4-p2g2-terminal-snapshot-small-word bytes 16)",
        "(g3c4-p2g2-terminal-snapshot-small-word bytes 24)",
        "(g3c4-p2g2-terminal-snapshot-small-word bytes 32)",
        "(or (= count 1) (= count 2))",
        "(= count (vector-ref live 6))",
        "(= cache (vector-ref live 15))",
        "(= id-zero (vector-ref live 8))",
        "(= id-one 0)",
        "(= id-one (vector-ref live 9))",
        "bytes))))",
    )
    if any(fragment not in code for fragment in required):
        return False
    if code.count("(g3c4-p2g2-terminal-output-entry-live") != 3:
        return False
    if code.count("(g3c4-p2g2-terminal-snapshot-live-for") != 4:
        return False
    if code.count("(eq? native (vector-ref output 3))") != 2:
        return False
    if code.count("(= (bytevector-length bytes) 72)") != 2:
        return False
    if any(fragment in code for fragment in (
        "g3c4-p2g2-native-terminal-commit",
        "g3c4-native-output-release",
        "g3c4-public-generation-output",
    )):
        return False
    before = code.index("(bytevector-u8-set! bytes 0 64)")
    native = code.index("(g3c4-p2g2-native-terminal-snapshot native bytes 72)")
    after = code.rindex("(g3c4-p2g2-terminal-output-entry-live")
    compare = code.index("(= count (vector-ref live 6))")
    return before < native < after < compare


assert valid(source)
mutations = (
    ("make-bytevector 72 0", "make-bytevector 64 0"),
    ("bytes 0 64", "bytes 0 63"),
    ("native bytes 72", "native bytes 64"),
    ("(= count (vector-ref live 6))", "#t"),
    ("(= cache (vector-ref live 15))", "#t"),
    ("(= id-zero (vector-ref live 8))", "#t"),
    ("(= id-one 0)", "#t"),
    ("(= id-one (vector-ref live 9))", "#t"),
    ("(eq? native (vector-ref output 3))", "#t"),
    ("(g3c4-p2g2-terminal-output-entry-live", "(g3c4-registry-find"),
    ("(g3c4-p2g2-terminal-snapshot-live-for", "(vector"),
)
for before, after in mutations:
    assert before in source and not valid(source.replace(before, after, 1)), before
print(f"G3-C4 terminal detached snapshot source PASS: {len(mutations)} negative mutations")
