#!/usr/bin/env python3
"""Source-only checks for the private native-zero boundary and closed tail."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3c4_p2g2_terminal_precommit_frame_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3c4_p2g2_terminal_commit_extension.esk",
    "scripts/check-g3c4-p2g2-terminal-commit-source.py",
    "scripts/check-g3c4-p2g2-terminal-null-ffi-source.py",
    "scripts/test-g3c4-p2g2-terminal-commit-source.sh",
    "native/g3c4_p2g2_terminal_commit_source_closure.txt",
]
closure = (root / additions[-1]).read_text().splitlines()
assert closure == base + additions and len(closure) == len(set(closure)) == 218
assert all((root / path).is_file() for path in closure)
source = (root / additions[0]).read_text()
abort_only = (root / "native/g3c4_p2g2_terminal_abort_coordinator_extension.esk").read_text()
call_entry = (root / "native/g3c4_call_entry_extension.esk").read_text()


def clean(text):
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def form_end(text, start):
    if text[start] != "(":
        return -1
    depth = 0
    for index in range(start, len(text)):
        depth += (text[index] == "(") - (text[index] == ")")
        if depth == 0:
            return index + 1
        if depth < 0:
            return -1
    return -1


def valid(text):
    code = clean(text)
    if form_end(code, code.index("(define (g3c4-p2g2-terminal-commit-private!")) != len(code.rstrip()):
        return False
    required = (
        "(extern i64 g3c4-p2g2-native-terminal-commit ptr ptr ptr i64 ptr i64",
        ":real et_g3c4_private_p2g2_terminal_commit_v1)",
        "(g3c4-p2g2-prefix-preflight generator input)",
        "(m3-call operation",
        "(barrier (vector #f))",
        "(registry-node (cons call (vector-ref g3c4-registry 0)))",
        "(record-node",
        "(protected-node",
        "(prefix-node",
        "(if (vector-ref barrier 0) (exit 134))",
        "(g3c4-p2g2-terminal-pending-abort!\n                                   call pending)",
        "(g3c4-p2g2-protected-abort! call)",
        "(g3c4-native-call-acquire native 2 2)",
        "(set! pending (g3c4-p2g2-terminal-pending-reserve! call))",
        "(g3c4-p2g2-terminal-first-sample-decode! call pending)",
        "(g3c4-p2g2-terminal-non-eos-prefix-commit!",
        "(g3c4-p2g2-terminal-second-sample-decode!",
        "(g3c4-p2g2-terminal-text-preflight!",
        "(g3c4-p2g2-terminal-precommit-frame",
        "(g3c4-p2g2-native-terminal-commit",
        "(vector-ref frame 7) (vector-ref frame 8)",
        "(vector-ref frame 9) (vector-ref frame 10)",
        "(if (= status 0)",
        "(vector-set! barrier 0 #t)",
        "(vector-set! output 1 'terminal-output)",
        "(vector-set! output 2 'live)",
        "(vector-set! pending 0 'live)",
        "(vector-set! g3c4-p2g2-terminal-pending-originals 0",
        "(vector-set! ledger 4 #f)",
        "(vector-set! model-entry 10 #f)",
        "(vector-set! generator-entry 9 #f)",
        "(g3c4-call-dead! call)",
        "(vector-set! g3c4-p2g2-protected-records 0\n                               protected-records-next)",
        "(vector-set! g3c4-p2g2-protected-anchors 0\n                               protected-anchors-next)",
        "(vector-set! g3c4-p2g2-prefix-input-anchors 0 prefix-next)",
        "(g3c4-native-check status operation)",
    )
    if any(code.count(fragment) != (2 if fragment == "(g3c4-call-dead! call)" else 1)
           for fragment in required):
        return False
    order = (
        "(barrier (vector #f))",
        "(guard (caught",
        "(g3c4-native-call-acquire native 2 2)",
        "(g3c4-p2g2-terminal-pending-reserve! call)",
        "(g3c4-p2g2-terminal-first-sample-decode! call pending)",
        "(g3c4-p2g2-terminal-text-preflight!",
        "(g3c4-p2g2-terminal-precommit-frame",
        "(g3c4-p2g2-native-terminal-commit",
        "(if (= status 0)",
        "(vector-set! barrier 0 #t)",
        "(vector-set! output 2 'live)",
        "(vector-set! pending 0 'live)",
        "(g3c4-call-dead! call)",
        "(vector-set! g3c4-p2g2-prefix-input-anchors 0 prefix-next)",
        "(g3c4-native-check status operation)",
    )
    positions = [code.rindex(part) if part == "(g3c4-call-dead! call)"
                 else code.index(part) for part in order]
    if positions != sorted(positions):
        return False
    handler = code[code.index("(guard (caught"):code.index("(vector-set! g3c4-registry 0 registry-node)")]
    if not (handler.index("(if (vector-ref barrier 0) (exit 134))") <
            handler.index("(g3c4-p2g2-terminal-pending-abort!")):
        return False
    success_start = code.index("(begin\n                  (vector-set! barrier 0 #t)")
    success = code[success_start:form_end(code, success_start)]
    if not success.startswith("(begin\n                  (vector-set! barrier 0 #t)"):
        return False
    forbidden = ("g3c4-fail", "g3c4-native-check", "g3c4-p2g2-terminal-pending-check",
                 "g3c4-p2g2-terminal-pending-abort!", "g3c4-native-call-finish",
                 "g3c4-p2g2-zero-bytes!", "make-bytevector", "cons ",
                 "guard ", "raise ", "extern ")
    if any(fragment in success for fragment in forbidden):
        return False
    for fragment in (
        "(bytevector-u8-set! first-raw 0 0)",
        "(bytevector-u8-set! second-raw 0 0)",
        "(vector-set! output 4 #f)", "(vector-set! output 5 #f)",
        "(vector-set! output 6 #f)", "(vector-set! output 8 selected-text)",
        "(vector-set! pending 1 #f)", "(vector-set! pending 2 #f)",
        "(vector-set! pending 3 #f)", "(vector-set! pending 16 #f)",
        "(vector-set! pending 17 #f)", "(vector-set! pending 18 #f)",
        "(vector-set! record 0 #f)", "(vector-set! protected-anchor 0 #f)",
    ):
        if fragment not in success:
            return False
    for carrier in ("first-staging", "second-staging"):
        for index in range(8):
            if success.count(f"(bytevector-u8-set! {carrier} {index} 0)") != 1:
                return False
    return success.rstrip().endswith("shell)")


assert valid(source)
assert "g3c4-p2g2-native-terminal-commit" not in clean(abort_only)
call_dead = clean(call_entry)
call_dead = call_dead[call_dead.index("(define (g3c4-call-dead! entry)"):]
call_dead = call_dead[:form_end(call_dead, 0)]
assert call_dead.count("(vector-set! entry ") == 10
assert not any(item in call_dead for item in (
    "g3c4-fail", "g3c4-native", "make-", "cons ", "guard ", "raise ",
))
mutations = (
    ("(barrier (vector #f))", "(barrier #f)"),
    ("(if (vector-ref barrier 0) (exit 134))", "#t"),
    ("(g3c4-p2g2-terminal-pending-abort!\n                                   call pending)", "#t"),
    ("(g3c4-p2g2-protected-abort! call)", "#t"),
    ("(g3c4-native-call-acquire native 2 2)", "(g3c4-native-call-acquire native 2 1)"),
    ("(set! pending (g3c4-p2g2-terminal-pending-reserve! call))", "#t"),
    ("(g3c4-p2g2-terminal-precommit-frame", "(begin"),
    ("(vector-ref frame 9) (vector-ref frame 10)", "#f 9"),
    ("(if (= status 0)", "(if #t"),
    ("(vector-set! barrier 0 #t)", "#t"),
    ("(vector-set! output 1 'terminal-output)", "(vector-set! output 1 'output)"),
    ("(vector-set! pending 0 'live)", "(vector-set! pending 0 'pending)"),
    ("(vector-set! pending 16 #f)", "#t"),
    ("(bytevector-u8-set! second-staging 7 0)", "#t"),
    ("(vector-set! g3c4-p2g2-terminal-pending-originals 0", "(begin"),
    ("(vector-set! model-entry 10 #f)", "#t"),
    ("(g3c4-call-dead! call)", "#t"),
    ("(vector-set! g3c4-p2g2-prefix-input-anchors 0 prefix-next)", "#t"),
)
for before, after in mutations:
    assert before in source and not valid(source.replace(before, after, 1)), before
assert not valid(source.replace("(vector-set! barrier 0 #t)",
                                "(vector-set! barrier 0 #t)\n                  (g3c4-fail 'internal operation \"fault\")", 1))
assert not valid(source.replace("(vector-set! barrier 0 #t)",
                                "(vector-set! barrier 0 #t)\n                  (g3c4-p2g2-terminal-pending-abort! call pending)", 1))
print("G3-C4 P2/G2 terminal commit source PASS: 20 negative mutations")
