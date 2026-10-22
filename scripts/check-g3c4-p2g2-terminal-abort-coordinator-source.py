#!/usr/bin/env python3
"""Source-only closure and mutation checks for same-call abort composition."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3c4_p2g2_terminal_text_preflight_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3c4_p2g2_terminal_abort_coordinator_extension.esk",
    "scripts/check-g3c4-p2g2-terminal-abort-coordinator-source.py",
    "scripts/test-g3c4-p2g2-terminal-abort-coordinator-source.sh",
    "native/g3c4_p2g2_terminal_abort_coordinator_source_closure.txt",
]
closure = (root / additions[-1]).read_text().splitlines()
assert closure == base + additions and len(closure) == len(set(closure)) == 209
assert all((root / name).is_file() for name in closure)
coordinator = (root / additions[0]).read_text()
reservation = (root / "native/g3c4_p2g2_terminal_extension.esk").read_text()


def code_only(text):
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def balanced(text):
    depth = 0
    for char in text:
        depth += (char == "(") - (char == ")")
        if depth < 0:
            return False
    return depth == 0


def form_end(text, start):
    assert text[start] == "("
    depth = 0
    for index in range(start, len(text)):
        depth += (text[index] == "(") - (text[index] == ")")
        if depth == 0:
            return index + 1
    return -1


def valid_coordinator(text):
    code = code_only(text)
    if not balanced(code):
        return False
    required = {
        "(g3c4-with-p2g2-prefix-call operation generator input (call)": 1,
        "(g3c4-p2g2-terminal-pending-reserve! call)": 1,
        "(g3c4-p2g2-terminal-first-sample-decode! call anchor)": 1,
        "(g3c4-p2g2-terminal-pending-check call anchor operation)": 1,
        "(m3t-exact-i64? eos)": 1,
        "(= first-selected eos)": 1,
        "call anchor first-selected #f)": 1,
        "(g3c4-p2g2-terminal-non-eos-prefix-commit!": 1,
        "(g3c4-p2g2-terminal-second-sample-decode!": 1,
        "call anchor first-selected second-selected)": 1,
        "(g3c4-p2g2-terminal-pending-abort! call anchor)": 2,
        "(raise caught)": 1,
        "(exit 134)": 2,
    }
    if any(code.count(fragment) != count for fragment, count in required.items()):
        return False
    order = (
        "(g3c4-with-p2g2-prefix-call",
        "(g3c4-p2g2-terminal-pending-reserve! call)",
        "(guard (caught",
        "(g3c4-p2g2-terminal-first-sample-decode! call anchor)",
        "(g3c4-p2g2-terminal-pending-check call anchor operation)",
        "(if (and (m3t-exact-i64? eos) (= first-selected eos))",
        "call anchor first-selected #f)",
        "(g3c4-p2g2-terminal-non-eos-prefix-commit!",
        "(g3c4-p2g2-terminal-second-sample-decode!",
        "call anchor first-selected second-selected)",
        "(guard (abort-failure (#t (exit 134)))\n"
        "          (g3c4-p2g2-terminal-pending-abort! call anchor))",
    )
    if list(map(code.index, order)) != sorted(map(code.index, order)):
        return False
    if code.index("(g3c4-p2g2-terminal-pending-abort! call anchor)") > code.index("(raise caught)"):
        return False
    failure_guard_start = code.index("(guard (caught")
    failure_guard = code[failure_guard_start:form_end(code, failure_guard_start)]
    if "(g3c4-p2g2-terminal-first-sample-decode! call anchor)" not in failure_guard:
        return False
    if "(g3c4-p2g2-terminal-text-preflight!" not in failure_guard:
        return False
    if any(name in code for name in (
        "g3c4-p2g2-native-terminal-commit",
        "g3c4-p2g2-native-terminal-snapshot",
        "g3c4-p2g2-native-terminal-ids-clone",
        "g3c4-entry-live!", "g3c4-call-dead!",
    )):
        return False
    return code.rstrip().endswith("#t))))")


def valid_reservation(text):
    code = code_only(text)
    if not balanced(code):
        return False
    start = code.index("(define (g3c4-p2g2-terminal-pending-reserve!")
    end = code.index("(define (g3c4-p2g2-terminal-pending-abort!", start)
    body = code[start:end]
    order = (
        "(vector-set! g3c4-p2g2-terminal-pending-anchors 0 anchor-node)",
        "(vector-set! g3c4-p2g2-terminal-pending-originals 0 original-node)",
        "(guard (caught",
        "(g3c4-p2g2-terminal-pending-abort! call anchor)",
        "(raise caught)",
        "(g3c4-p2g2-terminal-pending-check call anchor operation)",
    )
    if any(body.count(fragment) != 1 for fragment in order):
        return False
    if list(map(body.index, order)) != sorted(map(body.index, order)):
        return False
    guard_start = body.index("(guard (caught")
    guard = body[guard_start:form_end(body, guard_start)]
    return all(fragment in guard for fragment in (
        "(guard (abort-failure (#t (exit 134)))",
        "(g3c4-p2g2-terminal-pending-abort! call anchor)",
        "(raise caught)",
        "(g3c4-p2g2-terminal-pending-check call anchor operation)",
    ))


assert valid_coordinator(coordinator)
assert valid_reservation(reservation)
coordinator_mutations = (
    ("(g3c4-with-p2g2-prefix-call operation generator input (call)", "(begin"),
    ("(g3c4-p2g2-terminal-pending-reserve! call)", "#f"),
    ("(g3c4-p2g2-terminal-first-sample-decode! call anchor)", "0"),
    ("(m3t-exact-i64? eos)", "#t"),
    ("(= first-selected eos)", "#t"),
    ("call anchor first-selected #f)", "call anchor first-selected 0)"),
    ("(g3c4-p2g2-terminal-non-eos-prefix-commit!", "(begin"),
    ("(g3c4-p2g2-terminal-second-sample-decode!", "(begin"),
    ("call anchor first-selected second-selected)", "call anchor first-selected #f)"),
    ("(raise caught)", "#t"),
)
for before, after in coordinator_mutations:
    assert before in coordinator and not valid_coordinator(coordinator.replace(before, after, 1)), before
for before, after in (
    ("(g3c4-p2g2-terminal-pending-abort! call anchor)", "#t"),
    ("(raise caught)", "#t"),
    ("(guard (abort-failure (#t (exit 134)))", "(begin"),
):
    assert before in reservation and not valid_reservation(reservation.replace(before, after, 1)), before
assert not valid_reservation(reservation.replace(
    "(g3c4-p2g2-terminal-pending-check call anchor operation)))))",
    ")\n      (g3c4-p2g2-terminal-pending-check call anchor operation))))",
    1,
))
assert not valid_coordinator(coordinator + "\n(g3c4-p2g2-native-terminal-commit native output)\n")
print("G3-C4 P2/G2 abort coordinator source PASS: 15 negative mutations")
