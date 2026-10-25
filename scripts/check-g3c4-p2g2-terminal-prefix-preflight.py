#!/usr/bin/env python3
"""Source-only dependency and mutation checks for committed-prefix identity."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
source_path = root / "native/g3c4_p2g2_terminal_prefix_preflight_extension.esk"
fixture_path = root / "tests/g3c4/p2_g2_terminal_prefix_preflight_test.esk"
source = source_path.read_text()
fixture = fixture_path.read_text()


def contract(candidate):
    required = (
        "(g3c4-p2g2-terminal-pending-check call operation)",
        "(stack (vector-ref g3c4-p2g2-prefix-input-anchors 0))",
        "(prefix (if (pair? stack) (car stack) #f))",
        "(pair? stack) (eq? (car stack) prefix)",
        "(eq? (vector-ref prefix 1) input)",
        "(g3c4-input-entry-live input operation)",
        "(eq? (vector-ref (vector-ref prefix 2) 3)",
        "(eq? (vector-ref record 13) 'prefix-committed)",
        "(eq? (vector-ref (g3c4-p2g2-anchor-for call) 13)",
        "'p2g2-prefix-committed)",
        "(eq? (vector-ref pending 16) 'pending)",
    )
    forbidden = (
        "(extern ", "g3c4-p2g2-native-", "g3c4-native-",
        "g3c4-p2g2-prefix-byte-check", "t1-private-g3-decode-raw-into!",
        "bytevector-u8-ref", "bytevector-u8-set!",
        "g3c4-p2g2-terminal-commit", "g3c4-p2g2-terminal-snapshot",
    )
    return all(item in candidate for item in required) and not any(
        item in candidate for item in forbidden
    )


assert contract(source)
mutants = (
    ("(g3c4-p2g2-terminal-pending-check call operation)", "#t"),
    ("(prefix (if (pair? stack) (car stack) #f))", "(prefix #f)"),
    ("(eq? (vector-ref prefix 1) input)", "#t"),
    ("(g3c4-input-entry-live input operation)", "input"),
    ("(eq? (vector-ref record 13) 'prefix-committed)", "#t"),
    ("'p2g2-prefix-committed)", "'p2g2-protected)"),
    ("(eq? (vector-ref pending 16) 'pending)", "#t"),
)
for before, after in mutants:
    assert before in source
    assert not contract(source.replace(before, after, 1)), before

for name in (
    "before-prefix is not terminal-ready",
    "foreign original input rejected",
    "malformed top input anchor rejected",
    "changed live input binding rejected",
    "changed original input anchor rejected",
    "ledger phase drift rejected",
    "repair preserves original committed admission",
    "authenticated abort",
):
    assert name in fixture, name

assert fixture.index("terminal-pending-prepare call") < fixture.index(
    "g3c4-p2g2-prefix-commit! call input"
)
assert fixture.index("g3c4-p2g2-prefix-commit! call input") < fixture.index(
    "authenticated abort"
)
assert source.count("(") == source.count(")")
assert fixture.count("(") == fixture.count(")")
print(f"G3-C4 P2/G2 committed-prefix source PASS: mutants={len(mutants)}")
