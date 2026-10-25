#!/usr/bin/env python3
"""Guard the private C4 same-region ledger and its closed tail."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3c4_call_entry_extension.esk").read_text()
scope = (root / "native/g3c4_p2g2_protected_scope_extension.esk").read_text()
prefix = (root / "native/g3c4_p2g2_t1_prefix_coordinator_extension.esk").read_text()
ledger = (root / "native/g3c4_p2g2_stable_prefix_ledger_extension.esk").read_text()
fixture = (root / "tests/g3c4/p2_g2_stable_prefix_ledger_test.esk").read_text()
feature_off = (root / "tests/g3c4/p2_g2_stable_prefix_ledger_feature_off_test.esk").read_text()
contract = (root / "docs/g3/G3_R_C4_ESHKOL_STABLE_PREFIX_LEDGER_CONTRACT.md").read_text()


def balanced(source: str) -> bool:
    depth = 0
    quoted = False
    escaped = False
    for line in source.splitlines():
        for char in line:
            if not quoted and char == ";":
                break
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    return False
    return depth == 0 and not quoted


def valid(candidate: str) -> bool:
    if not balanced(candidate):
        return False
    start = candidate.find("(define (g3c4-p2g2-stable-prefix-close! call input)")
    if start < 0:
        return False
    close = candidate[start:]
    native = close.find("(g3c4-p2g2-native-stable-close")
    guard = close.find("(guard (caught (#t (if (vector-ref tail-started 0)")
    tail = close.find("(vector-set! tail-started 0 #t)")
    dead = close.find("(g3c4-call-dead! call)")
    install = close.find("(vector-set! generator-entry 8 stable-record)")
    if not (0 <= guard < native < tail < dead < install):
        return False
    if any(word in close[tail:] for word in ("(cons ", "(make-", "(vector ", "(g3c4-p2g2-native-")):
        return False
    required = (
        "(g3c4-p2g2-protected-check call operation)",
        "(g3c4-p2g2-prefix-byte-check record selected operation)",
        "(eq? (vector-ref ledger 4) output)",
        "(eq? (vector-ref input-anchor 1) input)",
        "(g3c4-entry-dead! output)",
        "(vector-set! model-entry 10 #f)",
        "(vector-set! generator-entry 9 #f)",
        "(vector-set! generator-entry 10 stable-anchor)",
        "(cdr (vector-ref g3c4-p2g2-protected-records 0))",
        "(cdr (vector-ref g3c4-p2g2-protected-anchors 0))",
    )
    return all(item in close for item in required)


for source in (base, scope, prefix, ledger, fixture, feature_off):
    assert balanced(source)
assert valid(ledger)
assert "g3c4-p2g2-stable-ledger-gate (vector #f)" in base
assert "(g3c4-p2g2-stable-slots? entry)" in base
assert "C4 stable prefix needs a separate idle-P3 entry" in base
assert "(vector-set! g3c4-p2g2-stable-ledger-gate 0 #t)" in ledger
assert "(g3c4-p2g2-native-stable-check" in ledger
assert "(g3c4-with-p2g2-prefix-call" in fixture
assert "(load \"g3c4_p2g2_stable_prefix_ledger_extension.esk\")" not in feature_off
assert "stable ledger gate remains off" in feature_off
assert "ordinary slots remain empty" in feature_off
for control in (
    "foreign P2 input rejected", "changed phase rejected",
    "changed raw byte rejected", "held I1 rejects close",
    "held A2 rejects close", "swapped stable input rejected",
    "changed policy rejected", "generator close retires stable owner",
    "all transient byte carriers scrubbed",
):
    assert control in fixture, control
assert "No compiled Eshkol admission" in contract
assert "SAVE hold" in contract and "idle-P3" in contract

mutants = (
    ledger.replace("(guard (caught (#t (if (vector-ref tail-started 0)",
                   "(guard (caught (#t (if #f", 1),
    ledger.replace("(g3c4-entry-dead! output)", "#t", 1),
    ledger.replace("(vector-set! model-entry 10 #f)", "#t", 1),
    ledger.replace("(vector-set! generator-entry 8 stable-record)", "#t", 1),
    ledger.replace("(g3c4-p2g2-native-stable-close", "(native-missing-close", 1),
    ledger.replace("(g3c4-p2g2-prefix-byte-check record selected operation)", "#t", 1),
    ledger.replace("(g3c4-call-dead! call)", "(cons call '())", 1),
)
assert all(not valid(mutant) for mutant in mutants)
print("G3-C4 stable-prefix ledger source/mutation PASS: checks=25 mutants=7")
