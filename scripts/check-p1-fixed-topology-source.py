#!/usr/bin/env python3
"""Check the inert P1 slot-73 source contract and build a test-only overlay."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
source = (ROOT / "internal/p1/lib/transformer/module.esk").read_text()
probe = (ROOT / "tests/p1/fixed_topology_slot73_probe.esk").read_text()
fixture = (ROOT / "tests/p1/fixed_topology_slot73_test.esk").read_text()
native_header = (ROOT / "native/p1_identity_internal.h").read_text()

start = source.index("(define (p1-fixed-seal-fail)")
end = source.index("\n(vector\n  (lambda (module)", start)
verifier = source[start:end]
for required in (
    "p1-fixed-sealed-record", "p1-fixed-scoped-registry?",
    "p1-fixed-expected-member-count", "p1-fixed-shell-registry-count",
    "p1-fixed-handle-path?", "p1-fixed-shape?",
    "p1-fixed-list-length?", "construction-registry-root",
    "shell-registry-current", "p1-fixed-count?",
    "p1-fixed-native-shell?", "p1-fixed-native-token-kind",
    "p1-fixed-native-token-live",
):
    assert required in verifier, required
for forbidden in ("(extern ", "(cons ", "(make-vector ", "(vector-set! ",
                  "(module-parameters ", "(shell-for-raw ", "(native-check ",
                  "(tensor-provider-call ", "(set-cdr! ",
                  "(hash-table-set! "):
    assert forbidden not in verifier, forbidden
assert "(and (integer? value) (exact? value) (= value expected))" in verifier
assert source.count(":real et_p1_public_token_kind_v1") == 1
assert source.count(":real et_p1_public_token_live_v1") == 1
assert "ET_P1_TOKEN_MODULE = 2" in native_header
assert "ET_P1_TOKEN_PARAMETER_HANDLE = 3" in native_header
for manifest in ("p1_identity_public_symbols.txt",
                 "p1_identity_trusted_symbols.txt"):
    symbols = (ROOT / "native" / manifest).read_text().splitlines()
    assert symbols.count("et_p1_public_token_kind_v1") == 1
    assert symbols.count("et_p1_public_token_live_v1") == 1
assert "(p1-fixed-native-shell? (vector-ref entry 0) kind)" in verifier
assert "(if (eq? kind 'module) 2 3)" in verifier
for slot, expected in ((9, 15), (10, 17), (11, 4), (12, 5)):
    assert verifier.count(
        f"(p1-fixed-count? (vector-ref root {slot}) {expected})") == 1
assert source.count("Append-only P1-only sealed current-root verifier, slot 73") == 1
for action in ("detached-module", "detached-handle", "duplicate-shell",
               "duplicate-record", "wrong-parent", "wrong-shape",
               "wrong-device", "wrong-counter-type", "wrong-tie",
               "native-wrong-kind", "native-tombstone",
               "native-module-wrong-kind", "native-module-tombstone"):
    assert f"(eq? action '{action})" in probe
    assert action in fixture
assert "(reject (car actions)" in fixture
assert "native identity restored same-root retry" in fixture
assert "native module restored same-root retry" in fixture

anchor = "\n(vector\n  (lambda (module)"
assert source.count(anchor) == 1
overlay = source.replace(anchor, "\n" + probe + anchor, 1)
tail = ("  ;; Append-only P1-only sealed current-root verifier, slot 73.\n"
        "  (lambda (root-shell expected-trie expected-handles)\n"
        "    (p1-fixed-sealed-topology!\n"
        "      root-shell expected-trie expected-handles))\n"
        "  )))")
assert overlay.count(tail) == 1
overlay = overlay.replace(
    tail, tail.replace(
        "  )))",
        "  ;; Test-only mutation and restoration, slots 74..75.\n"
        "  (lambda (action root-shell handle-shell)\n"
        "    (p1-fixed-test-mutate! action root-shell handle-shell))\n"
        "  (lambda () (p1-fixed-test-restore!))\n"
        "  )))"),
    1,
)

def balanced(text: str) -> bool:
    depth = 0
    quoted = escaped = comment = False
    for char in text:
        if char == "\n":
            comment = False
            continue
        if comment:
            continue
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == ";":
            comment = True
        elif char == '"':
            quoted = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not quoted

assert balanced(source)
assert balanced(overlay)
assert balanced(fixture)
if len(sys.argv) == 3 and sys.argv[1] == "--output":
    output = Path(sys.argv[2])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(overlay)
elif len(sys.argv) != 1:
    raise SystemExit("usage: check-p1-fixed-topology-source.py [--output FILE]")
print("P1 slot73 source: PASS")
