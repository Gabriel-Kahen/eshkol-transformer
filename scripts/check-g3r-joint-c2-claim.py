#!/usr/bin/env python3
"""Mutation gate for the unlinked original-owner C2 claim prerequisite."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
native = (root / "native/g3r_joint_hold.c").read_text()
c2 = (root / "native/c2_training_state_extension.esk").read_text()
claim = (root / "native/g3r_joint_c2_claim_extension.esk").read_text()


def balanced(source: str) -> bool:
    depth = 0
    quoted = escaped = comment = False
    for char in source:
        if char == "\n":
            comment = False
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
            assert depth >= 0
    return depth == 0 and not quoted


def check(n: str, c: str, s: str) -> None:
    assert balanced(s)
    transfer = n[n.index("int64_t et_g3r_joint_hold_transfer_issued_v1("):
                 n.index("int64_t et_g3r_joint_hold_end_v1(")]
    assert transfer.index("state != hold->issued") < transfer.index(
        "et_g3r_joint_hold_issued_match_v1(hold, state)") < transfer.index(
            "hold->issued = NULL")
    assert "if (!hold || hold->issued) return joint_invalid();" in n
    compose = c[c.index("(define (c2-training-state-compose-internal request)"):
                c.index("(define (c2-training-state-borrow-activation-rollback!")]
    guard = compose.index("(guard (caught")
    assert guard < compose.index("(c2-training-state-claim-p1! p1 owner-token)")
    assert compose.index("(o2-find-identity o2 o2-states)", guard) < compose.index(
        "(c2-training-state-rollback-p1! p1 owner-token)", guard)
    assert "(eq? (vector-ref o2-record 6) owner-token)" in compose
    assert "(o2-state-claim-c2-rollback-internal! o2 owner-token)" in compose
    allocate = s.index("(words (make-bytevector 136 0))")
    handler = s.index("(guard (caught")
    compose_call = s.index("(c2-training-state-compose-internal request)")
    receipt = s.index("(c2-training-state-find (vector-ref request 5))")
    transfer_call = s.index("(g3r-joint-c2-transfer-issued joint issued)")
    end_call = s.index("(g3r-joint-o2-end joint)", transfer_call)
    assert allocate < s.index("(request (vector c2-training-state-compose-tag")
    assert s.index("(result (vector g3r-joint-c2-claim-tag") < handler < compose_call
    assert s.index("(g3r-joint-c2-copy-words joint words)") < compose_call
    assert compose_call < receipt < transfer_call < end_call
    assert "(eq? (vector-ref o2-record 6) 'live)" in s[handler:compose_call]
    assert "(g3r-joint-o2-release-issued joint)" in s[handler:compose_call]
    assert "(not (= 2 (g3r-joint-o2-lifecycle joint)))" in s
    assert "(vector-set! o2-ledger 8 'c2-owned)" in s


check(native, c2, claim)
mutations = (
    (native.replace("state != hold->issued", "state == hold->issued"), c2, claim),
    (native.replace("hold->issued = NULL;\n  return 0;", "return 0;"), c2, claim),
    (native, c2.replace("(o2-state-claim-c2-rollback-internal! o2 owner-token)", "#t"), claim),
    (native, c2, claim.replace("(g3r-joint-c2-transfer-issued joint issued)", "0")),
    (native, c2, claim.replace("(g3r-joint-o2-release-issued joint)", "0")),
    (native, c2, claim.replace("(g3r-joint-o2-lifecycle joint)", "2")),
)
for changed in mutations:
    try:
        check(*changed)
    except (AssertionError, ValueError):
        continue
    raise AssertionError("C2 claim mutation escaped source gate")

print("joint C2 claim source gate: pass (6 mutation controls)")
