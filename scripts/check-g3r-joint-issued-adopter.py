#!/usr/bin/env python3
"""Source gate for exact joint-issued pointer enrollment and preclaim drain."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
c_source = (root / "native/g3r_joint_hold.c").read_text()
scheme = (root / "native/g3r_joint_issued_o2_adopter_extension.esk").read_text()


def check(c: str, s: str) -> None:
    owner = c[c.index("int64_t et_g3r_joint_hold_optimizer_match_v1"):c.index(
        "et_o2_optimizer_state *et_g3r_joint_hold_issue_v1")]
    assert "optimizer != hold->optimizer" in owner
    assert "et_g3r_joint_hold_check_v1(hold)" in owner
    issued = c[c.index("int64_t et_g3r_joint_hold_issued_match_v1"):c.index(
        "int64_t et_g3r_joint_hold_copy_words_v1")]
    assert "state != hold->issued" in issued
    assert "et_g3r_private_o2_save_issued_match_v1" in issued
    assert ":real et_g3r_joint_hold_issue_v1" in s
    assert ":real et_g3r_joint_hold_optimizer_match_v1" in s
    assert ":real et_g3r_joint_hold_issued_match_v1" in s
    adopt = s[s.index("(define (g3r-joint-o2-adopt!"):s.index(
        ";;; Preclaim drain")]
    allocation = adopt.index("(let* ((shell (o2-make-identity")
    issue = adopt.index("(let ((issued (g3r-joint-o2-issue joint)))")
    guard = adopt.index("(guard (caught", allocation)
    assert allocation < adopt.index("(enrollment (cons state o2-states))") < guard < issue
    assert allocation < adopt.index("(pointer (vector #f))") < guard
    assert allocation < adopt.index("(failure (vector 0 0 0))") < guard
    assert adopt[guard:issue].count("(guard (") == 1
    assert issue < adopt.index("(vector-set! pointer 0 issued)") < adopt.index(
        "(vector-set! state 1 issued)") < adopt.index(
            "(g3r-joint-o2-issued-match joint issued)") < adopt.index(
                "(set! o2-states enrollment)")
    drain = s[s.index("(define (g3r-joint-o2-drain-preclaim!"):]
    assert drain.index("(g3r-joint-o2-release-issued (vector-ref ledger 1))") < drain.index(
        "(g3r-joint-o2-end (vector-ref ledger 1))")
    assert "(g3r-joint-o2-record-error! ledger)" in drain
    assert "(vector-set! ledger 8 'drained)" in drain
    native_end = c[c.index("int64_t et_g3r_joint_hold_end_v1"):]
    assert "if (!hold || hold->issued) return joint_invalid();" in native_end
    assert native_end.index("et_g3r_private_o2_save_end_v1") < native_end.index(
        "et_g3r_private_g3t_save_end_v1")


check(c_source, scheme)
for changed_c, changed_s in (
    (c_source.replace("optimizer != hold->optimizer", "optimizer == hold->optimizer"), scheme),
    (c_source.replace("state != hold->issued", "state == hold->issued"), scheme),
    (c_source, scheme.replace("(vector-set! pointer 0 issued)", "#t")),
    (c_source, scheme.replace("(guard (caught\n                 (#t", "(NO-GUARD (caught\n                 (#t")),
    (c_source, scheme.replace("(g3r-joint-o2-end (vector-ref ledger 1))", "#t")),
):
    try:
        check(changed_c, changed_s)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("joint-issued adopter mutation escaped source gate")

print("joint-issued O2 adopter source gate: pass (5 mutation controls)")
