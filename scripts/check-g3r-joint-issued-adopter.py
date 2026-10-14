#!/usr/bin/env python3
"""Source gate for exact joint-issued pointer enrollment and preclaim drain."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
c_source = (root / "native/g3r_joint_hold.c").read_text()
g3_source = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
scheme = (root / "native/g3r_joint_issued_o2_adopter_extension.esk").read_text()


def check(c: str, g3: str, s: str) -> None:
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
    assert drain.index("(g3r-joint-o2-optimizer-match\n") < drain.index(
        "(g3r-joint-o2-release-issued (vector-ref ledger 1))")
    assert drain.index("(g3r-joint-o2-release-issued (vector-ref ledger 1))") < drain.index(
        "(g3r-joint-o2-end (vector-ref ledger 1))")
    assert drain.index("(g3r-joint-o2-end (vector-ref ledger 1))") < drain.index(
        "(g3r-joint-o2-lifecycle (vector-ref ledger 1))") < drain.index(
            "(vector-set! ledger 8 'drained)")
    assert "(not (= 2 (g3r-joint-o2-lifecycle (vector-ref ledger 1))))" in drain
    assert "(g3r-joint-o2-record-error! ledger)" in drain
    assert "(vector-set! ledger 8 'drained)" in drain
    native_end = c[c.index("int64_t et_g3r_joint_hold_end_v1"):]
    assert "if (!hold || hold->issued) return joint_invalid();" in native_end
    assert "if (hold->o2_hold != NULL) abort()" in native_end
    assert "et_g3r_private_g3t_save_hold_lifecycle_v1(hold->g3_hold) != 2" in native_end
    assert native_end.index("et_g3r_private_o2_save_end_v1") < native_end.index(
        "et_g3r_private_g3t_save_end_v1")
    assert "et_g3r_joint_hold_lifecycle_v1" in native_end
    assert "if (hold->live == 1u) return 1" in native_end
    assert "if (hold->live == 0u) return 2" in native_end
    g3_lifecycle = g3[g3.index("int64_t et_g3r_private_g3t_save_hold_lifecycle_v1"):]
    assert "record == candidate" in g3_lifecycle
    assert "record->kind != G3T_SAVE_HOLD" in g3_lifecycle
    assert "record->state == G3T_DEAD" in g3_lifecycle


check(c_source, g3_source, scheme)
for changed_c, changed_g3, changed_s in (
    (c_source.replace("optimizer != hold->optimizer", "optimizer == hold->optimizer"), g3_source, scheme),
    (c_source.replace("state != hold->issued", "state == hold->issued"), g3_source, scheme),
    (c_source, g3_source, scheme.replace("(vector-set! pointer 0 issued)", "#t")),
    (c_source, g3_source, scheme.replace("(guard (caught\n                 (#t", "(NO-GUARD (caught\n                 (#t")),
    (c_source, g3_source, scheme.replace("(g3r-joint-o2-end (vector-ref ledger 1))", "#t")),
    (c_source, g3_source, scheme.replace("(g3r-joint-o2-lifecycle (vector-ref ledger 1))", "#t")),
    (c_source, g3_source, scheme.replace("(g3r-joint-o2-optimizer-match\n                          (vector-ref ledger 1) (vector-ref ledger 3))", "0")),
    (c_source.replace("et_g3r_private_g3t_save_hold_lifecycle_v1(hold->g3_hold)", "0"), g3_source, scheme),
    (c_source, g3_source.replace("record->state == G3T_DEAD", "record->state == G3T_LIVE"), scheme),
):
    try:
        check(changed_c, changed_g3, changed_s)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("joint-issued adopter mutation escaped source gate")

print("joint-issued O2 adopter source gate: pass (9 mutation controls)")
