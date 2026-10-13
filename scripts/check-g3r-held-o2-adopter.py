#!/usr/bin/env python3
"""Source gate for the feature-private held O2 enrollment prerequisite."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
bridge = (root / "native/o2_wave2_package_bridge.c").read_text()
adopter = (root / "native/g3r_held_o2_adopter_extension.esk").read_text()


def check(c: str, s: str) -> None:
    assert "issued == NULL || issued != candidate" in c
    assert "et_g3r_private_o2_save_owner_match_v1" in c
    assert "et_g3r_private_o2_save_issued_match_v1" in c
    assert "et_g3r_private_o2_save_state_v1" in c
    assert "et_o2_optimizer_state_lifecycle_v1" in c
    allocation = s.index("(let* ((shell (o2-make-identity")
    issue = s.index("(let ((issued (g3r-held-o2-issue hold)))")
    handler = s.index("(guard (caught", allocation)
    assert allocation < s.index("(enrollment (cons state o2-states))") < issue
    assert allocation < s.index("(pointer (vector #f))") < issue
    assert allocation < s.index("(ledger\n") < issue
    assert allocation < handler < issue
    assert s[handler:issue].count("(guard (") == 1
    assert issue < s.index("(vector-set! pointer 0 issued)")
    assert s.index("(vector-set! pointer 0 issued)") < s.index(
        "(vector-set! state 1 issued)"
    )
    assert s.index("(g3r-held-o2-adopter-match! ledger operation)", issue) < s.index(
        "(set! o2-states enrollment)"
    )
    assert "(not (eq? issued (vector-ref (vector-ref ledger 6) 1)))" in s
    for status in ("((= status 0)", "((= status -1)", "((= status -2)"):
        assert status in s
    assert "(else (g3r-held-o2-fail-stop))" in s
    assert "(raise caught)" in s


check(bridge, adopter)
# Mutation controls make the source checks sensitive to the two causal
# authority edges. Native runtime and C2 partial-claim cuts remain separate.
for changed_bridge, changed_adopter in (
    (bridge.replace("issued != candidate", "issued == candidate"), adopter),
    (bridge, adopter.replace("(vector-set! pointer 0 issued)", "#t")),
    (bridge, adopter.replace("(set! o2-states enrollment)", "#t")),
    (bridge, adopter.replace("(g3r-held-o2-adopter-match! ledger operation)", "#t")),
    (bridge, adopter.replace("(guard (caught\n                 (#t", "(NO-GUARD (caught\n                 (#t")),
):
    try:
        check(changed_bridge, changed_adopter)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("source mutation escaped the held-adopter gate")

print("held O2 adopter source gate: pass (5 mutation controls)")
