#!/usr/bin/env python3
"""Static dependency and mutation checks for the pending-only terminal leaf."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3c4_p2g2_protected_scope_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3c4_p2g2_terminal_pending_anchor_extension.esk",
    "tests/g3c4/p2_g2_terminal_pending_anchor_test.esk",
    "tests/g3c4/p2_g2_terminal_pending_anchor_feature_off_test.esk",
    "scripts/check-g3c4-p2g2-terminal-pending-anchor.py",
    "scripts/test-g3c4-p2g2-terminal-pending-anchor.sh",
    "docs/g3/G3_C4_P2_G2_TERMINAL_PENDING_ANCHOR_LEAF.md",
    "native/g3c4_p2g2_terminal_pending_anchor_source_closure.txt",
]
closure = (root / additions[-1]).read_text().splitlines()
assert closure == base + additions
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure)

source = (root / additions[0]).read_text()
fixture = (root / additions[1]).read_text()
feature_off = (root / additions[2]).read_text()
runner = (root / additions[4]).read_text()

def source_contract(candidate):
    required = (
        "(define g3c4-p2g2-terminal-pending-records (vector '()))",
        "(define g3c4-p2g2-terminal-pending-anchors (vector '()))",
        "(define (g3c4-p2g2-terminal-pending-check call operation)",
        "(eq? (vector-ref pending i)",
        "(vector-ref anchor i)",
        "(eq? (vector-ref anchor 3) (vector-ref record 6))",
        "(bytevector? (vector-ref anchor 11))",
        "(= (bytevector-length (vector-ref anchor 14)) 2)",
        "(vector 10 24 9 16 9 16 9 10)",
        "(g3c4-p2g2-protected-reserve call)",
        "(g3c4-p2g2-protected-abort! call)",
        "(define-syntax g3c4-with-p2g2-terminal-pending-probe",
    )
    return all(phrase in candidate for phrase in required) and not any(
        forbidden in candidate for forbidden in (
            "et_g3c4_private_p2g2_terminal_commit_v1",
            "et_g3c4_private_p2g2_terminal_ids_clone_v1",
            "g3c4-native-call-finish",
        ))

assert source_contract(source)
for original, replacement in (
    ("(eq? (vector-ref pending i)", "(eq? (vector-ref pending 0)"),
    ("(eq? (vector-ref anchor 3) (vector-ref record 6))",
     "(eq? (vector-ref anchor 3) #f)"),
    ("(= (bytevector-length (vector-ref anchor 14)) 2)",
     "(= (bytevector-length (vector-ref anchor 14)) 1)"),
    ("(vector 10 24 9 16 9 16 9 10)",
     "(vector 10 24 9 16 9 16 9 9)"),
    ("(g3c4-p2g2-protected-abort! call)", "#t"),
):
    assert original in source
    assert not source_contract(source.replace(original, replacement, 1))

assert source.index("(g3c4-p2g2-protected-reserve call)") < source.index(
    "(cons pending (vector-ref g3c4-p2g2-terminal-pending-records 0))")
abort = source[source.index("(define (g3c4-p2g2-terminal-pending-abort! call)"):]
assert abort.index("(g3c4-p2g2-terminal-pending-check call operation)") < abort.index(
    "(g3c4-p2g2-protected-abort! call)") < abort.index(
    "(g3c4-p2g2-zero-bytes! (vector-ref anchor 11) 1)")
for phrase in (
    "equal-length second raw replacement rejected",
    "equal-length logical text replacement rejected",
    "changed allocated extent rejected",
    "output shell drift rejected",
    "held lease keeps terminal anchor pending",
    "exception wrapper aborts and retains no pending anchor",
):
    assert phrase in fixture, phrase
assert "g3c4_p2g2_ownership_extension.esk" not in feature_off
assert "pending terminal wrapper rejects feature-off entry" in feature_off
assert "p2_g2_terminal_pending_anchor_test.esk" in runner
assert "p2_g2_terminal_pending_anchor_feature_off_test.esk" in runner
assert "compile_mode normal" in runner and "compile_mode sanitize" in runner
assert 'cmp "$evidence/normal-run.stdout"' in runner
print("G3-C4 P2/G2 terminal pending anchor static PASS: mutants=5")
