"""Causal source gate for the private data-only held C2 transaction."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "native/g3r_c2_held_transaction_extension.esk").read_text()
LOADER = (ROOT / "native/g3r_c2_held_transaction_root.esk").read_text()
P1 = (ROOT / "internal/p1/lib/transformer/module.esk").read_text()
O2 = (ROOT / "native/g3r_held_o2_adopter_extension.esk").read_text()
C2 = (ROOT / "native/c2_training_state_extension.esk").read_text()


def between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def check(source: str, loader: str) -> None:
    assert loader.index("g3r_original_owner_coordinator_root.esk") < loader.index(
        "g3r_held_o2_adopter_extension.esk") < loader.index(
        "g3r_c2_held_transaction_extension.esk")
    assert "(provide" not in source + loader
    assert not any(word in source + loader for word in (
        "c2-save-image", "save-receipt", "file-open", "file-write",
        "checkpoint-save", "g3r-save-pair"))

    ledger = between(source, "(define (g3r-c2-held-transaction-ledger)",
                     "(define (g3r-c2-held-original-check!")
    assert ledger.count("(vector #f)") == 3
    assert "(vector c2-training-state-compose-tag #f #f #f" in ledger
    assert "'g3r-c2-held-transaction-internal #f)" in ledger

    original = between(source, "(define (g3r-c2-held-original-check!",
                       "(define (g3r-c2-held-controls-check!")
    assert "(vector-ref record 13)" in original
    assert "(vector-ref (vector-ref record 14) 0)" in original
    assert "(g3r-original-owner-g3-recheck!" in original
    assert "(tr3-lease-snapshot-recheck-held-o2-internal record parent hold)" in original
    controls = between(source, "(define (g3r-c2-held-controls-check!",
                       "(define (g3r-c2-held-release-o2-or-stop!")
    assert "(equal? current captured)" in controls

    release = between(source, "(define (g3r-c2-held-release-o2-or-stop!",
                      "(define (g3r-c2-held-abort!")
    assert "(g3r-held-o2-adopter-release! adopter)" in release
    assert "(g3r-held-o2-adopter-match!" in release
    assert release.count("(g3r-held-o2-adopter-release! adopter)") == 2
    assert "(eq? (vector-ref adopter 8) 'dead) first" in release
    assert "(eq? (vector-ref adopter 8) 'enrolled)" in release
    assert "(g3r-held-o2-fail-stop)" in release

    abort = between(source, "(define (g3r-c2-held-abort!",
                    "(define (g3r-c2-held-transaction-internal")
    assert "(vector-ref request 5)" in abort
    assert "(not (eq? (vector-ref request 1) p1))" in abort
    assert "(not (eq? (vector-ref request 2) o2))" in abort
    assert "(eq? (vector-ref (vector-ref adopter 6) 6)" in abort
    assert abort.index("(g3r-c2-held-release-o2-or-stop! adopter)") < abort.index(
        "(state-dict-release! p1)") < abort.index("(g3r-original-owner-end! owner)")
    assert "(if (not (state-dict-release! p1))" in abort
    assert "(if (and release-error (not first))" in abort
    assert "(guard (cleanup-error (#t (g3r-held-o2-fail-stop)))" in abort

    transaction = source.split("(define (g3r-c2-held-transaction-internal", 1)[1]
    ordered = (
        "(ledger (g3r-c2-held-transaction-ledger))",
        "(guard (caught",
        "(g3r-original-owner-begin! trainer generator)",
        "(vector-set! (vector-ref ledger 1) 0 owner)",
        "(tr3-c-snapshot-controls! record (vector-ref g3-ledger 2))",
        "(module-state-dict-held-clone-internal",
        "(vector-ref record 8)",
        "(vector-set! (vector-ref ledger 2) 0 p1)",
        "(g3r-held-o2-adopt!",
        "(vector-ref (vector-ref owner 4) 0)",
        "(vector-ref record 13)",
        "(vector-set! (vector-ref ledger 3) 0 adopter)",
        "(g3r-c2-held-controls-check! owner (vector-ref request 3))",
        "(g3r-c2-held-compose-call request)",
        "(g3r-c2-held-claim-call adopter request p1)",
        "(g3r-original-owner-end! owner)",
        "(vector-set! ledger 5 'dead)",
        "shell)))))",
    )
    positions = [transaction.index(item) for item in ordered]
    assert positions == sorted(positions)
    assert "(raise (g3r-c2-held-abort! ledger caught))" in transaction
    assert "(eq? (vector-ref request 1) #f)" in transaction
    assert "(eq? (vector-ref request 2) #f)" in transaction
    assert "(vector-ref request 5)" in transaction
    assert "(g3r-held-o2-fail-stop)" in transaction
    assert transaction.count("(g3r-c2-held-controls-check! owner (vector-ref request 3))") == 2


check(SOURCE, LOADER)
mutants = (
    (SOURCE, LOADER.replace("(load \"g3r_held_o2_adopter_extension.esk\")", "")),
    (SOURCE.replace("(vector-ref (vector-ref record 14) 0)", "#f", 1), LOADER),
    (SOURCE.replace("(equal? current captured)", "#t", 1), LOADER),
    (SOURCE.replace("(eq? (vector-ref adopter 8) 'dead) first", "#t first", 1), LOADER),
    (SOURCE.replace("(g3r-held-o2-adopter-match!\n               adopter", "#t ; adopter", 1), LOADER),
    (SOURCE.replace("(vector-ref request 5)\n            (not", "#f\n            (not", 1), LOADER),
    (SOURCE.replace("(g3r-c2-held-release-o2-or-stop! adopter)", "#t", 1), LOADER),
    (SOURCE.replace("(state-dict-release! p1)", "#t", 1), LOADER),
    (SOURCE.replace("(g3r-original-owner-end! owner)", "#t", 1), LOADER),
    (SOURCE.replace("(module-state-dict-held-clone-internal\n                    (vector-ref record 8))",
                    "(module-state-dict-held-clone-internal generator)", 1), LOADER),
    (SOURCE.replace("(g3r-c2-held-claim-call adopter request p1)", "#f", 1), LOADER),
    (SOURCE.replace("(g3r-c2-held-compose-call request)",
                    "(c2-save-image-receipt request)", 1), LOADER),
)
for mutated_source, mutated_loader in mutants:
    assert (mutated_source, mutated_loader) != (SOURCE, LOADER)
    try:
        check(mutated_source, mutated_loader)
    except (AssertionError, ValueError, IndexError):
        continue
    raise AssertionError("held transaction mutation escaped source gate")

assert "(define (module-state-dict-held-clone-internal module)" in P1
assert "(published-state-shell-for-raw operation state)" in P1
assert "(g3r-held-o2-issued-match" in O2
assert "(not (eq? (vector-ref owner 1) p1))" in O2
assert "(c2-training-state-rollback-o2-or-live!" in C2
assert "(c2-training-state-rollback-p1!" in C2
print("G3R held C2 transaction source PASS (12 causal mutations)")
