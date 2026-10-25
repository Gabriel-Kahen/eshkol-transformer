# G3-R private C2/SHA receipt to TR3 restore seam

**Source-staged only.** This leaf composes the accepted fresh-process reader
root and the accepted TR3 joint restore under one CLI3/T1/C2/TR3 registry. It
does not establish a restored model or continued generation.

`(g3r-c2-receipt-tr3-restore-internal! receipt trainer) -> #t` accepts only
an exact live C2/SHA receipt identity from the owning registry and an exact
enrolled, idle, plan-free TR3 trainer. It checks that the receipt's retained
C2 owner is still live, then repeats the TR3 receiver idle admission before
changing receipt phase. Forged, dead, borrowed or restoring receipts and a
foreign, stale or busy trainer reject before the phase change. The source
uses no path, caller digest, public C2 LOAD, or second C2 registry.

The seam marks the receipt `restoring`, passes its exact retained C2 owner
directly to `tr3-c-joint-restore-internal!`, and keeps the receipt busy until
TR3 returns. TR3 owns its source borrow, deep staging, receiver transaction
and rollback. On a recoverable exception, the seam restores the receipt to
`live` and rethrows the original value; on success it restores `live` and
returns only `#t`. An unexpected non-`#t` return from the committed TR3 tail
fail-stops. The seam never returns or releases the owner. The caller's
candidate ledger retains receipt and trainer ownership and must release them
in dependency order after success or failure.

This API authenticates an enrolled idle receiver; it cannot prove when that
receiver was constructed. The next aggregate must create a fresh same-registry
model, optimizer, dataset and trainer under its candidate ledger, check the
pair/C2/T1/X1/profile controls, call this seam once, and verify restored
parameter bits before publication. A current caller trainer is not a valid
candidate. Source mutation tests cover the identity, phase, owner and trainer
checks. Compiled normal/repeat/native-sanitizer behavior and forged/dead/busy
receipt or wrong-trainer runtime negatives remain unrun.
