# Data-only held C2 transaction source

Status: feature-private source candidate. The compiled joint owner and failure
gate has not run. This source does not create a receipt, SAVE pair, file, replay,
or public trainer API.

`native/g3r_c2_held_transaction_root.esk` loads the original-owner coordinator,
the held O2 adopter, and this wrapper into one registry universe. The private
`g3r-c2-held-transaction-internal(trainer, generator)` returns a detached C2
shell only after the original O2 and G3 holds and trainer lease drain. It
preallocates one six-slot C2 request and rooted cells for the original-owner
ledger, P1 clone, and O2 adopter before the first hold begins. It captures
validated trainer controls from the leased original record, clones P1 from that
record's exact original model, and issues O2 through the exact held optimizer.
The C2 request is `[tag, P1, O2, controls, operation, #f]`.

The wrapper checks the original G3/model/optimizer binding and controls before
composition. C2's accepted rollback reconciles an O2 claim that mutates and
then raises, before its error escapes. On a preclaim error, the wrapper accepts
only an unchanged request with both children live. It releases the exact O2
issued pointer first, then the P1 clone, then drains the original holds. A
proved-live O2 release failure gets one retry on the same rooted ledger; a
proved-dead provider defect keeps the first typed error. P1 release or hold
drain uncertainty fails stop with the ledger rooted.

On a successful C2 compose, the existing adopter claim checker verifies the
exact O2 pointer, optimizer, C2 owner token, child shells, and P1 identity. The
wrapper also checks that the checker returned the exact shell and rechecks the
original controls before draining holds. Any exception, false success, or
ambiguous request after transfer fails stop. No owner shell escapes while
original holds remain live.

The source gate `python3 tests/g3r/test_c2_held_transaction_source.py` includes
12 causal mutations across root composition, original bindings, controls,
cleanup order, exact O2 retry, P1 origin, claim check, and publication boundary.
The existing P1 held-clone overlay covers clone and wrap failures and original
path/finalized identity; the existing C2 owner runtime source covers an O2
claim that mutates before raising. These are separate prerequisite results.
The joint Eshkol runtime, failure-injection matrix, original byte comparison,
and compiled atomic-shell gate remain required before admission.
