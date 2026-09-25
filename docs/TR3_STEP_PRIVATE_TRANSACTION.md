# TR3 private pre-write step transaction

`native/tr3_step_transaction_root.esk` adds one source-private transaction
extension to the accepted single-update root. It is test-only and exports no
public trainer, C producer, checkpoint, or commit-tail ABI.

`tr3-step-start-internal trainer` requires the exact enrolled idle shell,
the fixed P1 train-mode set, no live M3/D2 work, the expected O2 update count,
and absent gradients. It captures a new D2 cursor bytevector, a copy of the
five RNG fields, all 17 fixed module modes, and the three trainer counters.
It roots a self-authenticating frame before entering the lease's `step` phase.
The five constructor receivers remain caller-owned. A second begin and a
forged/copied frame reject without changing the rooted transaction.

`tr3-step-prepare-internal! frame contributions weight-bits` requires the
configured X1 accumulation count and no live M3/D2 work. The accepted O2
prepare validates the exact per-parameter contribution count, total binary32
weight, gradients, and scratch before any parameter write. On success the
opaque native plan is rooted in the frame and bound to the lease's `commit`
phase. A rejected expectation leaves the frame active in `step` for rollback.
This leaf deliberately does not invoke the O2 commit entry.

`tr3-step-abort-internal! frame` requires all caller-owned batches, outputs,
and logits to be released. It aborts an O2 plan if prepared, seeks the
captured D2 cursor, clears all O2 gradients, restores the copied RNG and
counters, returns the lease to idle, and unroots the frame. A live D2/M3
owner rejects before cleanup; release and retry on the same frame is valid.
The fixed model must remain in train mode throughout the step. Mode changes
are outside the accepted accumulation operations and reject as invalid state;
the captured mode vector is an invariant witness rather than an eval-to-train
repair mechanism. A component cleanup fault can leave the frame rooted for
retry; this leaf does not claim fault-free automatic cleanup under arbitrary
provider failures.

The pinned f31/LLVM21 runtime witness uses genuine D2 masks and one analytic
TR3-B numerator VJP, then exercises a live-batch abort rejection, cursor/RNG/
counter/gradient rollback, forged/repeat/busy negatives, wrong X1/O2 prepare
expectations, an exact prepared-plan abort, and same-authority retry. It checks
the key parameter bytes and O2 count remain unchanged. It does not perform a
parameter update, epoch replay, metrics publication, post-write no-fail
trainer-control tail, public `trainer-step!`, or resume-equivalence proof.
