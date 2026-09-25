# G3-C4 private manual role step

This bounded continuation starts at integrated commit `45a05dc`. It adds one
source-private native `role_step(ctx, ordinal)` dispatcher for the accepted
manual prefill T1/T2 and decode transcript. It adds no Eshkol public root,
generation loop, provider route, or new numerical kernel.

The guarded entry authenticates the active generator context before examining
the ordinal, rejects selectors outside 0..20, and requires a running frame
whose `next_ordinal` equals the request. It invokes exactly one of the seven
accepted helper groups: role0, pre-A2 1..9, A2 10, attention merge 11,
attention-out 12, residual 13, and tail 14..20. Each helper already checks its
own pending result/cache/pins and increments `next_ordinal` only after a
successful provider result. The dispatcher neither writes scratch nor advances
the cursor. A failed provider call can be retried at the same ordinal.

`ET_G3C4_MANUAL_ROLE_STEP_PRIVATE` requires the completed manual publication
feature. The sole additional defined symbol is
`et_g3c4_private_role_step_v1`; it remains outside the normal archive.
Tests cover T1, T2 and decode, every ordinal, repeated/out-of-order/invalid
requests, seven injected tail-provider failures and retries per route, an A2
allocation failure/retry, reference final logits, and end-to-end publication.
The runner seals normal, repeat and sanitizer outputs and the exact source
closure under the pinned LLVM21 image.
