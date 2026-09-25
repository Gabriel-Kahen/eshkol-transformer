# G3-C4 private manual frame publication

This bounded continuation starts at integrated commit `2fc1b03`. It implements
manual prefill T1/T2 and decode frame preparation/publication after the accepted
ordinal-20 transcript. No Eshkol entry, public role step, or generation loop is
added.

`frame_prepare(ctx, logits)` requires the exact pending result and completed
ordinal-21 frame. It checks the live 14 pins, idle committed cache, complete
staged A2 transaction and expected effective length, current binding for decode,
and writable result before copying the last 256 logits. The prepared state
advances only after the copy succeeds. Failed preparation leaves cache, binding,
RNG and publication state unchanged; a caller may retry or abort.

`call_prepare_end` repeats the publication preflight and marks the manual frame
end-ready. `frame_commit` repeats it once more before any publication, then
commits the A2 transaction, destroys the preflighted old cache, installs the
candidate and prefill binding, and detaches the published logits. Any impossible
native failure after the first commit mutation fails closed via `abort()`.
The trusted caller invokes `call_finish` immediately and exactly once; that
postcommit path only frees the frame and drains pins/active state. `call_abort`
rejects a committed frame. RNG stays unchanged for manual calls.

The feature is guarded by `ET_G3C4_MANUAL_FRAME_COMMIT_PRIVATE`, which requires
the completed manual tail. The published logits use a distinct live transport
state without changing the accepted 48-byte owner layout. Tests cover T1/T2,
decode, copy/readiness, allocation and borrow failures, nested A2 view rejection,
retry, atomic cache/binding/RNG observations, release and finish ownership.
