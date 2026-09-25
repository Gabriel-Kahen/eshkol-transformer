# G3-T source-private manual P1 prefill leaf

This conditional leaf admits an active kind-0 call, its exact pending kind-3
owned f32 `[1,256]` result, and a live kind-2 owned CPU i64 `[1,1]` byte input.
`frame_begin` creates an unpublished A2 candidate. `role_step` dispatches the
same 21 reviewed G3-N/N2/A2 P1 roles as generate prefill, in exact ordinal
order. `frame_prepare` copies the 256 final logits into the pending I2 and
captures the pinned 14-parameter binding. `call_prepare_end` preflights the
candidate transaction, old cache, result bits and binding. `frame_commit`
atomically replaces the cache/binding/last-logit state and publishes the
detached kind-3 result. `call_finish` drains pins and clears call linkage.
No RNG draw occurs. A later manual prefill may replace the old cache.

The Eshkol transport has only the accepted private `g3t-logits-reserve`,
`g3t-call-prepare-end`, and `g3t-call-finish` wrappers. It preallocates/root
the shell and registry node before native reserve, cleans up on a later
exception, and uses a no-allocation publication tail. The 21-role schedule is
test-local pending an accepted production G3-M orchestration signature. This
leaf adds no public facade, package export, CLI, manual decode, or N>1 claim.

The focused aggregate compares all 256 output logit bits and first-position
K/V bits to an independently executed M3T P2 reference, checks the A2
committed cache and unchanged RNG, verifies detached result ownership after
parent release, and exercises identity, ordinal, borrowed-result, borrowed-old-
cache, binding, candidate allocation, and abort cuts. The accepted generate
P1/G0, P1/G1, and P2/G0 tests remain in the same aggregate. Gate result and
source commit are recorded in the roadmap after the pinned run.
