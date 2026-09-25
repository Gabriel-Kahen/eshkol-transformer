# G3-T private G1 final publication leaf

This leaf completes the pending P1/G1 token frame. Final
`frame_prepare` requires numeric, ID and raw-text readiness. A separate
`call_prepare_end` checks the live 14-pin binding, exact output I1 byte,
owned result lengths/RNG, and the staged A2 view at cache length two. It
ends both leases before returning. The final `frame_commit` repeats that
fallible preflight, then commits the A2 transaction and publishes the
successor RNG and independently live output in one no-failure tail.
An impossible A2 commit rejection after this preflight is fail-stop.
`call_finish` immediately drains pins and active call tokens.

The source-private `g3t-final-commit!` validates rooted call/output
linkage, prepares the frame, preflights cleanup, commits, finishes, then
seals the output entry and drops its parent/staging roots. Exact live
output release preflights an active I1 borrow, destroys the ID owner, and
keeps an authenticated tombstone. A generated output survives its
generator; its raw bytevector remains detached.

Recoverable failures before the commit preserve the prompt-only cache,
old RNG, and pending output. Abort scrubs the speculative append and
pending result. G0, a public facade, generation loop, CLI, and public
accessors remain open.
