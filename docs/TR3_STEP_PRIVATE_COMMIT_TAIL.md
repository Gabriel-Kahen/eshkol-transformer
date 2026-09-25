# TR3 private fixed-profile commit tail

The test-only `native/tr3_step_commit_tail_root.esk` loads the accepted lease,
objective, O2, and pre-write transaction roots in one registry universe. Its
source-private `tr3-step-commit-internal! frame active-tokens` is restricted to
a step that consumes exactly the configured accumulation count without an EOS
rewind. The frame and O2 plan must remain solely with the private composer;
their mutable Eshkol vectors/pointers are not a public authority.

Before the first parameter write, the operation authenticates the exact
prepared frame/lease/O2 plan, fixed train modes, idle D2/M3 receivers, unchanged
RNG and old counters, and the decoded D2 cursor span. It requires the supplied
positive active-token count to be within the fixed bool-mask range and at most
2^24, then encodes that integer exactly as binary32 and compares it to the
weight bits already validated by O2 prepare against all 14 gradients. It
checks i64 token/update overflow and stages the next token, update, and epoch
values and frame-registry successor in a frame-rooted ledger. A precommit
rejection leaves the prepared frame and parameters unchanged for explicit
abort or corrected retry.

The accepted O2 native commit preflights the exact registered plan before its
`COMMITTING` transition. Thereafter it either completes its nonallocating
parameter/moment/gradient/count tail or `_Exit(134)` on an impossible provider
defect; it never returns a recoverable partial update. After native success,
the TR3 tail performs only staged primitive writes to the three durable
counters, lease phase, frame status, and root list. The live D2 cursor and
fixed-profile RNG are already at their committed values; no post-write seek,
allocation, owner release, error construction, or provider call occurs.

The compiled witness tests malformed token count, forged frame, wrong cursor
span, prior accumulation/prepare failures, same-plan correction, one genuine
two-microbatch AdamW commit, exact counters/cursor/RNG, gradient clear,
retirement, stale-plan rejection, and the independent PyTorch key-weight
comparison. This is a private **no-EOS commit boundary**, not public
`trainer-step!`: it does not own the D2/M3 microbatch loop, automatically
abort every accumulation failure, generate public metrics, handle EOS/epoch
rewind, prove all-parameter trajectory, or establish resume equivalence.
