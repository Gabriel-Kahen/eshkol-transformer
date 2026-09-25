# TR3 private fixed-profile no-EOS step composer

The test-only `native/tr3_step_composer_root.esk` composes the accepted lease,
TR3-B objective, pre-write transaction, and no-EOS O2 commit tail in one Eshkol
registry universe. `tr3-step-composer-internal! trainer` is source-private and
requires the fixed context length two and accumulation count two. It owns a
rooted ledger for the authentic step frame, one M3 input and seed, and the
current D2 batch, M3 graph output, and graph-logits lease. It returns `#t`
after one update; it publishes no public metrics or trainer facade.

Each D2 batch advances exactly one cursor ordinal. The composer stages its
input, runs the genuine M3 forward and TR3-B analytic VJP, reads the objective
observation's bool-mask weight (one or two), then releases graph logits,
output, and batch in that order. After two microbatches it releases input and
seed, prepares the accepted O2 plan with the exact cumulative binary32 weight,
and invokes the accepted private commit tail with the integer active-token
count. The latter validates cursor span, weight bits, and control bounds
before O2's first parameter write.

Every caught accumulation or prepare failure passes through the ledger:
release any exact live graph logits/output/batch/input/seed, abort the
authenticated step frame (including a prepared O2 plan), restore the D2
cursor/RNG/counters, clear transient gradients, and rethrow the original
categorized error. Exact release/abort failure is treated as an invariant
defect and fail-stops rather than returning a recoverable partial step.
After O2 enters `COMMITTING`, its accepted native contract either completes
or fail-stops; the TR3 tail then uses only staged primitive writes. The frame
and test hook remain private, never exposed in a public surface.

The pinned compiled witness injects early failure before D2 acquisition,
mid failure after VJP with graph and batch live, and precommit failure after
O2 prepare. Each cut checks owner retirement, absent gradients, unchanged
parameter/O2 state/cursor/RNG/counters, and a same-trainer retry. Success
checks one real two-microbatch update against independent PyTorch numerical
evidence. A subsequent EOS is an explicit `invalid-state` and preserves the
previously committed step. General accumulation counts, EOS replay/epoch
updates, automatic training loops, metrics, public APIs, C ownership, and
resume-equivalence proof remain separate dependencies.
