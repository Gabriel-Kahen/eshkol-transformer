# TR3 private fixed-profile finite-D2 step composer

The test-only `native/tr3_step_composer_root.esk` composes the accepted lease,
TR3-B objective, pre-write transaction, and O2 commit tail in one Eshkol
registry universe. `tr3-step-composer-internal! trainer` is source-private and
requires the fixed context length two and a positive configured accumulation
count with an exactly representable cumulative binary32 mask weight. It owns a
rooted ledger for the authentic step frame, one M3 input and seed, and the
current D2 batch, M3 graph output, and graph-logits lease. It returns `#t`
after one update; it publishes no public metrics or trainer facade.

An EOS sentinel is not a microbatch. At finite EOS, the composer validates
the trainer-owned epoch-start cursor, seeks it, stages one completed epoch,
and requests a real batch again. The accepted constructor rejects an empty
finite dataset before lease enrollment. One update may cross more than one
epoch boundary when the finite dataset has fewer rows than accumulation
steps. The rewind guard permits at most `A` boundaries in one update. Each
real D2 batch advances exactly one cursor ordinal. The composer stages its
input, runs the genuine M3 forward and TR3-B analytic VJP, reads the objective
observation's bool-mask weight (one or two), then releases graph logits,
output, and batch in that order. After `A` microbatches it releases input and
seed, prepares the accepted O2 plan with the exact cumulative binary32 weight,
and invokes the accepted private commit tail with the integer active-token
count and staged epoch delta. The latter validates the accepted global
`U*A = E*(M-S)+(C-S)` equation for both step-start and successor controls,
plus cursor identity, exact weight bits, and control bounds before O2's first
parameter write. Here `S` is trainer-owned epoch start, `M` is finite total
rows, `C` is current ordinal, `U` is completed updates, `A` is accumulation
steps, and `E` is completed epochs.

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
evidence. It additionally proves EOS before the first microbatch and between
microbatches, injected failure after rewind with exact same-trainer rollback,
two epoch boundaries within one update, and empty-D2 constructor rejection.
The two-update key-weight trajectory is checked against an independent
PyTorch reference with gradients recomputed at each updated parameter set.
The checkpoint trajectory witness also exercises `A=1` and `A=3` over three
resumed updates with unequal active-token weights. Counts whose minimum mask
weight exceeds `2^24`, or whose observed weight exceeds that exact integer
range, reject before the first parameter write. Streaming datasets, automatic
training loops, metrics, public APIs, and C ownership remain separate
dependencies.
