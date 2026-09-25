# G3-T source-private P1/G0 zero-budget leaf

The G0 path reserves an empty I1 ID owner and empty Eshkol staging/raw
bytevectors before authentic one-byte P1 numerical prefill. It runs all 21
G3-T roles, prepares zero IDs and unchanged RNG, validates T1 raw decoding,
then preflights the pinned binding, empty I1 owner, staged A2 view and old
cache lease. The no-failure tail commits the prompt-only A2 cache, binding,
last logits and live empty output. It invokes no token frame or sampler and
consumes no draw. Output release retains the authenticated tombstone.

This leaf remains source-private, fixed to diagnostic C2/P1. Public facades,
P2 input, accessors and a general generation loop remain downstream. The
private witness compares all 256 P1 logits bitwise with an independent M3T
two-token forward first row. Allocation, borrow, ordering and rollback cuts
run in normal, repeated and ASan/UBSan/LSan pinned f31/LLVM21 execution.
No multi-call retention or performance horizon is measured in this leaf;
authenticated tombstones accumulate by design.
