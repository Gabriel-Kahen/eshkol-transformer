# MOD5/O2 CPU-f32 component continuation

Run `make test-mod5-o2-continuation` on x86-64 with Clang. The opt-in gate
builds the MOD5 provider, runs the native composition at `-O2`, and repeats
it with AddressSanitizer, UndefinedBehaviorSanitizer, and LeakSanitizer.

The gate reuses the MOD5 training smoke's width-eight pre-norm block. It runs
RMSNorm, SwiGLU, and causal variant attention with ALiBi and a fixed window
through the existing MOD5 provider. Both four-query-head profiles use
head width two. MQA supplies one key/value head and GQA supplies two.
Each of the six updates uses a deterministic input batch and a fresh
backward pass through all nine parameter tensors. I2 admits the complete
gradient batch as one transaction and O2 performs binary32 AdamW with a
linear warmup and decay schedule.

At update two the test copies each I2 weight and both O2 moments as exact
binary32 words and reads the optimizer options and completed-update count.
It releases the source moment snapshot, creates new I2 parameter owners
from the copied weights, and loads a new O2 owner via the C2 optimizer
component's native reconstruction and load seams. The recipient starts
with zero learning rates and a constant schedule. The gate compares all
weight words, both moment tensors per parameter, scalar options, successful
update count, and the next schedule factor against the uninterrupted owner
at the restore boundary and after each remaining update.

The negative cases exercise a same-element-count rank mismatch at the last
parameter, a nonfinite final second moment, and a pre-transfer reconstruction
failure after all moments are prepared. The mismatched recipient retains
its exact prior state. The reconstruction failures publish no owner and
restore the live snapshot, moment, and builder counts.

Each run keeps six live optimizer and parameter fixtures until process exit.
O2's generic live optimizer contract is process-scoped. The private G3-R
retirement seam requires a fixed 14-entry candidate and does not admit
these nine-entry fixtures. The gate explicitly releases detached state
snapshots, reconstruction builders, temporary tensors, and gradient plans.

The pinned Ubuntu 22.04 x86-64 environment with Clang 21.1.8 passes both
variants in optimized and sanitized builds. This is an in-process component
test using the fixture's fixed shape, parameter order, and optimizer
configuration. The accepted [MOD5-C witness](MOD5_E2E_C1_IO5.md) separately
covers model file interchange for the width-16 decoder. Its rank-two
parameter shapes exceed the current public C2 LOAD capability list
documented in [C2](C2_TRAINING_STATE.md). MOD5 C2 file publication, new-process
resume, decoder tie reconstruction, and joint model/optimizer/dataset/RNG
restore remain open.
