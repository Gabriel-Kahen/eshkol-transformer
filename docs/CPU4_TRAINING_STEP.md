# CPU4 full-step READ/WRITE handoff

This source-private leaf uses the genuine CPU4 trainer/control producer and its
canonical C4/P1/I2 binding. It owns an exact-frame registry and temporarily sets
the trainer state to `step` with the exact frame as parent. Existing control
APIs continue to reject busy trainers. It does not use the TR3 trainer ledger or
an M3T model owner, and adds no public/native ABI or caller callback.

Load `cpu4_training_step_extension.esk` after the control producer, accepted
`cpu4_training_schedule_extension.esk`, and `tr3_o2_step_clear_extension.esk`.
The last dependency reuses the reviewed O2 update-and-clear algorithms only.
Native O2 compilation requires `ET_TR3_O2_STEP_CLEAR_NATIVE`; its bridge requires
`ET_TR3_O2_STEP_CLEAR_BRIDGE`. The objective/schedule requires its existing CPU4
training/objective/retire flags and the accepted T4 L3S provider.

`cpu4-training-step-internal!(trainer)` returns the preallocated vector
`[activeTokens,nextUpdates,nextEpochs,observations]`. `observations` contains one
actual 12-byte little-endian CE `[numerator,weight,mean]` observation per configured
microbatch. Eshkol does not scalar-sum floating-point numerators. Each result is
independent of later steps. A separate reviewed provider can aggregate these
observations for evaluator/overfit metrics.

Private granular start/microbatch/prepare/commit/abort functions take an exact
registered frame (start takes the genuine trainer). Start captures the original
D2 current and epoch cursor, RNG, counters and all 17 TRAIN modes, allocating the
whole result before publishing a busy frame. Each microbatch obtains and owns a
real D2 batch, calls the accepted explicit Eshkol 21-forward/25-reverse schedule,
and releases the batch after all native READ children drain. At finite EOS the
step rewinds to the authentic epoch-start cursor and stages an epoch increment.
Configured accumulation is explicitly unsupported above 2^22 (4,194,304), so
even worst-case T4 mask weight `4*A` remains within the exact binary32 integer
range (2^24) throughout every admitted contribution. Signed-i64 overflow checks
also precede publication/update.

Prepare authenticates the same components, unchanged saved controls/modes, and
the complete successor C2-D2 cursor/count equation. It stages successor controls
and result counts, enters native WRITE (which proves READ children/storage pins
have drained), then asks real O2 to prepare the update. O2 checks all fourteen
finite gradients, contribution count and summed normalization weight and stages
parameters/moments/gradient clear. A prepared plan keeps native WRITE until
commit or successful abort. Commit rechecks identities/cursor before O2's first
write, executes its atomic update-and-clear, then performs only prevalidated
control stores and terminal WRITE leave. Impossible post-write defects fail-stop.
The committed flag is bound and its guard installed before O2 commit: pinned
fe9 guard lowering invokes `eshkol_push_exception_handler`, whose hosted runtime
can allocate via malloc (`codegenGuard` at codegen.cpp21765+, hosted runtime
`runtime_exceptions_hosted.cpp:1014–1020`). The installed handler rethrows a
categorized pre-write rejection while the flag is false; after successful O2
return the fixed flag store selects fail-stop. No new guard is installed after
the first parameter write. The cheap source regression checks this exact order.

Abort releases any owned batch, aborts the exact O2 plan, seeks to the saved D2
cursor, and clears gradients using genuine failure-atomic operations. Failed
plan abort retains the exact frame, plan and held WRITE because its copy/reset
plans can still pin storage. Seek/clear/leave failure also retains the exact busy
frame; WRITE is released only when native storage permits it. Retry the same
`cpu4-training-step-abort-internal!` frame after the dependency condition is
repaired. The top-level step attempts rollback before rethrowing a pre-write
error; rollback failure is itself reported with the exact frame still rooted.

The genuine staged fixture checks real contribution/update/clear, all1192
parameter bytes and all2384 moment words against independent orchestration using
ordinary O2 step plus explicit clear, actual retained CE observations, two
updates with finite EOS rewind, and automatic rollback before and after rewind.
It covers native READ/WRITE contention, rejected O2 plan abort retaining WRITE,
D2 seek and O2 zero-gradient allocation failures with same-frame retry, prepared
plan allocation failure/retry, and busy/foreign/consumed frame rejection. Test
only C code exposes existing O2 failpoints and compares genuine O2 snapshots;
it contains no update/numerical algorithm.

Cheap checks: `python3 tests/cpu4_step/test_source.py`, Guile source reading,
strict host C syntax for the test bridge, `bash -n` and `git diff --check`.
Supported normal/repeat and ASan/UBSan gate: run
`scripts/test-cpu4-training-step.sh FRESH_EVIDENCE normal` and separately
`sanitize` inside the pinned supported image under the shared heavy lock.
The recipe explicitly enables `ET_O2_TESTING` solely for native failpoint tests.

Compiled full-step/runtime acceptance: NOT RUN. This leaf does not authenticate
a full C2 checkpoint, implement checkpoint resume, publish an installed trainer,
or claim overfit/evaluation metrics. Existing objective numerical checks remain
the numerical oracle; the new integer reference checks finite cursor equations.
