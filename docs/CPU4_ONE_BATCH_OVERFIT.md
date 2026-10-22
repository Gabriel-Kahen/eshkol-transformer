# CPU4 fixed one-batch overfit gate

This fixture-only leaf trains the same canonical seeded C4 through the actual
`cpu4-training-step-internal!` producer. It changes no production source, creates
no M3T model, and uses no Python/PyTorch in the Eshkol training path. The exact
profile is CPU-f32 N1/T4/V256/D4/L1/F8, two query/KV heads with Dh2, fourteen tied
canonical parameter allocations (1192 values / 4768 bytes), learned positions
4x4, seed 1729, and one raw-byte D1 sequence `(3,197,3,41,7)`. D2 unpacked N1/T4
produces exactly one full-mask row with four next-token targets. Accumulation is
one and the same finite row is rewound across 250 actual updates.

Optimizer choices reuse the accepted TR3 overfit witness: constant AdamW,
learning-rate bits `3dcccccd` (0.1), betas `3f000000`/`3f000000` (0.5/0.5), epsilon
`3a83126f` (0.001), zero decay and no clipping. The prior TR3 one-batch gate
predeclares at least 25% loss reduction over 32 updates. Before any CPU4 overfit
execution, this gate fixes **at least 75% loss reduction and final observed mean
CE <=1.0**, over 250 actual updates. Failure does not authorize relaxing that
criterion or changing the optimizer silently.

The step returns actual 12-byte little-endian CE `[numerator,weight,mean]` per
microbatch. The fixture prints exact words at steps 1, 2, 10, 25, 50, 100, 150, 200, 250,
plus actual trainer counters. No Eshkol scalar float summation or separate model
evaluation is used. Observation timing is explicit: step 1 sees zero prior
updates; step 250 sees 249 prior updates, then the 250th update commits. Thus the
last reported loss concerns the model after 249 updates. This does not claim a
separate evaluation of the final 250-update model. A development-only Python
checker decodes the emitted words after native execution, verifies finite
nonnegative CE (positive baseline), exact full-mask numerator/4 relation, all
sample timings/counts,250 updates / 1000 tokens / 249 epochs, and the fixed loss gate.

Every step authenticates the original canonical trainer/binding, checks all
fourteen gradients absent, proves native IDLE through a genuine READ enter/leave,
and checks the finite one-row cursor equation. Exact before/after snapshots of
all 14 parameters must total 4768 bytes and show at least one changed allocation;
the canonical slot 7 remains #f and learned-position shape remains4x4.

Retention assertions apply only to this finite 250-update witness. Read-only test
instrumentation requires all F32 live tensor/parameter/borrow/copy/gradient/reset
plan/clone counts return to the constructor baseline after every step (14
parameters and 56 live tensors including optimizer moments). It measures initial
and final F32 retired control bytes, requiring growth <=64 MiB, and measures 250
committed O2 plans with zero live/aborted plans or live stages and retained O2
control bytes <=256 KiB. Exactly 250 rooted closed Eshkol step frames are checked.
These bounds were fixed before execution; retained tombstones are intentional.
Native transport/lease child control bytes, D2 shell/other Eshkol heap retention
and whole-process RSS are not measured here. This is not a constant-memory or
long-run retention claim, and sanitizers do not establish such a bound.

Cheap checks:

```
python3 tests/cpu4_overfit/test_source.py
python3 tests/cpu4_overfit/test_checker.py
bash -n scripts/test-cpu4-one-batch-overfit.sh
```

After independent source/fixture/checker/native-instrumentation/recipe review,
the orchestrator runs the normal and sanitizer modes of
`scripts/test-cpu4-one-batch-overfit.sh FRESH_EVIDENCE MODE` in the pinned
supported image under the shared heavy lock. Native sources come from the
accepted training closure plus the existing O2 optimizer/bridge and read-only
test bridge. `ET_F32_TENSOR_TESTING` and `ET_O2_TESTING` enable measurement only;
no failpoint is selected. Each binary runs twice with arena poison, empty stderr,
identical complete stdout and independently checked loss records. Sanitizer mode
adds ASan/UBSan/leak detection. Compile timeout is 1200s; each 250-update run is
bounded to 600s. The script creates no nested lock or parallel compilation.

Compiled overfit execution: **NOT RUN**. The a8 full-step gate found a separate
test-only moment-inspection precondition bug; accepted repair 2574d0fb preserves
its production code and has a fresh retry pending. This fixture uses no moment
snapshot helper. A successful step retry remains a prerequisite to overfit gate
admission. No checkpoint/resume/held-out/generalization/installed trainer claim
is made. Held-out evaluation waits for the separately accepted E3 consumer.
