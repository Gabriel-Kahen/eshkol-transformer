# TR3 private single-update composition leaf

The test-only `native/tr3_step_private_root.esk` loads the accepted TR3 lease
root, TR3-B D2/M3/L2/L3S objective extension, and TR3-O update-and-clear
extension exactly once in one Eshkol registry universe. It adds no installed
name, public trainer entry, native ABI, or checkpoint format. Its purpose is
to prove that the existing source operands can produce one genuine optimizer
update together before a trainer transaction is implemented.

The compiled witness uses the fixed CPU f32 diagnostic model and raw byte
tokenizer. Its D2 batches have i64 input/target and bool mask shape `[1,2]`,
with genuine masks `11` and padded `10`. The M3 graph logits and seed are
f32 `[1,2,256]`. TR3-B calls the accepted indexed-cross-entropy and masked
objective kernels, then the accepted analytic M3 VJP. The 14 unique P1
parameter/gradient shapes are the fixed set in [TR3 P1](TR3_P1_FIXED_SET.md):
four `[4,4]`, two `[4,8]`/`[8,4]`, six `[4]`, one `[256,4]`, and one `[2,4]`.
The head/token embedding tie is one unique parameter, not a second gradient.

Two contributions carry weights 2 and 1, ordinals 0 and 1, and the exact
binary32 cumulative weight 3. O2 prepare checks all 14 numerator-gradient
counts/weight bits and normalizes once before AdamW. The test's O2 config is
constant-schedule AdamW, learning rate 0.1, betas 0.5/0.5, epsilon 0.001,
no clipping or decay. A wrong expected weight rejects before any update;
parameter bytes, O2 count, gradients, and cursor remain unchanged. An exact
plan abort also leaves them unchanged, and an exact retry commits one update,
clears all 14 gradients, advances the O2 count to one, and rejects repeat
commit. Malformed observation length rejects before the first VJP and
preserves absent gradients. Input, seed, graph, batch, lease, and dataset
owners are explicitly released or unenrolled. The fixed M3T model and live O2
moments retain their accepted process-local lifetime.

The development-only pinned PyTorch checker independently reconstructs both
objective observations, the exact initialized `[4,4]` key weight, its summed
two-batch numerator gradient, and one AdamW update. The supported f31/LLVM21
gate compares finite values under the existing TR3 reference tolerances,
requires a changed parameter and rejects an injected NaN observation. The
compiled witness itself checks the component failure/retry and ownership
invariants. It uses the sealed native closure from the linked private package;
the gate verifies that predecessor seal and compiles this source fresh under
the pinned `81298` runner with the network disabled.

This is **not** `trainer-step!`: the witness enters and exits private lease
phases and publishes the trainer counters directly as test code. It does not
prove a no-fail control tail, whole-step rollback after a failed microbatch or
release, EOS/epoch replay, public metrics, all-parameter numerical trajectory,
retention slope, C construction, or resume equivalence. The next source leaf
must own the complete step-start cursor/RNG/mode/counter rollback and cleanup,
then bind its O2 plan and trainer controls before the first parameter write.
