# TR3 private finite-D2 checkpoint trajectory witness

The test-only `native/tr3_c_private_resume_trajectory_root.esk` composes the
accepted TR3-C snapshot/SAVE and C2 LOAD/joint-restore source with the accepted
fixed-profile step composer in one Eshkol registry universe. It adds no
installed trainer API or C ABI.

The compiled witness creates genuine two-row D2 trainers at context length
two and accumulation counts `A=1`, `A=2`, and `A=3`. Each performs one real
configured update (`K=1`),
checks token/update/epoch/cursor controls, and saves a versioned, checksummed
C2 checkpoint through `tr3-c-snapshot-save-internal!`. A fresh, separately
enrolled trainer with the same X1/T2/D2/M3T/O2 profile receives that file
through `tr3-c-checkpoint-restore-internal!`. The receiver retains its exact
model, optimizer, and dataset shell identities. Its snapshot then matches the
source at `K` for all 14 parameter payloads, 28 optimizer moments, RNG,
tokenizer/X1 identity, both D2 cursors, and T/U/E controls.

The uninterrupted and restored trainers each perform three further genuine
updates (`R=3`). All three profiles cross finite D2 EOS in that suffix.
The witness compares the complete detached image after every corresponding
update and byte-identical final canonical C2 checkpoints. The `A=1` and
`A=3` cases also verify unequal active-token counts across updates and
inject a failure after O2 prepare but before the first parameter write; the
exact `K` image survives and the full suffix succeeds on retry. The `A=2`
case rejects checksum corruption and an authentic X1 run-seed mismatch before
receiver mutation. Its injected failure immediately after the restored
trainer's EOS rewind likewise preserves the exact `K` image.

This is a private three-step trajectory proof for one CPU f32 model and
two-row corpus. The full gate in `TR3_C_LIVE_STATE_CONTRACT.md` §13 still
requires fresh-process comparison, next-step metrics, effective learning rate,
and broader failure and environment evidence. Public trainer packaging and
C ownership remain separate dependencies.
