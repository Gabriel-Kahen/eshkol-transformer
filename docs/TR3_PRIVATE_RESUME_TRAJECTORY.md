# TR3 private finite-D2 checkpoint trajectory witness

The test-only `native/tr3_c_private_resume_trajectory_root.esk` composes the
accepted TR3-C snapshot/SAVE and C2 LOAD/joint-restore source with the accepted
fixed-profile step composer in one Eshkol registry universe. It adds no
installed trainer API or C ABI.

The compiled witness creates a genuine two-row D2 trainer at context length
and accumulation count two. It performs one real two-microbatch update (`K=1`),
checks token/update/epoch/cursor controls, and saves a versioned, checksummed
C2 checkpoint through `tr3-c-snapshot-save-internal!`. A fresh, separately
enrolled trainer with the same X1/T2/D2/M3T/O2 profile receives that file
through `tr3-c-checkpoint-restore-internal!`. The receiver retains its exact
model, optimizer, and dataset shell identities. Its snapshot then matches the
source at `K` for all 14 parameter payloads, 28 optimizer moments, RNG,
tokenizer/X1 identity, both D2 cursors, and T/U/E controls.

The uninterrupted trainer and restored trainer each perform one more genuine
step (`R=1`) that starts at finite D2 EOS, rewinds to epoch start, and commits
the next update. The witness compares their complete detached images and
byte-identical final canonical C2 checkpoints. It also rejects checksum
corruption and an authentic X1 run-seed mismatch before receiver mutation.
An injected failure immediately after the restored trainer's EOS rewind
preserves its exact `K` image and supports same-trainer retry.

This is a private one-step trajectory proof for one CPU f32 profile. It is not
the full interrupted/resumed equivalence gate in
`TR3_C_LIVE_STATE_CONTRACT.md` §13, which requires longer suffixes, other
accumulation profiles, next-step metrics, effective learning rate, and broader
failure and environment evidence. Public trainer packaging and C ownership
remain separate dependencies.
