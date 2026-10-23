# Distinct external CPU4 training resume gate

This new test-only leaf starts from immutable `b0af97d10a96fc62ef414a26953582f5c79ad642`
and preserves the earlier accepted normal/repeat gate and its receipts. It adds
no public API, production numerical path, checkpoint format or native owner.

The future producer process creates the genuine fixed CPU4 N1/T4/V256 candidate,
trains uninterrupted from A0 through A125 and A250, and saves both complete C2
images to new paths. Its A125 file must match the retained 25,504-byte external
checkpoint SHA256 `499008516bce12c9ca07de19e97cee99ec373b0070cf13b34d035296889692cc`
before the host accepts A250 as the independent oracle. The producer never
loads or restores a checkpoint.

A distinct, directly observed child PID calls the existing private FILE factory
once on that external A125 input. It saves the restored image to a new file and
compares the entire file with external A125 **before step 126**. The provided
`c1-read-file` helpers use the bounded C1 subpolicy for raw bytes; they do not
create another C2/model owner. It then performs exactly steps 126..250, evaluates
the existing held-out fixture and saves B250. The host requires full A250/B250
C2 byte equality and exact nine-word metrics. This includes all 14 unique value
planes (4,768 bytes), all 28 moment planes (9,536 bytes), and persisted topology,
aliases, optimizer, scheduler, RNG, cursors, tokenizer/config identities and
controls. Nested C1 serializes 8,864 value bytes including the tied 4,096-byte
embedding copy. Its digest is labelled with that alias explicitly.

Three short additional processes reject a checksum-corrupt copy, truncated copy
and genuine seed-1728 config against the retained seed-1729 checkpoint. Exact
error category/operation, zero training steps and cleanup are required. Snapshots
are released, children retired and D2s closed before PASS. Caller-owned canonical
T2 remains until process exit, matching b0. No output overwrites are permitted.

`driver.py` reuses the frozen 211-member production manifest, original eight
caller pins, same 36-TU normal observer overlay and exact four resource files.
The new 14-row dependency manifest additionally freezes directly reused b0
helpers. `Runner` retains actual child PIDs, argv, bounded process-group timeout,
observed return code and raw streams. Source/dependencies and the external input
are checked before/after, including program failure. Outputs and oracle metadata
are sealed by SHA256 manifests. The root-owned prepared recipe retains one
`heavy.lock`, f31/fe9/C7, one compile job, 3 CPUs and 8 GiB, 3600-second inner and
3900-second outer bounds. Each positive runtime has 900 seconds; each negative
has 30 seconds. Whole-gate timeout can terminate an individually valid slow stage.

Run only cheap source/host tests here:

```
python3 -B -m unittest tests.cpu4_fresh_resume.test_host -v
```

These tests generate deliberately non-numeric parser doubles, fake scenario
transcripts and Python-only PID/timeout probes. They do not establish training
correctness. Root must independently review and schedule the frozen recipe before
any native compilation, container, Eshkol binary or 375-step training execution.
Compiler/runtime, SAN, installed CLI, generation, GPU, mixed precision and full
CPU-v1 admission remain **NOT RUN** for this leaf.
