# G3-R private typed RNG import implementation leaf

**Source candidate; supported gate pending.** The guarded
`ET_G3T_RECORD_RNG_IMPORT_PRIVATE` native constructor creates an exact G3-T
kind-8 clone from four signed i64 words after version/seed validation. The
source-private Eshkol wrapper enrolls a pending 12-slot entry in the same
G3-T registry, uses the existing native release on cleanup, and publishes
the live shell only after its pointer and ledger are installed. The existing
`:rng` generator constructor copies all four words, including both counters.
The feature-off aggregate omits the import symbol and wrapper.

This scalar constructor has no checkpoint authority. A trusted future restore
transaction must authenticate the immutable G3RCV1/C2 pair, policy,
model and T1 identities before calling it. No file I/O, live model restore,
cache/logit replay, terminal output or continuation equivalence is provided.
Malformed G3RCV1 bytes belong to the separate record codec; this leaf has no
decoded-record input and rejects malformed scalar domains only. Its supported
record-codec predecessor gate remains pending; malformed-record rejection must
be composed with this import before a restore transaction is admitted.
The accepted [ownership contract](G3_R_TYPED_RNG_IMPORT_CONTRACT.md) remains
the controlling boundary.

The source fixtures include native word-matrix/zeroization/allocation checks,
Eshkol registry, forged/stale/busy/cross-registry and rollback checks, actual
P1/G1 imported categorical/greedy runs, and an independent G3-S/Philox
selected-ID and successor-RNG oracle. The supported pinned normal/repeat and
ASan+UBSan+LSan run is required before this leaf can be marked accepted.
