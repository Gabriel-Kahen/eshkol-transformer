# G3-R private typed RNG import implementation leaf

**Supported private leaf accepted.** The guarded
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
record-codec predecessor gate has passed separately; malformed-record rejection
must be composed with this import before a restore transaction is admitted.
The accepted [ownership contract](G3_R_TYPED_RNG_IMPORT_CONTRACT.md) remains
the controlling boundary.

The source fixtures include native word-matrix/zeroization/allocation checks,
Eshkol registry, forged/stale/busy/cross-registry and rollback checks, actual
P1/G1 imported categorical/greedy runs, and an independent G3-S/Philox
selected-ID and successor-RNG oracle. The independently accepted supported
fe9/f31 gate ran on source `8ebed705452e20176c3524e159325c3034fdf12c`
(tree `4217a5e80101ba358aac73597e8b60c99e514ac5`) from 2026-10-02
10:57:10–11:01:28 UTC. Feature-off 1 and native C passed. Normal and
ASan+UBSan+LSan each passed 101 checks with identical output and empty compile
and runtime stderr. The gate validated a repeat comparison, though it did not
retain separate repeat logs. Three actual imported P1/G1 runs matched the
independent G3-S/Philox oracle for selected IDs and published clone successor
words. The 29-file evidence seal is
`g3r-rng-import-8ebed70-fe9-f31-20261002-prepared-a/SHA256SUMS-RUN`
(`90c52bf074c7cea0c9ec3004aae08ee331c0de0d05e78c203e72a6482d1e0fbf`),
with launcher/Docker exits 0 and pre/post pins PASS. Acceptance covers the
scalar private kind-8 owner import only. It does not authenticate a G3RCV1
record, restore C2 or live owners, replay draws, or establish continuation.
