# E3 N1 source-only union on the main/CLI lineage

Status: **source-only candidate; real-read and all authority gates HOLD**.

This union starts from `b98321efa4f2c049088a489b590f6b9f749c048a`
(tree `a15d1a8c5725d72a89576dea723ba1a84cb4b7b0`) and carries the 55
N1-specific paths from `a632c80ffa531ea4537b5ff963c4e31246472f7e`:
`docs/e3/E3_N1_*`, the three f32 JSON request/input records,
`tests/e3_n1_broker/*`, and N1-specific `tests/e3_private/*`. The generated
attempt contract corrects the setup worker's phase-pipe EOF description;
the outer observer checks setup report EOF. All other carried N1 files retain
their source-commit bytes. The prior failed evidence and source branch remain
preserved; neither is replaced by this union.

The older N1 lineage conflicts with current main/CLI changes in
`docs/ROADMAP.md`, `scripts/build-e1b-consumer.sh`,
`scripts/e3-private-package-policy.sh`, `src/eshkol_transformer/e3_frame.c`,
and `tests/e3_private/predecessor_sources.sha256`. The roadmap receives this
additive status entry. The other four paths keep the b983 bytes; the selected
N1 files do not invoke those scripts or name the native frame or predecessor
checksum manifest. This is a source/docs/tests union, not a compiled package
or CLI integration.

The source-only tests use generated fixtures and do not open OCI/Docker input.
The broker suite passes 24 tests. The N1 private suite passes 98 tests when
loaded through a `python3 -I -B -` stdin runner. On this host, invoking the
same suite with `python3 -I -B -m unittest discover` reports an initial
`ru_maxrss` above the fixed 2-GiB raw-capture cap and causes seven
raw-capture preflight failures/errors. The cap is unchanged; that runner
discrepancy is unresolved and is not admission evidence. One stdin rerun
also encountered a transient `source-identity-drift` in the large synthetic
metadata case; an immediate repeat passed all 98 tests.

Protected pre-read custody, trusted actor graph closure, v3 receipt/seal
authority, actual OCI read, the 30-second/268,435,456-byte attempt predicate,
and E3 §9 compiled acceptance remain separate gates. See the
[authority host gate](E3_N1_AUTHORITY_HOST_GATE.md),
[protected attempt boundary](E3_N1_F32_ATTEMPT_BOUNDARY.md), and
[generated attempt](E3_N1_GENERATED_ATTEMPT.md) for the exact limits.
