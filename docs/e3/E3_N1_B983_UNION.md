# E3 N1 source-only union on the main/CLI lineage

Status: **source-only candidate; real-read and all authority gates HOLD**.

This union starts from `b98321efa4f2c049088a489b590f6b9f749c048a`
(tree `a15d1a8c5725d72a89576dea723ba1a84cb4b7b0`) and carries the 55
N1-specific paths from `a632c80ffa531ea4537b5ff963c4e31246472f7e`:
`docs/e3/E3_N1_*`, the three f32 JSON request/input records,
`tests/e3_n1_broker/*`, and N1-specific `tests/e3_private/*`. The generated
attempt contract corrects the setup worker's phase-pipe EOF description;
the outer observer checks setup report EOF. The small-observation
destination-negative test now uses an explicit `/var/tmp` path. The N1
raw-capture scanner and its test now use the current process image's Linux
`VmHWM` peak in place of process-lifetime `ru_maxrss`; the synthetic metadata
test fixture now lives under the caller's home instead of shared `/tmp`.
The raw-capture contract records the measurement change. The other 49 carried
N1 files retain their source-commit bytes. The prior
failed evidence and source branch remain preserved; neither is replaced by
this union.

The older N1 lineage conflicts with current main/CLI changes in
`docs/ROADMAP.md`, `scripts/build-e1b-consumer.sh`,
`scripts/e3-private-package-policy.sh`, `src/eshkol_transformer/e3_frame.c`,
and `tests/e3_private/predecessor_sources.sha256`. The roadmap receives this
additive status entry. The three scripts/native paths keep the b983 bytes;
`predecessor_sources.sha256` remains absent, as it is in b983. The selected
N1 files do not invoke those scripts or name the native frame or predecessor
checksum manifest. This is a source/docs/tests union, not a compiled package
or CLI integration.

The source-only tests use generated fixtures and do not open OCI/Docker input.
The broker suite passes 24 tests. Earlier direct raw-capture `-m unittest`
runs had seven failures/errors because this host reported a 2.14-GiB
`ru_maxrss` while `/proc/self/status` reported an approximately 12-MiB
`VmHWM` for the same Python process. The direct suite now passes 14 tests;
missing/malformed and over-cap `VmHWM` cases still HOLD. The metadata fixture
no longer has shared `/tmp` as an ancestor when the caller's home is outside
`/tmp`: 16 tests passed during 23,533 unrelated `/tmp` changes. A deliberate
fixture-ancestor change still returns `source-identity-drift`; the production
identity comparison is unchanged. These test-path fixes do not establish
protected custody or the 30-second/256-MiB whole-attempt predicate.

Protected pre-read custody, trusted actor graph closure, v3 receipt/seal
authority, actual OCI read, the 30-second/268,435,456-byte attempt predicate,
and E3 §9 compiled acceptance remain separate gates. See the
[authority host gate](E3_N1_AUTHORITY_HOST_GATE.md),
[protected attempt boundary](E3_N1_F32_ATTEMPT_BOUNDARY.md), and
[generated attempt](E3_N1_GENERATED_ATTEMPT.md) for the exact limits.
