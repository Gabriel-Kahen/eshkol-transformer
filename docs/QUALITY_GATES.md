# Quality gates

## CI tiers

The accepted [CI-E2 topology](CI_EFFICIENCY.md#ci-e2--native-critical-path-and-main-push-reuse)
uses one full-coverage engine in
`.github/workflows/full-coverage.yml`, shared by blocking CI and exhaustive
acceptance. It runs one clean canonical build with smoke/benchmark, nine component
suites including the separate unchanged O2 optimizer gate (checkpoint I/O owns C1
alone), and six independent C2 groups. Accepted M3T extends this
engine with one diagnostic-transport suite: 17 suites and 24 top-level local test
commands, preserving all 16 accepted suites and their 23 commands. Historical
accepted CI-E2 evidence below remains a 16-suite / 23-command measurement; it does
not establish M3T acceptance. M3T's separate accepted evidence is supported run
35441033357, independent PR #84 review, and identical-tree merge `f501609`;
see [transport acceptance](M3T_TRANSPORT.md).
Accepted M3 adds the model-composition suite, bringing the then-current full coverage to
18 suites / 25 commands. Supported run 35465971406, independent PR #88 approval,
identical-tree merge and focused merged-main checks establish the separate
[model acceptance](M3_MODEL.md). Earlier 16/23 and 17/24 results retain their
historical scope.
L3S adds one masked-objective gate to native-numerics, preserving that full union:
18 suites / 26 commands. Supported candidate run 35531132634, independent L3S-R
approval, PR #91 merge, the focused merged-main check and fresh combined-main
run 35680925863 establish bounded [L3S acceptance](L3S_MASKED_OBJECTIVE.md#integration-provenance).
The gate covers all six schemas, real L2 gradients, bool/f32/weighted masks, exact
environment/byte atomicity,
observable compiled mutants and the documented two-term reversal equivalence,
private I1/I2/AOT composition, exact manifests and ASan/UBSan/LSan.
G3-N adds the separate forward-only `g3n-forward` suite, bringing the union to
19 suites / 27 commands. Independent approval 5771670331, supported run
35681252651 attempt 2, PR #97 merge and the root focused merged-head gate establish
bounded [G3-N acceptance](G3N_PRIMITIVES.md#integration-provenance). The gate checks
all eleven exact operation/row pairs, numerical/metadata mutations, malformed
requests, byte/fenv atomicity, genuine I1/I2 borrows, exact packaging inventories,
two fresh deterministic private AOT builds/runs and ASan/UBSan/LSan. The initial
P1 timeout and sole authorized failed-job retry remain in the provenance; supported
candidate and merged prose-union trees are distinct. Newer main CI is tracked
separately and is not inferred green from the local compatibility retest.
The evaluator prerequisite union in PR #112 adds BOOL metrics to native-numerics
and the private D2 identity/idle gate to shard-loader. Current accepted main
`33a54ef7` therefore has 19 suites / 29 top-level commands. Supported run
[35744832879](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/35744832879)
passed the complete union at tree `2256194d2daeb233ec1745d0039b383d4358e3d5`,
identical to the actual merge. Main run 35798192984 verified and reused that
result. This accepts bounded prerequisites, not the complete evaluator.

C2 composition invokes each unique leaf gate once instead of repeating regression
tails nested inside other gates. Standalone core, load, and operational commands
retain their historical regression tails; explicit focused flags are only used by
the complete partition driver. Every purposeful repeated fresh-cache/AOT build,
compiler comparison, sanitizer, malformed-input case, exact-limit assertion, and
same-process memory/lifetime trajectory remains required.

No compiled test binary or mutable runtime state is shared between jobs. Each job
cleans/configures its own build and constructs the canonical prerequisites it
actually reads. The separate canonical-build job still runs `make clean && make
build`, smoke and benchmark. No cross-run binary cache or fresh-proof bypass is
introduced. The existing pinned compiler cache remains subject to toolchain
verification. Per-job prerequisite and test seconds are recorded in job summaries.

Outer budgets remain 105 minutes for both native suites, 75 minutes for ordinary
component and clean-build jobs, and 240 minutes for each C2 group. The previous
checkpoint budget is not confused with a runtime resource bound: inner compiler
timeouts, the fixed C2 RSS/arena ceilings, test counts, and leak checks are
unchanged. Main topology/selection and exhaustive evidence selection receive three
minutes; evidence and final-status jobs receive two minutes. An explicit structural
checker and mutation tests reject missing/duplicate gates, altered
environment/budgets, failure masking, and skipped final assertions.

PRs consisting solely of allowlisted prose (`README.md`, `CONTRIBUTING.md`, and
`docs/**/*.md`, excluding `AGENTS.md`) run topology and change-selection tests.
Renames are checked on both sides; mixed/unknown paths, empty diffs and unavailable
history require full CI. Every main push executes authenticated selection. A
prose-only main push may skip the matrix only after direct verification of completed
full coverage for its base code. An eligible merged code PR may reuse only its
directly verified original full result; missing, partial, pending, failed,
ambiguous, or stale evidence falls back to fresh full coverage. Skipped or reused
CI does not itself become reusable full evidence. Merge-queue runs, explicit CI
dispatches, and nightly acceptance remain fresh full coverage.

Nightly acceptance always runs fresh full coverage through the same engine.
Manual acceptance can reuse only the latest completed successful CI for the exact
candidate revision whose recorded actual checkout tree equals the candidate tree.
The verifier checks repository/run/attempt identity, every expected suite including
the clean build, successful evidence and final aggregation jobs, and the actual
tested commit/tree through GitHub. Missing, stale, partial, malformed, or failed
evidence cannot produce acceptance: it requires fresh full coverage instead.
An in-progress matching CI is reported as pending rather than launching a duplicate.
There is no arbitrary artifact execution and no reuse across changed trees. This
is reuse of a completed full-coverage result, not omission of a gate.

Historical baseline evidence remains valid only for its original revision:
- run 35350092230 completed its C2 job in 3h40m43s, with about 3h30m in testing;
  its predecessor job hit 240 minutes after a 44-minute clean build, before Q0,
  smoke and benchmark completed.
- run 35349163791 attempt 2 passed; native prerequisites took 30m50s and tests
  60m47s. Attempt 1's fresh N2 oracle mismatch remains a recorded failure. Later
  controlled runs isolated a recent same-class failure to 18 linear tensor words
  affected by Intel MKL dispatch; all 12 cross-vendor
  `MKL_CBWR=COMPATIBLE` controls matched the frozen fixture. The accepted
  reference-only wrapper pin changes no runtime or golden data.
- the predecessor-only 300-minute scheduling correction in PR #77 is separate
  from CI-E. Its then-running acceptance evidence was not cancelled or
  retroactively relabeled as evidence for this optimization.

Supported run 35393213200 passed the accepted engine's 15 suites, all 23 commands,
canonical smoke and benchmark. Independent authoritative review approved the exact
tree, and acceptance run 35401088338 verified and reused it with the full matrix
skipped, completing in 27 seconds. See [CI efficiency evidence](CI_EFFICIENCY.md)
for exact provenance, before/after measurements, and variance limits. The final
93m19s wall time is 131m39s / 58.5% below the 224m58s uninterrupted baseline;
421m23s summed runner time is 42m37s / 9.2% below 464m00s. A failed required
full-coverage or exhaustive run remains an acceptance blocker.

CI-E2 supported run 35407378830 at exact head
`22e54314078ffb133cd937bd454c89e6a7712282` passed all 16 suites, all 23
commands, canonical smoke and benchmark on the independently approved tree
`d5137fca07ef71648e108ac5776072097b551df3`. PR #82 merged that tree as
`f9d8366b17ecf408a60d8f62e8bf6c4d4eebbbe1`; main run 35410997717 directly
verified and reused the original full run with exactly topology success, full suites
skipped, and final success in 18 seconds. The supported full run took 55m22s wall
and 409m13s summed runner time. This is 37m57s / 40.7% wall and 12m10s / 2.9%
summed-runner improvement over accepted CI-E, and 169m36s / 75.4% wall and
54m47s / 11.8% summed-runner improvement over the uninterrupted baseline. These
public-repository runner-time measurements are not billing or cost claims. At this
documentation review, the live prose-base branch has not yet run; its subsequent
result is recorded in issue #81.

## Required on every numerical component

- Shape, dtype, device, and error-contract tests.
- Known-value forward tests.
- Central finite-difference gradient checks where differentiable.
- Independent reference parity for high-risk kernels.
- Repeated-input and accumulation cases.
- NaN, Inf, empty, boundary, and malformed-input cases.

## Required integration gates

1. Tokenizer encode/decode round-trip and deterministic vocabulary training.
   BPE evidence includes exact merge tie-breaking, admitted document-order and
   chunk-partition invariance, whole/stream rank-stage parity, all-byte fallback,
   every strict UTF-8 split plus F0/F4 scalar boundaries, and measured exact-limit
   admission. The decoder maximum includes both one 73,728-ID chunk and 73,728
   one-ID omit-only chunks under count-pinned time/RSS/no-warning gates. Delivered
   compiled parsers must reject every frozen header/payload/order invariant, and the
   compiled D1 seam must preserve malformed/truncated/checksum corruption categories
   while rejecting fingerprint or vocabulary mismatch.
2. Shard corruption detection and exact loader-cursor resume.
3. Tiny transformer forward and gradient parity against a frozen oracle fixture.
4. One-batch overfit.
5. Small-corpus loss reduction and held-out evaluation.
6. Bitwise-identical next training step after checkpoint reload where supported;
   otherwise a documented tolerance with a proved cause.
7. AOT/JIT parity.
8. Fixed-seed reproducibility.
9. Flat resident memory across a long bounded training loop.
10. Save/reload generation equivalence.

## Required D2 review gates

- Frozen Q0-format exact `int64` inputs/targets and `bool` loss-mask fixture,
  regenerated byte-identically twice by a stdlib-only development generator.
- Empty, singleton, short, exact, padded, packed, unpacked, unused-final-row, and
  arbitrary cross-shard shift/mask cases; shuffle-window and rejection-sampling
  golden vectors; exact replay from start, batch, final, and EOS cursors.
- Two fresh-cache strict AOT compilations and runs with byte-identical artifacts and
  stdout, plus source/symbol/dependency inspection proving no Python runtime or
  development-oracle path in the production closure.
- Public compiled end-to-end D1-shard-to-batch checks must inspect exact carrier
  contents through the same-aggregate scoped view for every packed/unpacked shift and
  mask boundary. Warning-clean C11 and C++17 consumers, ASan/UBSan and supported-lane
  LSan, forged/stale/cross-owner/active-borrow negatives, callback-throw cleanup,
  injected allocation/read failures, and native live-allocation return to baseline
  are required.
- Exact/one-over manifest, shard, total-token, batch-payload, and shuffle-window
  admission. Report semantic payload, peak RSS, and file-descriptor deltas separately;
  the bound must be independent of total corpus size and no smaller hidden limit or
  allocation fallback may be used.
- The optimized-AOT long loop must put each next/validate/use/release interval in one
  lexical `with-region`, compare the exact retained Eshkol arena byte counter after
  1,024 and 8,192 batches on the same admitted corpus, and require byte equality.
  Native carrier/live counts must return to baseline. Report RSS separately as an
  advisory host/runtime measurement. A shell deliberately retained or promoted
  across the region is caller-owned and must be separately accounted. This gate is
  the implementation requirement described by the amended lifetime acceptance in
  [issue #1 comment 5608140148](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5608140148).
  Stale/idempotent-alias guarantees require live shell allocations; accessing freed
  lexical-region storage is outside those guarantees.
- Equal-token one-shard and many-shard corpora must prove that optimized-AOT open
  retention differs by no more than one exact admitted working payload and that a
  packed batch crossing the same tokens retains exactly equal arena bytes. Exact
  limit admission must pass and one-over must fail.
- D1 manifest-last publication and corrupt/missing/mismatched tokenizer/shard
  negatives, on-demand revalidation after namespace mutation, cursor failure
  atomicity, invalid-config-before-I/O precedence, final-live-batch-before-EOS, and
  idempotent release of every authentic already-issued generation after successor,
  seek, and close.
- Exact `ESHKDCU1` 1.0 `208+F` physical/canonical/error-precedence matrices, exact
  58-global/52-export source-composed aggregate manifests, trusted-root uniqueness,
  hostile include/link/privacy negatives, and affected A0/D1/T1/T2/I2/N2/Q0 gates.
  The private boundary must admit exactly the dataset and batch compiled-constructor
  code identities, retain no shell/environment pointer, and reject treating that
  per-process mechanism as a public, serialized, or cross-aggregate ABI.
- The exact generated source closure and every public AOT executable must be scanned
  for production-language and dependency isolation.
- Supported Ubuntu 22.04 x86-64 / LLVM-Clang 21.1.8 CI at the reviewed exact head,
  followed by independent D2-R, merge, merge-head retest, and acceptance-document
  follow-up. ROADMAP may move from `review` to `complete` only after all are
  satisfied.

## Required C2 checkpoint gates

- Prove exact outer/C1 fixed-header staged admission on one `O_NOFOLLOW` regular-file
  descriptor, retained-header byte equality, final same-fd size/EOF agreement, and
  zero codec/publication calls for every unstable, corrupt, version, or limit failure.
- Exercise the accepted C2 1.0 header, every checked sum/product/span, inert identity
  grammar, X1 canonical projection, both real D2 cursor families/relations, complete
  C1 and O2 metadata, per-tensor and whole digests, and all cross-component counts,
  paths, shapes, aliases, seeds, vocabulary, and update counters.
- Prove failure-atomic detached P1/O2 reconstruction and C2 publication at every
  allocation/transfer/release boundary. Exact dead release is idempotent; forged,
  copied, wrong-owner, busy, active-borrow, and provider-defect cases retain the
  documented error and exact-once cleanup behavior.
- Repeated model and moment saves use the direct exact-bit copy seams and leave all
  live plus newly instrumented retired-control counts/bytes flat. Scratch remains
  lexical-region bounded; no implicit promotion or finalizer is evidence.
- The 16-MiB file, 512-KiB artifact metadata, 8-MiB tensor, and 64-tensor operational
  tuple requires exact/one-over and jointly attainable pinned-runtime measurements,
  bounded time, no heap warning, and peak plus retained RSS below 512 MiB. It is not
  accepted merely because the wire parser declares those constants.
- Public integration proves the actual source-composed 81/75 candidate counts,
  81 public-name strings, eleven truthful capability rows, exact supplied-report
  admission for every staged tensor, hostile linkage/input isolation, deterministic
  clean builds, flat 1,024/8,192-iteration root retention, full predecessor gates,
  supported blocking CI, and independent exact-head review.

## Required configuration gates

- Strict, bounded, non-executable parsing with duplicate and unknown keys rejected
  before a configuration value is constructed.
- Every explicitly present source leaf is validated before overrides, so an override
  cannot mask an invalid source type, range, enum, or policy. Defaults, valid input
  values, explicit overrides, and absent-field derivation are tested at their
  documented precedence boundary, including incompatible combinations.
- Repeated fresh compilation of the configuration test, plus byte-identical canonical
  manifests and fingerprints across fresh processes, working directories, and
  hostile-environment runs.
- Golden canonical bytes, direct external-fingerprint recomputation,
  resolved/provenance mutation, malformed manifest, and unsupported version/feature
  tests.
- Production dependency inspection proving no Python/PyTorch runtime, evaluator,
  include expansion, environment interpolation, or hidden execution fallback.
- Successor aggregate boundary inspection proving hostile include/path isolation,
  exact repository tuple admission, localized private/native symbols, no archive
  index leakage, deterministic localized objects/archives/evidence/AOT binaries,
  public-caller closure, and duplicate registry ownership rejection.

## Required K2 capability-facade gates

- Genuine fixed-I2 K1 discovery, exact descriptor/reserved/table/callback audit, and
  exact sorted eleven-row report audit before publication; every allocation and
  post-discovery audit failpoint cleans up and permits retry. Emit the genuine
  K1/I2 report twice and assert byte equality, exactly 3,133 bytes, and SHA-256
  `50c7078af4d3e495c4d668b6b8aaab29024cbab6a775fa6e98c9e9d10124389e`.
- Exact rank-zero/rank-one boundary matching plus the ordered five-shape rank-two
  matrix with both boolean determinism values; rank-two near-miss/one-over,
  higher-rank, wrong symbol and all ten unverified-row nonmatches; complete
  logical-A0 versus K1-representability checks for all four symbol positions, rank,
  and unsigned extents without truncation or contradictory native validation.
- Three pairwise-distinct compiled closure factories, authentication before hidden
  query, relay/query-capture/wrong-kind/forged/unregistered/stale-origin negatives,
  private test-only injected failed-publication candidate invalidation, and ordinary
  in-region alias success.
- Exact 264-byte aligned little-endian private diagnostics, admission precedence,
  full success/error writes, unchanged invalid buffers/tails, category/return/code
  consistency, and separate K1 versus K2-private E1 source mapping.
- Fresh-copy mutation isolation for every report, entry, constraint and request
  carrier; optimized-AOT 1,024/8,192 lexical-region equality with native singleton
  and caller-retained shells measured separately.
- One completed aggregate per process, exact 59-global/53-export/59-public-string
  manifests, source/native dependency closures, duplicate-aggregate and hostile
  repository tuple rejection, strict fresh-cache AOT with every arity, deterministic
  rebuilds, sanitizers, and production Python/PyTorch isolation.
- Fork evidence is limited to serialized single-threaded fork with no K2 operation
  in flight. Parent-origin shells reject before payload/provider use and only a newly
  audited child runtime is eligible. No multithreaded-fork, reentrancy, signal, or
  inherited non-K2 receiver claim is accepted.

## Performance evidence

Benchmarks record commit, hardware, OS, compiler, backend, dtype, tensor shapes,
warmup, repetitions, throughput, latency, and peak memory. A backend is not called
accelerated until execution on that device is directly observed and tested.

## Required native-lifetime gates

- Enumerate every owned-allocation transition: publication, ledger insertion,
  receiver transfer, rollback, explicit release, and callback defect. Each owned
  carrier must have exactly one owner and one destruction event.
- Repeated success and every injected partial-failure path must return provider/native
  live-allocation counts to their declared baseline. Process-lifetime identity
  tombstones are acceptable only when they retain no native tensor storage.
- Explicit release must be idempotent and must invalidate dependent handles before
  storage destruction. Forged, copied, stale, cross-owner, use-after-release,
  double-release, active-borrow, and callback-failure cases are negative tests.
- Run ASan/UBSan and LSan on a supported lane where the runner permits leak
  detection. A local environment that disables LSan is recorded as a limitation,
  not reported as leak evidence.

## Merge policy

- No red required gates.
- No unresolved red exhaustive-acceptance run on the candidate revision for a release
  or workstream acceptance.
- No undocumented fallback or unsupported case.
- No public format change without a version/migration decision.
- No performance rewrite without correctness parity.
- Cross-workstream API changes require orchestrator review.
