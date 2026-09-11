# Integration log

This repository-side ledger mirrors contract decisions recorded in
[issue #1](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1).
Only the integration owner changes a proposed decision to `accepted` after review.

## 2026-09-10 — O2 merged implementation accepted; C2 prerequisites explicit

- **Decision:** O2 is accepted and complete within its unchanged bounded contract.
  The historical premerge entries below are superseded in status, not erased.
- **Evidence:** [O2 acceptance and merged-main retest](WAVE2_PRIMITIVES_ACCEPTANCE.md#o2-merge-and-bounded-post-merge-retest)
  records PR #58 merge `884a074`, approval comment 5627783349, supported blocking
  run 34533652814 and exhaustive run 34533681600, exact reviewed/CI/merge tree
  identity, and the compatibility-only merged-main retest with 6,201 adversarial
  checks, AOT, reference parity, exact authority manifests, sanitizers/LSan,
  topology, isolation, and smoke. No numerical/API/ABI/format change is made here.
- **C2 coordination:** [amended architecture direction 5628231282](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5628231282)
  and [required pre-freeze corrections 5628292779](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5628292779)
  leave exact checkpoint bytes and callable seam contracts unfrozen. K2 issue #73
  is a separate active production capability-facade prerequisite; C2 public
  integration waits for its independent approval and merge. Neither task is
  complete. Component-state continuation and atomic file publication belong to
  C2; atomic live-trainer restore/full trajectory and generation proof remain
  TR3/G3 first-release gates. This closeout does not declare Wave 2 complete.

## 2026-09-09 — D2 / issue #42 accepted implementation complete

- **Decision:** accepted and complete against the binding contract in
  [issue #1 comment 5562461427](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5562461427).
  The exact implementation head
  `8bdca978abcd4f53f0c94cb2f4562fb576e736b2` received
  [independent D2-R approval](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/50#issuecomment-5604709742)
  with no actionable findings. [PR #50](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/50)
  merged as `d805024764a8661815be3b2c3036695195b7075f`.
- **Supported provenance:** [CI run 34301118384, job 102308027982](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/34301118384/job/102308027982)
  passed on Ubuntu 22.04 x86-64 / LLVM-Clang 21.1.8 for build; the complete
  F0/A0/A2/B0/C1/D1/D2/E1/E1B/I1/I2/K1/L2/N2/N3K/P1/Q0/T1/T2/X1 matrix;
  smoke; and the reproducible benchmark. CI metadata names exact PR head
  `8bdca978abcd4f53f0c94cb2f4562fb576e736b2`; Actions checked synthetic merge
  `58ec324e9fc983f20bea9e0094eff1367dcbeff7`. Its tree and the actual merge tree
  are exactly `69eca436397cfe8a2ff9cd3e2c8f2764af2d4f35`; `git diff --exit-code`
  between those two commits passed. CI did not run the later-named merge commit.
- **Supported D2 evidence:** 25 deterministic reference/resource/scope tests;
  warning-clean native replay; sanitizers; two fresh deterministic strict AOTs;
  frozen Q0 fixtures; exact cursor, error, and admission matrices; smoke; and
  benchmark all passed. The optimized public loop retained exactly 4,521,984 arena
  bytes at both 1,024 and 8,192 batches. At both horizons native
  dataset/batch/borrow/D2-allocation counts were baseline `0/0/0/0`, open
  `1/0/0/2`, live `1/1/0/4`, released `1/0/0/2`, and post-close `0/0/0/0`;
  exact-read descriptors were baseline/peak/post-close `0/1/0` with final delta
  zero. One-shard/1,024-shard open retention was 31,616/88,904 bytes and packed
  retention was exactly 16,448/16,448 bytes. Process-wide advisory evidence was
  93,672 KiB/5 descriptors for the one-batch small corpus and 94,252 KiB/6
  descriptors for the 8,192-batch large corpus, a 580 KiB RSS delta.
- **Merged-main retest:** from exact merge `d805024764a8661815be3b2c3036695195b7075f`,
  `/usr/bin/bash -c 'ESHKOL_ALLOW_UNSUPPORTED_HOST=1 LLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config CC=/usr/bin/clang CXX=/usr/bin/clang++ make test-d2'`
  passed on the explicitly unsupported CachyOS / LLVM 22.1.6 compatibility host.
  It reproduced 25/25 tests, sanitizers, deterministic AOT, exact 4,521,984-byte
  arena equality, all lifecycle and native descriptor transitions above,
  one/many-shard 31,616/88,904-byte open and 16,448/16,448-byte packed retention,
  production isolation, and final `D2 PASS`. Local process-wide advisory values
  were 82,052 KiB/3 descriptors for the one-batch small corpus and 92,680 KiB/4
  descriptors for the 8,192-batch large corpus, a 10,628 KiB RSS delta; topology
  process peaks were 3/3 with native peak 1/1 and final delta zero.
- **Packaging:** the canonical aggregate remains one member, exact 18-source
  closure, 58 globals, and 52 package exports. The temporary test aggregate is
  isolated at 59/53, deterministic across two builds, and absent from production
  outputs; production objects and AOTs contain no test hook or Python/PyTorch
  runtime dependency. The unchanged raw `strings -a` heuristic passed the
  merged-main rerun. A prior unsupported LLVM 22 exact-head replay rendered
  instruction bytes as the false-positive text `=Py_`; independent source, symbol,
  link, and disassembly checks proved no dependency, and the supported LLVM 21 lane
  passed the identical scan. This layout-sensitive unsupported-host heuristic is a
  nonblocking test-harness hardening caveat, not runtime evidence.
- **Limitations / downstream:** D2 remains CPU-only with fixed dense i64/i64/bool
  storage, serialized and nonreentrant finite traversal, bounded-window rather than
  global shuffle, and no epoch, causal/document mask, alternate dtype/device,
  public C ABI, C2/O2/TR3 behavior, or power-loss claim. D1 trusted-directory,
  symlink, and concurrent-mutation limits remain. Deliberately retained/promoted
  caller shells remain caller-owned. The lexical `with-region` requirement is the
  accepted implementation and resource-gate discipline, but
  [issue #1 comment 5563425950](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5563425950)
  was proposed and nonbinding when this closeout was reviewed. This entry itself
  did not accept it; the later amended acceptance recorded below supersedes that
  pending status with explicit live-shell conditions.
- **Reference:** [issue #42](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/42);
  [PR #50](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/50).

## 2026-09-09 — Wave 2 merged-primitive acceptance closeout

- **Decision:** accepted and complete for the bounded A2, T2, L2, I2, and N2
  workstreams. Their earlier active/proposed/review entries remain historical and
  are superseded in status only.
- **Evidence:** [Wave 2 primitive acceptance](WAVE2_PRIMITIVES_ACCEPTANCE.md)
  records each exact-head independent approval, implementation merge, supported
  PR CI, and post-merge supported test run. A separate read-only Sol/high audit
  reconciled review comments, merge/head/CI tree identity, and actual workflow
  steps; it did not infer completion merely from merge status.
- **N2 qualification:** immediate post-merge run 34062447715 failed on fresh-oracle
  byte identity for an unproved cause. The later supported run 34301118384 passed
  all 31 pinned N2 oracle tests without skips, including fresh regeneration, and
  the full N2/integration/smoke/benchmark gates. Its tree equals merged main
  `d805024`; all 27 N2-owned paths equal the N2 implementation merge. Integration
  accepts this explicitly superseding retest, without erasing the earlier failure.
- **Scope:** documentation/status correction only; no API, ABI, numerical domain,
  lifetime, format, or required-gate changes. This does not close D2/O2/C2, mark
  all of Wave 2 complete, or claim a complete language model or training runtime.
- **Closeout repair:** the N3K predecessor manifest also pins the two A2/N2
  contract documents. Update only those two document digests for the reviewed
  acceptance-status text; every executable/header/fixture digest remains unchanged.
  The complete predecessor checksum gate is still required. The README's stale
  pending D2 wording is superseded by the same accepted live-shell amendment below.

## 2026-09-09 — D2 caller-region clarification accepted with amendment

- **Decision:** [amend and accept](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5608140148)
  proposal 5563425950. Native release does not reclaim caller-region Eshkol shells;
  bounded batch loops require one lexical next/use/release region.
- **Live-allocation condition:** authentic stale/idempotent-alias guarantees apply
  only while the shell allocation is live. Accessing an unretained shell after
  region exit has no safety or structured-error guarantee. Deliberately preserved
  live aliases remain caller-owned and accounted separately; a raw reference alone
  is not allocation retention. C2/TR3 must preserve these scopes.
- **Evidence:** final merged public-loop measurements are 4,521,984 arena bytes at
  both 1,024 and 8,192 batches, native lifecycle baseline return, and exact-read FD
  baseline/peak/post-close 0/1/0 with final delta zero. The older proposal's
  4,849,664-byte totals are historical measurements of another tree, not the final
  totals. No API, cursor, carrier, aggregate count, or native ABI changes.

## 2026-09-06 — D2 caller-region lifetime clarification (historical proposal)

- **Historical decision:** proposed and pending integration-owner disposition in
  [issue #1 comment 5563425950](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5563425950).
  This original proposal is superseded by the amended acceptance above; its
  measurements and original wording remain chronological evidence.
- **Proposed lifetime clarification:** replace references to GC-managed generation
  capabilities with caller-region-managed Eshkol capabilities. The pinned runtime
  has no tracing collector. `token-batch-release!` invalidates the live generation
  and ends the native borrow/carrier lifetime, freeing the exact native `17*N*T`
  carrier; it does not individually reclaim the batch/tensor closure, environment,
  or capability-vector objects allocated in the caller's Eshkol region. A
  long-running caller must put each `token-dataset-next-batch` through
  `token-batch-release!` interval in one lexical `with-region`. No batch or tensor
  shell may cross that boundary unless the caller deliberately retains or promotes
  it and accounts for the resulting caller-owned storage. C2/TR3 must preserve this
  lexical batch scope.
- **Authentication boundary:** D2 retains no batch/tensor shell and no
  per-generation authentication object or tombstone. The compiled aggregate
  registers the generated-code identities of exactly two fixed private shell
  constructors, dataset and batch, and native code retains only those two code
  addresses. This is a pinned, per-process private implementation ABI: it is not a
  public C ABI, serialized identity, portable closure-layout contract, or authority
  valid across a process or independently compiled aggregate.
- **Evidence:** optimized AOT without a lexical batch region retained 35,782,656
  arena bytes after 1,024 batches and 254,345,216 bytes after 8,192 batches. The same
  binary shape and admitted corpus, with one lexical batch `with-region`, retained
  exactly 4,849,664 bytes at both horizons, an exact zero-byte slope. The proposed
  review gate therefore compares the exact Eshkol arena counter at both horizons;
  native live counts must return to baseline and RSS remains separately reported and
  advisory.
- **Compatibility:** no public name/arity, carrier, cursor/format, shuffle, aggregate
  count, or native data ABI changes. `ESHKDCU1` remains version 1.0 and the aggregate
  remains exactly 58 globals/52 exports. Independent D2-R, supported CI, merge, and
  merge-head retest remain pending; ROADMAP stays `review`.

## 2026-09-06 — D2 / issue #42 accepted current-main contract

- **Decision:** accepted in
  [integration issue #1 comment 5562461427](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5562461427)
  and mirrored in
  [issue #42 comment 5562461448](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/42#issuecomment-5562461448).
  It accepts the superseding exact proposal and canonical/adversarial corrections in
  [comment 5557201280](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5557201280).
  Implementation is reconciled with merged main
  `231f9354f14389db15faac7820ef23dc038ae941`; independent approval and merge remain
  pending.
- **Public contract and A0 amendments:** the delta is exactly the ten existing A0
  dataset/batch operations plus unary `token-batch-release!`. Every exact authentic
  generation already issued by a dataset remains an idempotent release authority,
  including after a successor, seek, or close; all other stale access remains
  `invalid-state`. The three batch accessors return stable read-only state-backed
  opaque tensor identities, not newly
  owned clones. `token-dataset-cursor` returns newly owned detached mutable canonical
  bytevector storage, not an immutable registry object. The dataset config is the
  exact ten-key acyclic flat list specified in [D2_SHARD_LOADER.md](D2_SHARD_LOADER.md),
  not X1. Open normalizes and deep-copies it, retains neither caller carriers nor the
  tokenizer, rejects NUL paths, and validates every shard before publication.
- **Rows, carrier, and lifetime:** packed rows may cross D1 shards; unpacked rows do
  not, and no shard/document equivalence is claimed. Inputs and targets are the exact
  one-token shift, with zero only at false-mask positions and in unused final rows.
  The carrier is exactly two CPU dense `i64[N,T]` planes and one CPU dense one-byte
  `bool[N,T]` loss-mask plane, payload `17*N*T`. One dataset has one current native
  registry entry and at most one live batch. Authenticated shell copies alias that
  generation. Release invalidates the batch and three tensor identities before the
  fixed destruction tail, is idempotent for every exact authentic already-issued
  generation, and retains no per-generation carrier/tombstone. Scoped K1 views
  cannot escape; active borrow blocks release/close before mutation. Generation
  exhaustion never wraps.
- **Ordering and cursor:** consecutive bounded windows use descending Fisher-Yates
  with domain-separated SHA-256 u64-le draws and rejection sampling. The algorithm
  removes modulo bias within a window conditional on the digest stream but is not a
  global-uniform claim. `ESHKDCU1` version 1.0 is the accepted, exact `208+F`-byte
  C2-facing state. Seek validates physical structure, canonical encoding, version/
  features/algorithm, checksum, dataset/options identity, recomputed row count, and
  ordinal before one receiver commit. It defines no C2 container or epoch state.
- **Resources, errors, and packaging:** the semantic working-set admission is exactly
  `maximum_manifest_bytes + maximum_shard_bytes + 17*N*T + 8*min(B,M)`, plus
  separately measured fixed control/view/hash state. Checked exact ceilings admit;
  one-over rejects before the associated read/allocation. At most one manifest, one
  shard, one batch, and one shuffle window are retained; there is no whole-corpus or
  unbounded row/record table. The source-composed review aggregate extends the 47/41
  T2/I2 base to exactly 58 globals/52 package exports, composes shared trusted roots
  once, and does not link localized I2/T2/O2 aggregates. N2 adds no wrapper/count.
  Exact error categories and precedence are specified in the D2 document.
  The implementation routes D2 manifest/shard reads through the reviewed exact-range
  descriptor primitive, while preserving Eshkol-native parsing, SHA-256, and errors;
  it therefore retains no process-lifetime hosted-port entries. The private dataset
  control owns exactly `min(B,M)` checked i64 shuffle slots, with Eshkol retaining the
  accepted unbiased Fisher-Yates algorithm. This is an internal storage/lifetime
  decision only: it changes no public operation, carrier, cursor/format, aggregate
  count, or public ABI.
- **Dependencies / retest:** affected A0, D1, T1, T2, I2, N2, and Q0 gates; public
  compiled content/lifetime/resource/corruption evidence; deterministic AOT and
  sanitizers; full supported Ubuntu 22.04/LLVM 21 CI; and independent D2-R are
  required. ROADMAP may remain only `review` until approval, merge, merge-head
  retest, and acceptance-document follow-up.
- **Reference:** [issue #42](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/42);
  [PR #50](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/50).

## Schema

| Field | Content |
|---|---|
| Date | ISO 8601 date |
| Workstream / issue | Roadmap ID and tracking issue |
| Decision | `proposed`, `accepted`, `rejected`, or `superseded` |
| Contract | Stable command, path, API, ABI, or file-format surface |
| Evidence | Exact command/result, CI run, review, or pending reason |
| Dependencies / retest | Affected downstream workstreams and gates |
| Reference | Commit, PR, or issue URL |

## 2026-08-28 — F0 / issue #2

- **Decision:** accepted.
- **Contract:** Canonical entry points are `make toolchain`, `make configure`,
  `make build`, `make test`, and `make smoke`, invoked with `/usr/bin/bash`.
  Eshkol source lives under `src/eshkol_transformer/`; tests under `tests/`;
  developer tooling under `scripts/`; compatibility pins under `toolchain/`.
  The initial supported CI lane is Ubuntu 22.04 x86-64, LLVM/Clang 21.1.8,
  and Eshkol commit `90cbd7130f47b8184bcc77b8d5c1b0026da980de`.
- **Evidence:** `/usr/bin/bash -c 'make clean && make configure && make build &&
  make test && make smoke'` passed locally as an explicitly unsupported CachyOS
  x86-64 / LLVM 22.1.6 compatibility probe. The test runner performed two fresh AOT
  compilations and executions with byte-identical `eshkol-transformer-smoke:v1\n`
  output and an actionable missing-toolchain negative. The supported Ubuntu 22.04
  x86-64 / LLVM-Clang 21.1.8 lane passed toolchain, fresh configure, build, test, and
  smoke in [CI run 33220609386](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33220609386).
- **Dependencies / retest:** B0 may consume these commands after F0 merges.
- **Reference:** issue #2; [PR #8](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/8);
  merge commit `0f95dc732405e1de24256ecca27906f9149990e2`.

## 2026-08-28 — B0 / issue #6

- **Decision:** accepted.
- **Contract:** `make benchmark` measures the F0 native smoke artifact from the
  canonical `tsotchke/eshkol@90cbd7130f47b8184bcc77b8d5c1b0026da980de`
  toolchain. `benchmarks/smoke_v1.json` is the canonical version-1 checksummed
  definition. Generated reports use the version-1 checksummed stable/volatile schema
  documented in `docs/BENCHMARK_FORMAT.md`. B0 supports only direct Linux host-CPU
  process execution, `CLOCK_MONOTONIC` elapsed nanoseconds, `pidfd` completion
  notification, and `wait4` peak RSS KiB.
- **Evidence:** 41 deterministic B0 format, failure-path, runner, and native-launcher
  tests passed. The full F0 configure/build/test/smoke chain and `make benchmark`
  passed locally against the exact upstream pin as an explicitly unsupported CachyOS
  x86-64 / LLVM-Clang 22.1.6 compatibility probe. The generated report validated and
  labeled itself `compatibility-only`. Supported Ubuntu 22.04 / LLVM 21.1.8
  [CI run 33223213742](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33223213742)
  passed the pinned toolchain build, F0/A0/B0 tests, native smoke, generated report,
  and report verification. Independent final review found no material blocker.
- **Dependencies / retest:** AMD4 and PA4 may consume the schema only after B0
  integration review. They must add direct device evidence and may not reinterpret
  host RSS as device memory or this smoke result as an acceleration baseline.
- **Reference:** [issue #6](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/6);
  [PR #14](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/14);
  merge commit `56d2c7ff6052e58f883b7a5be0d6e17e8f6e460e`.

## 2026-08-28 — A0 / issue #3

- **Decision:** accepted.
- **Contract:** The first-release Eshkol API fixes public names and arities, tensor
  shapes/layouts, dtypes/devices, ownership and mutation, structured errors,
  capability requests, logical/canonical tied-parameter paths, deterministic trainer
  accumulation across finite-dataset epoch boundaries, generation semantics, safe
  persistence limits, and independent API/ABI/format version domains. It commits to
  no serialized byte layout or native ABI. Runtime capabilities remain unverified
  unless separately proved.
- **Evidence:** `/usr/bin/bash -c 'make test-a0'` and
  `/usr/bin/bash -c 'make test'` passed against the F0-provenance-verified canonical
  `tsotchke/eshkol` commit
  `90cbd7130f47b8184bcc77b8d5c1b0026da980de` as an explicitly unsupported CachyOS
  x86-64 / LLVM-Clang 22.1.6 compatibility run. The A0 harness strict-compiled
  aggregate declarations twice with byte-identical objects, repeated exact negative
  diagnostics, repeated executable output, and per-invocation 60-second bounds.
  Supported Ubuntu 22.04 / LLVM-Clang 21.1.8
  [CI run 33222031424](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33222031424)
  passed the integrated F0/A0 gates after independent contract review.
- **Dependencies / retest:** P1, T1, D1, K1, C1, X1, and later model/training/
  generation workstreams must implement these contracts or propose a versioned
  change through issue #1. Supported Ubuntu 22.04 / LLVM-Clang 21.1.8 CI must pass
  before acceptance.
- **Reference:** issue #3; [PR #13](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/13);
  merge commit `84fbbbc76329fbe286923e943ff90b29372bba29`.

## 2026-08-28 — Q0 / issue #5

- **Decision:** accepted.
- **Contract:** Oracle fixtures are canonical, versioned, checksummed, non-executable
  JSON with strict tensor roles, dtypes, devices, shapes, values, and tolerance
  policies. PyTorch is isolated to the development reference generator and is not in
  the Eshkol runtime path.
- **Evidence:** `/usr/bin/bash -c 'scripts/test-q0.sh'` passed 23 tests, including
  malformed/corrupt fixture rejection, strict runtime scalar policies, finite
  differences, byte-identical fresh-process generation, Python isolation, and a real
  AOT-compiled Eshkol scalar-add parity check against the frozen reference.
- **Dependencies / retest:** All numerical Wave 1+ workstreams must use the fixture
  format and add operation-specific forward/gradient evidence without importing
  Python into production code.
- **Reference:** issue #5; [PR #10](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/10);
  merge commit `e4d945b3a9719d9ae2f9f624220ba6d3106517ce`.

## 2026-08-28 — R0 / issue #4

- **Decision:** accepted with explicitly bounded evidence.
- **Contract:** `probes/r0/run.sh` accepts only the canonical
  `tsotchke/eshkol@90cbd7130f47b8184bcc77b8d5c1b0026da980de` source and validated
  F0 provenance. The audit distinguishes observed support from
  `untested-with-reason` and records the Eshkol-core versus versioned-K1 disposition
  for each gap. It authorizes no fallback.
- **Evidence:** The representative `tensor_core` compatibility-lane run completed six
  AOT/JIT phases with zero failures, byte-identical repeated output, and retained
  compiler/source provenance. Independent review passed. The full suite, supported
  Ubuntu/LLVM 21 lane, sanitizers, and GPU remain explicitly untested rather than
  inferred.
- **Dependencies / retest:** P1 and K1 may use only observed rows; every untested row
  remains unsupported until new executable evidence is reviewed.
- **Reference:** issue #4; [PR #12](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/12);
  merge commit `f843f4ce826b56cca14bedfe8fc9f81155ecf1cb`.

## 2026-08-29 — K1 / issue #21

- **Decision:** accepted after independent re-review and exact-head supported CI.
- **Contract:** Native-kernel ABI 1.0 uses the major-only provider symbol
  `eshkol_transformer_kernel_provider_v1`, fixed x86-64 v1.0 structure prefixes,
  compatible-minor discovery, exact-span stride tables for every repeated
  size-tagged descriptor, explicit process-local resolver registration, deep
  immutable capability snapshots, strict shape/storage/ownership checks, two-phase
  failure-atomic dispatch, ABI-local errors, and canonical process-local JSON reports.
  No filesystem or environment library discovery exists. The provider-free report
  deterministically contains every required A0 capability as `unverified`, grounded
  only in R0 evidence merged through PR #15. K1 ships no numerical provider and
  claims no dtype, CPU backend, accelerator, GPU, operation, gradient, or determinism
  capability. A0 Eshkol names/arities are unchanged.
- **Evidence:** `/usr/bin/bash -c 'make test-k1'` passed on the explicitly unsupported
  CachyOS x86-64 / LLVM-Clang 22.1.6 compatibility lane against clean canonical
  `tsotchke/eshkol@90cbd7130f47b8184bcc77b8d5c1b0026da980de` and compiler
  `1.3.4-evolve`. The gate passed 596 C ABI, compatible-minor, discovery,
  exact-request, ownership, malformed-call, failure-atomicity, canonical JSON, and
  unsupported-path checks; C/C++ warning-clean compilation; ASan/UBSan; byte-identical
  `C`/`C.UTF-8` reports; strict JSON parse/canonical round trip; and two Eshkol AOT
  link/runs of the fixed-width ABI version probe. The baseline report SHA-256 is
  `7e14cc845902b6a37f9946a163d355085ea704c043863f7e37481b4ab0deec59`.
  The focused gate consumes the archive produced by `make build`, verifies its
  one-member shape, checks the same-rank canonical-order golden, and uses the same
  builder for its sanitized archive. The repository-wide `make test` and `make smoke`
  gates also passed with a writable explicit Eshkol JIT cache. Independent re-review
  approved exact head `b64f7c8a0988c98d190ea4b87fbf0eb77efe7ed6` with no findings after
  confirming the stride/span and persistent-build-artifact corrections. Supported
  Ubuntu 22.04 x86-64 / LLVM-Clang 21.1.8
  [CI run 33229145836](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33229145836)
  passed at that exact head. Integration reran focused K1, ASan/UBSan, full
  `make test`, `make build`, and `make smoke` after merge.
- **Dependencies / retest:** N2, A2, L2, MP4, AMD4, and DIST5 may consume only
  separately verified provider entries; none may infer support from ABI presence.
  E1 issue #23 owns `transformer.error_internal`; any Eshkol-facing K1 adapter must
  rebase onto reviewed E1 and preserve ABI categories and causes rather than adding a
  duplicate error representation. LeakSanitizer remains optional where the executor
  supports it.
- **Reference:** [issue #21](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/21);
  [integration issue #1](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1);
  [PR #25](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/25); merge commit
  `0e0573b2ceb1c1b5a2bbb46786c13578490e2ae9`.

## 2026-08-29 — E1 / issue #23

- **Decision:** accepted after independent security/contract review, pinned-runtime
  feasibility review, and exact-head supported CI.
- **Contract:** `transformer.error_internal` is the single construction and raising
  boundary.  It exports the six unchanged A0 unary accessors plus
  `transformer-error-make` (5), `transformer-error-raise` (5), and
  `transformer-error-wrap-foreign` (8).  Errors are native condition identity tokens
  with constant generic text and no irritants; copied metadata lives in a lexical,
  append-only process registry.  Only registered identities satisfy the predicate,
  so byte-identical forged native conditions fail.  Details are bounded, acyclic,
  canonical symbol-keyed data-only
  alists; messages/details/causes are copied at ingress and egress.  Foreign wrappers
  preserve source domain, exact-integer-or-symbol code, and source message while the
  caller supplies the mapped existing A0 category.  Public facades do not export the
  three internal helpers.  Exact grammar, bounds, ownership, and bootstrap behavior
  are documented in [ERROR_CONTRACT.md](ERROR_CONTRACT.md).
- **Evidence:** `/usr/bin/bash -c 'make test-e1'` passed 112 runtime assertions, two
  byte-identical AOT executions, two byte-identical JIT executions, AOT/JIT output
  parity, byte-identical fresh compile objects, production depfile proof, fresh-cache
  first-class internal binding execution, public accessor execution, stable
  wrong-arity rejection, and exact source checks for the declared six-op public and
  nine-op internal surfaces.  Adversarial cases cover
  forged same-message conditions, procedure non-invocation, repeated detached
  snapshots, distinct cause identities, fresh/interned same-spelling key collisions,
  tagged complex rejection with its observed pinned-runtime predicate matrix,
  malformed bridge shapes/opcodes, and 256 retained registry identities. The public
  facade also proves that mutating its returned message carrier does not affect a
  later snapshot. The
  local run used the exact Eshkol commit/compiler pin
  but is an explicitly unsupported CachyOS x86-64 / LLVM-Clang 22.1.6 compatibility
  probe. Supported Ubuntu 22.04 / LLVM-Clang 21.1.8
  [CI run 33267293761](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33267293761)
  passed at exact rebased head `a25dada6ac5dcdd39e8f9ba3a5451c022fdeaea7`.
  Integration then reran merged-main A0, K1, E1, full `make test`, and `make smoke`.
- **Pinned-runtime limitation:** `define-record-type` is a mutable vector and the
  earlier irritant-held seal was rejected after an adversarial forge succeeded.
  Pinned `provide` declarations are informational, so one validated, unsupported
  `e1-internal-dispatch` core bridge remains technically name-reachable although it
  is absent from the error-public/public/capabilities provide lists and every public
  API/SemVer surface.  The strong registry has
  process/arena lifetime, linear lookup, monotonic memory cost, no serialization,
  and no verified concurrency or thread-safety behavior.  This identity workaround
  is accepted for the pin and must be retested when upstream gains module opacity or
  an immutable opaque record facility.
- **A0 snapshot clarification:** on the pinned runtime, standard list and string
  carriers are not physically frozen. The A0 term `immutable snapshot` commits to
  immutable observable source state and non-aliasing: accessors return fresh
  deep-owned carriers whose caller mutation cannot affect the source or a later
  snapshot. It does not claim physical immutability for those detached carriers.
- **Dependencies / retest:** P1, T1, D1, K1, and X1 must import this internal boundary
  and remove subsystem-local structured-error representations.  Their explicit
  source-status mappings remain workstream-owned and require review when they create
  ABI or format commitments.  Rerun A0 and E1 gates after each facade integration.
- **Reference:** [issue #23](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/23);
  [PR #27](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/27); merge commit
  `80371a1f06cc71c44c2d940a57b294a89174a1f0`.

## 2026-08-30 — I1 / issue #24

- **Decision:** accepted after independent exact-head re-review, supported CI, and
  merged-main integration gates.
- **Contract:** Separate native I1 ABI 1.0 provides a deep-owned exact signed-`i64`
  dense row-major zero-offset CPU container for ranks 0 through 64 in the
  explicit-link archive `build/i1/libeshkol_transformer_i64.a`. Rank 0 has one
  element; a zero extent at positive rank canonically has null data, zero bytes,
  and all-zero byte strides. Checked products govern nonempty element and byte
  spans. One tracked borrow may be active; it blocks owner mutation and destruction
  but permits reads and direct external mutation through the borrowed K1 view.
  Pointer-to-pointer outputs require null contained slots, preserve both null and
  nonnull slots on failure, and publish only on success. Copy buffers, writable
  scalar/handle outputs, and error records are checked before mutation against a
  process-local registry of every live I1 control, shape, stride, data, borrow, and
  descriptor allocation. A nonnull shape declares a logical span of `rank`
  contiguous `uint64_t` elements even for an invalid rank; its checked exact span
  participates in alias validation without memory access. An unrepresentable byte
  span or exclusive end causes conservative no-write rejection when an error
  record is supplied. An aliased error record is rejected without diagnostic mutation
  because no safe error destination exists. Direct use of an authorized
  live borrowed K1 view remains the explicit owned-storage mutation exception.
  The explicit process-local provider accessor supplies exactly one deterministic
  verified K1 row: `tensor.i64` / `storage.copy` / `cpu` / `i64`, for rank 0 and
  rank 1 extent `0..2305843009213693951`. It performs no allocation, cast, transfer,
  fallback, or partial-success path. All I1 ABI entry points use the `_v1` symbol
  suffix. K1 sources and A0 names/arities are unchanged;
  the compiled-Eshkol facade is test-only and maps native failures through E1.
- **Evidence:** On the explicitly unsupported CachyOS x86-64 / LLVM-Clang 22.1.6
  compatibility lane, `/usr/bin/bash -c 'make test && make build && make smoke'`
  passed against a clean canonical
  `tsotchke/eshkol@90cbd7130f47b8184bcc77b8d5c1b0026da980de` checkout and compiler
  `1.3.4-evolve`. The focused I1 gate passed 764 normal exact-i64, shape/layout,
  ownership, alias, failure-atomic, malformed-call, and K1 dispatch checks twice
  with identical output; 2,476 allocation-hook and ASan/UBSan checks covering live
  opaque handles, same- and cross-object owned regions, pointer-slot preservation,
  success/failure error aliases, and invalid-rank checked shape spans without
  extent reads; a warning-clean
  C++17 consumer; and two canonical-pin AOT executions with 59 checks each. AOT
  output SHA-256 was
  `de78f6cb1b0ea6ffd169a4dfc1d7500f1f3f1db69dd113f8fd737a978c92d217`;
  compiler SHA-256 was
  `caa295b19a6e9388963aa0def99dade63656d2dcbffccad421bd1daaa1db3750`.
  The repository-wide run also passed the F0 smoke/missing-toolchain probes, 41 B0
  checks, A0, 596-check K1, 112-check E1, build, and smoke. Independent final
  re-review approved exact head
  `406b06928cf3089ce690dafad5c36a5189a3f940` with no remaining finding after
  reproducing the invalid-rank shape-span and all earlier alias/failure-atomicity
  cases. Supported Ubuntu 22.04 x86-64 / LLVM-Clang 21.1.8
  [CI run 33322029018](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33322029018)
  passed at that exact head. Integration then reran `make configure build test-i1`
  and repository-wide `make test build smoke` from merged main; both passed.
- **Dependencies / retest:** T1 may use only separately reviewed explicit native
  lifetime and E1 wrappers; I1 supplies no tokenizer behavior or production Eshkol
  finalizer. D2 may use rank-2 owned storage only as unverified container behavior;
  it may not infer a K1 capability, mask, packing, cursor, or resume contract.
  Re-test the exact AOT carrier/range checks if the pinned compiler changes.
- **Reference:** [issue #24](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/24);
  [PR #32](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/32); merge commit
  `53db2a19725bf4a4fbdb60f4abeb6bb521602b1f`.

## 2026-08-29 — E1B / issue #33

- **Decision:** accepted after independent exact-head review, supported CI, merge,
  and merged-main focused/full regression gates.
- **Contract:** Public package stubs import neither `transformer.error_internal` nor
  `transformer.error_core`. The existing `transformer.error_public` facade and all
  E1B package stubs use the same artifact-backed six accessors, so one completed AOT
  artifact owns the sole E1 registry in either import order. A trusted build-only
  Eshkol seam accepts exactly category, operation, message, details, and cause,
  calls only E1's validated raise path, and never returns. Package-specific native
  wrappers expose only reviewed narrow operations plus the six unchanged unary A0
  accessors. The generic seam, constructors, dispatcher, initialization helpers,
  compiler companions, and raw representation remain `STB_LOCAL` before arbitrary
  application source is compiled. Native code performs fixed box validation,
  transport, initialization, and calls only; E1 retains all value semantics. The
  final global definitions and runtime undefined references must each match their
  reviewed C-sorted build-input allowlists byte-for-byte before publication.
- **Evidence:** `/usr/bin/bash -c 'make test-e1b'` passed 35 public-package AOT
  semantic checks plus byte-identical repeated trusted artifacts, public objects,
  executables, link maps, and outputs. Cold-cache strict objects and AOT applications
  import the real `transformer.error_public` facade with consumer A/B in both orders,
  recognize the same errors and detached causes through all six accessors, and prove
  public depfiles exclude both source registry modules. The final artifact exactly
  matches one repository-owned 80-name runtime-undefined allowlist on both supported
  Clang 21.1.8 and compatibility Clang 22.1.6. Explicit
  `-fstack-protector-all` on the trusted native bridge makes `__stack_chk_fail`
  deterministic; the final object is still compared byte-for-byte, never against an
  optional subset. Unsorted public-export input is rejected rather than normalized.
  Distinct unlisted
  function, data, TLS, and weak references have verified ELF bindings/relocations and
  each fail before object/evidence publication or later application satisfaction.
  Exact five-argument diagnostics, malformed category/operation/message/details/
  cause, ownership, provide/export allowlists, `nm`, archive index, `readelf`, dynamic
  symbols, strings, link map, and guessed privileged-name negatives remain green.
  After rebasing onto accepted I1 main `5914b77`, focused E1 112 and A0 passed; the
  full `make test` built K1 and I1 and passed F0/B0/A0/K1/E1/E1B/I1, including
  sanitizers and 41 Python harness checks; separate `make build` and `make smoke`
  passed. This is canonical-pin evidence on the explicitly unsupported CachyOS /
  LLVM-Clang 22 compatibility lane. Independent registry/identity,
  packaging/linker, adversarial-consumer, and evidence reviews informed and
  reproduced the repair. Supported Ubuntu 22.04 / LLVM-Clang 21.1.8
  [CI run 33325565974](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33325565974)
  passed on code head `611da5a`: fresh toolchain/configure/build, F0/B0/A0/K1,
  E1 112, E1B 35, I1, 41 Python harness checks, sanitizers, smoke, and the
  reproducible smoke benchmark.
- **Limitations:** canonical-pin/x86-64 AOT only; one artifact and E1 registry per
  process; no JIT/bitcode publication; no verified concurrency; safe-only public
  source aliases remain technically reachable because pinned `provide` is
  informational. Malicious native-object injection into the trusted partial link is
  outside scope, while arbitrary compiled Eshkol linked afterward is in scope.
- **Dependencies / retest:** D1 and X1 must expose only reviewed package-specific
  wrappers in the same combined registry-owning artifact, add a reviewed
  repository-owned exact undefined-symbol policy for their larger trusted closure,
  and repeat public-
  closure, mixed-facade import-order, malformed-value, symbol/link, and full-suite
  gates. E1B changes no K1/I1 contract and adds no public persistence format.
- **Reference:** [issue #33](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/33);
  [integration issue #1](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1);
  [PR #34](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/34); merge commit
  `60c9afa182115bc3ebc9bad321e366b8b3979ae6`. Independent review approved exact
  head `1e2f2c9b8d895c43c88ecb2b770e6612fb57c767`; merged-main `make test-e1b`
  and `make test build smoke` passed on the documented local compatibility lane.

## 2026-08-28 — X1 / issue #20

- **Decision:** accepted after independent exact-head review, supported CI, merge,
  and merged-main focused/full regression gates.
- **Contract:** X1 uses the strict, flat, non-executable version-1 JSON source schema
  and the independently versioned `eshkol-resolved-run` `[1,0]` canonical manifest
  documented in [CONFIG_FORMAT.md](CONFIG_FORMAT.md). Every explicitly present source
  leaf is admitted by type, range, enum, and policy before override overlay, so an
  override cannot sanitize invalid input. Resolution precedence for admitted values
  is defaults, source, unique explicit overrides, then absent-field derivation and
  full validation. Canonical output contains all resolved leaves and per-leaf provenance;
  provenance is part of identity. `config-fingerprint` is
  `sha256:eshkol-config-json-v1:<lowerhex>` over the exact canonical bytes including
  the final LF. The manifest declares SHA-256 coverage but embeds no circular digest.
  Version 1 permits no code, includes, environment expansion, secrets, runtime
  evidence, hidden dtype/device/precision/CPU/scalar/Python fallback, or unsupported
  version inference.
- **Evidence:** on the unsupported CachyOS / LLVM-Clang 22.1.6 compatibility lane,
  the pinned Eshkol `90cbd7130f47b8184bcc77b8d5c1b0026da980de` compiler
  `v1.3.4-evolve` passes 111 native parse/override/source-admission/validation/
  opacity/canonical checks and 11 Python reference/isolation checks. The focused
  gate builds the combined artifact twice byte-identically, compiles strict source
  objects and AOT applications from fresh caches twice, exercises the E1 public
  facade and X1 in both import orders against one registry, checks all six A0
  arities, and rejects source and guessed-link access to constructors, dispatcher,
  private raise seam, representation helpers, and private X1 entries. The completed
  object has exactly 12 globals (six E1 accessors plus six X1 wrappers) and exactly
  matches its reviewed C-sorted 95-name runtime-undefined manifest; all privileged
  and implementation symbols are local. Golden canonical bytes, metadata/payload/
  provenance mutations, whole-document SHA-256 including final LF, hostile
  environment/CWD determinism, and absence of Python/PyTorch runtime dependencies
  pass. Focused A0, E1 (112), E1B (35), I1, and repository-wide `make test`, followed
  by separate `make build` and `make smoke`, pass after rebasing on accepted E1B main.
  Independent parser/canonicalization, E1B packaging/leakage, and test-evidence
  reviews report no blocker. Supported Ubuntu 22.04 / LLVM-Clang 21.1.8
  [CI run 33329897122](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33329897122)
  passed at exact reviewed head `246afd8ad6ad663082f7efeec32ec554421e909c`.
- **Limitations:** X1 is canonical-pin/x86-64 AOT-only and publishes no JIT/bitcode
  artifact. One combined artifact/registry is supported per process; concurrency is
  unverified. The pinned compiler leaves six safe-only source aliases reachable;
  each conveys only its corresponding public configuration operation. Malicious
  native injection into the trusted partial link is outside scope.
- **Dependencies / retest:** T1 must consume the validated configuration contract
  before `tokenizer-byte`; C2 may embed the exact manifest and fingerprint only after
  X1 integration. CLI3 may expose source parsing and overrides only after the same
  review. Rerun T1, C2, and CLI3 gates after any X1 contract change. Any field,
  default, version, canonical-byte, provenance, or fingerprint change requires issue
  #1 coordination and affected downstream retests.
- **Reference:** [issue #20](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/20);
  [integration issue #1](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1);
  [implementation PR #30](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/30);
  merge commit `18276b1284ad5e69bee6de91f3f00096e99e4ca5`. Merged-main
  `make test-x1` and `make test build smoke` passed on the documented local
  compatibility lane.

## 2026-08-29 — P1 / issue #22

- **Decision:** accepted after independent exact-head review, supported CI, merge,
  and merged-main focused regression gates.
- **Contract:** `transformer.module` retains exactly the 17 merged A0 public names and
  arities. Nested module, parameter, and buffer paths use deterministic UTF-8 byte
  lexical order; tied parameter paths retain destination identity while snapshots
  contain independent value-equal entries and canonical lexical alias groups. Logical
  state-dictionary schema 1.0 is an inert, non-executable contract independent of C1
  byte/container versions and contains only the provider identity
  `(transformer-tensor-provider 1 0 provider-id)`. Metadata is redundant validation
  data and never overrides tensor observations. The installed generated root imports
  only `transformer.error_consumer`; the never-installed trusted root imports only
  E1B's fixed five-value private raise seam. One prelocalized package object owns the
  sole E1 registry and exports exactly the 17 P1 operations plus six A0 accessors;
  every E1B seam, P1 thunk, and native identity definition is localized. The normal
  build's alternative native identity archive exposes four read-only observations
  only; its private replacement adds hidden fixed-arity identity/admission/binding
  cleanup operations and is never installed or linked beside either public product.
  P1's wider undefined-symbol policy is admitted before mutation only for the exact
  repository-owned trusted-root/bridge/rename/export tuple; copied roots, copied or
  mismatched bridge/native inputs, partial tuples, and environment overrides reject.
  Native code performs no schema, tensor, commit,
  serialization, or capability logic. Every parameter, tied-parameter, buffer, and
  child name—and every caller or generated path segment—shares one inclusive
  1..65536 encoded UTF-8 byte domain. Admission rejects one-over input before
  provider lookup/callback, allocation, identity/device work, or topology mutation;
  prebuilt attachment applies the same gate to its new edge. A process-local internal provider-v1 boundary
  defaults to no provider. Trusted C1/I1 callers must explicitly bind an
  already-admitted provider with the exact identity before load; serialized metadata
  never selects executable code and snapshots never serialize a binding. Admission,
  binding, exact path/kind/metadata/alias/storage validation, whole-batch preparation,
  and the no-recoverable-branch commit protocol are documented in
  [P1_MODULE_STATE.md](P1_MODULE_STATE.md). Construction admits at most 4096 modules,
  4096 state entries, 64 child edges, and 64 generated path segments; leaf and
  prebuilt-subtree registration validate prospective cached counts, shifted height,
  and shifted leaf span before any callback or mutation. Observation precomputes a
  bounded iterative finalization plan, leaving only finite vector writes after its
  commit boundary, and subtree storage checks compare unique authoritative handles
  so tied aliases cannot multiply provider-callback work. Provider admission retains an immutable
  bridge-owned identity/callback-token snapshot, uses exact 0..127-byte provider IDs,
  and revokes every unpublished identity on failure. No checkpoint bytes, checksums, encoding,
  magic, filesystem I/O, tensor backend, dtype implementation, or capability claim is
  added.
- **Evidence:** `/usr/bin/bash -c 'make test-p1'`, `/usr/bin/bash -c 'make test'`,
  `/usr/bin/bash -c 'make build'`, and `/usr/bin/bash -c 'make smoke'` passed against
  canonical `tsotchke/eshkol@90cbd7130f47b8184bcc77b8d5c1b0026da980de` and compiler
  `1.3.4-evolve` on the explicitly unsupported CachyOS x86-64 / LLVM-Clang 22.1.6
  compatibility lane. The focused gate passed 178 structural/ownership/binding/
  validation/atomicity assertions, 118 native token/security/ABI checks, 57 native
  allocation/entropy/partial-admission failpoint checks, 8 cross-role translation-
  unit identity checks, and 102 trusted test-only registry/error-mapping/topology/name-bound
  checks. The latter
  drive automatic partial-callback cleanup through an injected failure and prove exact
  live/tombstone deltas plus unchanged registry counts across failed and repeated strict
  loads; exact and one-over 64-segment/depth and 4096-entry/node fixtures additionally
  prove generated lookup, full tied/buffer state round-trip, atomic subtree/leaf
  rejection, bounded finalization, and exact/one-over 65536-byte multibyte name/path
  admission across every registration family, lookup, snapshot, and strict load.
  Their three hidden hooks exist only in a
  temporary non-production archive.
  Native failpoint errors raise directly through the same E1 registry with bounded
  canonical data-only status details and cause `#f`, even after cleanup overwrites the
  native context; an injected impossible status/code pair maps to `internal` and
  cleans up without adding a production hook. The gate also covers ASan/UBSan,
  both public import orders, and repeated
  strict AOT compilation and execution; byte-identical package objects, manifests,
  evidence, application objects, executables, outputs, and AOT diagnostics; exact
  source/object/AOT negative/no-artifact checks; and depfile/`nm`/`readelf`/`strings`
  production-isolation checks. Fresh-cache negatives cover every internal Eshkol name;
  all 25 private native functions remain unresolved against the completed package and
  resolve only against the deliberate non-installed trusted positive control.
  Adversarial cases include unbound/wrong-id/conflicting rebind,
  unknown/unverified provider, raised,
  substituted-vector, allocation-probe, and reentrant admission, later-entry prepare
  failure with unchanged destinations, conflicting aliases, duplicates, malformed
  schema, cyclic caller lists/composites, unsupported versions, and shape/dtype/device
  mismatches. Independent native-security, ABI, packaging/leakage, cyclic-validation,
  atomicity, evidence, and P1-contract reviews found no blocker. Supported Ubuntu
  22.04 x86-64 / LLVM-Clang 21.1.8
  [CI run 33352609244](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33352609244)
  passed at exact reviewed head `3b7ad986aba0c5dda860bc36a74aeee55c511878`.
  The actual merged-main commit also passed `make test-p1` with all 178 structural,
  183 native, and 102 registry/atomicity checks.
- **Dependencies / retest:** C1 may consume this merged logical contract and must
  treat itself, with I1, as a trusted explicit binder rather than perform
  metadata-driven provider lookup. The test-only provider proves only P1 control-flow
  and ownership semantics. I1/K1 and later real f32 support must repeat round-trip,
  mutation, storage-lifetime, device, and commit-infallibility tests before any
  numerical, physical tensor, backend, dtype, device, or module-capability claim.
- **Reference:** [issue #22](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/22);
  [integration issue #1](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1);
  [PR #31](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/31);
  [final independent review](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/31#issuecomment-5473456290);
  merge commit `b68ec9f9a64583800cfa888da2916f6ea99134b4`.

## 2026-08-28 — D1 / issue #18

- **Decision:** accepted after independent exact-head review, supported CI, merge,
  and an actual merged-main focused D1 rerun.
- **Contract:** `(require transformer.data)` exports the versioned, checksummed,
  manifest-last token-corpus writer, strict validator, identity-backed
  observationally immutable summary, and six summary accessors documented in
  `docs/PUBLIC_API_CONTRACT.md`. Summary metadata is validated, deep-owned, and held
  only in a private lexical identity registry; mutable or forged receivers are not
  summaries, and no constructor escapes. The corpus uses
  `manifest.etm`, canonical `shard-%016d.ets` names, distinct eight-byte magic values,
  little-endian signed-`i64` token semantics, SHA-256 trailers, opaque canonical UTF-8
  tokenizer fingerprints bounded to 1..192 bytes, repeated identity/count metadata,
  and exact v1.0 rejection rules in `docs/TOKEN_SHARD_FORMAT.md`. The fingerprint is
  stored and compared byte-for-byte; D1 does not parse or assume T1's concurrently
  proposed lexical form. A narrow native boundary exclusively creates each
  deterministic temporary, loops through partial writes, checks every byte and
  `close(2)`, and permits rename only on success; production contains no fault
  injection or fallback. The manifest rename is the publication commit point, not a
  whole-directory atomicity, `fsync`, or power-loss durability claim. The pinned
  compiler's supported D1 packaging is one combined, separately compiled E1B/D1
  object: its installed facade provides exactly the eight accepted procedures, the
  shared E1 registry and six accessors live in that object, and the fixed raise seam,
  private Eshkol implementation, and checked byte/write/close primitive are localized.
- **Evidence:** Historical evidence before the current D1-R corrections: after
  rebasing onto E1 merge `80371a1`, the local compatibility-only
  CachyOS x86-64 / LLVM-Clang 22.1.6 run passed `/usr/bin/bash -c 'make test-d1'`:
  byte-identical strict compilation with a production-E1 depfile, native SHA-256 and
  binary I/O/rename/lock/UTF-8 reachability under an unusable runtime `PATH`, checked
  arithmetic and production-E1 mapping/identity probes, and 14 deterministic/
  reference/corruption/boundary groups. The publication cleanup test passed five
  additional fresh runs. A0 passed; E1 passed 112 checks; Q0 passed 23 tests; the
  repository-wide `make test`, explicit `make build`, `make smoke`, and diff hygiene
  gates passed, including 41 B0 tests and the integrated K1 gate. All Eshkol evidence
  used provenance-verified canonical
  `tsotchke/eshkol@90cbd7130f47b8184bcc77b8d5c1b0026da980de`, compiler
  `v1.3.4-evolve`. Supported Ubuntu 22.04 / LLVM-Clang 21.1.8
  [CI run 33269298912](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33269298912)
  passed build, integrated tests, smoke, and the reproducible benchmark at obsolete
  PR head `58d3a5d`. D1-R subsequently requested three high-priority corrections at
  exact head `52cb854e3e76e9b834ce76631b0acf6c5328553d`: checked write/close status,
  nonforgeable observational summary immutability, and removal of named E1
  constructor reachability from `(require transformer.data)`. The candidate is
  rebased onto accepted K1/E1 documentation main commit
  `8ab5e5cb00a07d8bff6ffc7c52b72cbae2d6d832` and implements those contract changes.
  On the provenance-verified canonical pin, the unsupported CachyOS/LLVM-Clang 22
  compatibility run passed `make test-d1` with 16 format/native-I/O groups plus the
  compiled SHA-256, primitive, arithmetic, E1, summary-opacity, and fresh-cache AOT
  reachability probes. It also passed `make test-a0 test-e1` (including 112 E1
  checks), the standalone 23-test Q0 gate, full `make test` (including 41 B0 tests
  and K1 sanitizers), explicit `make build`, `make smoke`, shell syntax, and diff
  hygiene. Independent native-I/O/security, summary-opacity, public-boundary, and
  documentation reviews found no remaining actionable issue. Supported Ubuntu
  22.04 / LLVM-Clang 21.1.8
  [CI run 33273402656](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33273402656)
  passed build, integrated tests, smoke, and the reproducible benchmark at exact
  implementation head `aa3b9f2a83eff57af7a4f45e2e079f6a284ebd4b`. Any later
  documentation-only head requires its own exact-head confirmation before re-review.
  The subsequent D1-F repair rebased the complete candidate onto current main after
  accepted I1 integration and changed the normal build/test path to consume the same
  precompiled Eshkol facade. Fresh-cache source/object/AOT negatives now cover
  direct calls, first-class bindings, wrong arities, the former ninth FFI alias, and
  current write/publish/implementation/list/SHA helper families. Depfile, declared
  provide, `nm`, `readelf`, `strings`, and crafted relocatable-link checks prove those
  bindings and guessed native aliases are unavailable. The post-rebase local
  compatibility lane passed the 16-group D1 format/native-I/O suite and all compiled
  primitive, arithmetic, E1-mapping, summary-opacity, cleanup/no-manifest, and public
  boundary probes. Exact-head supported CI and D1-R re-review remain pending. The
  historical pre-correction runs are not acceptance evidence for this repair.
  After E1B acceptance, the dependency-finalization repair rebased D1 onto main
  `60c9afa` (including accepted I1 and E1B), replaced all generic core-dispatch use
  with the fixed five-value `et-e1b-private-raise` call in a never-installed trusted
  root, and packaged D1 with E1B as one registry-owning localized object. The normal
  artifact exposes exactly six E1B accessor symbols plus eight D1 wrapper symbols;
  checked I/O, the fixed raise seam, constructors, core dispatcher, private bridge,
  and D1 helper families are local. Public depfiles contain only
  `transformer.data` and `transformer.error_consumer`; exact repository normal/fault
  undefined-symbol manifests, `nm`, `readelf`, `strings`, and crafted-link evidence
  gate the completed artifact. A staged installed-public root also rejects explicit
  core, internal, trusted-root, and implementation imports in fresh source, object,
  and AOT probes. Both mixed import orders preserve error identity and
  all accessors across later failures. The pre-E1B golden corpus remained
  byte-identical, and the compatibility lane passed D1's 16 format/fault groups,
  A0, E1 (112 checks), E1B (35 checks), I1, Q0 (23 tests), and the full integrated
  suite. Exact-head supported CI and D1-R re-review remain pending.
  D1-R2 then identified a caller-controlled build-policy mismatch at exact head
  `383056853a1f269e28a413c346c2a56de0b1874f`. The repair removes
  `E1B_PACKAGE_POLICY` selection from the generic builder: D1 normal, D1 test-fault,
  and X1 policy now derive only from their complete canonical repository root,
  bridge, rename, export, and include tuples. Normal and fault D1 use distinct exact
  never-installed roots; their undefined manifests and `data_io.c` identity remain
  hard-coded. Copied/arbitrary roots, partial tuples, missing/extra/shadow includes,
  all policy overrides, and direct fault publication beneath the production artifact
  directory are rejected before toolchain use or output/evidence mutation. The
  compiler subprocess clears caller `ESHKOL_PATH` and pins `ESHKOL_LIB_DIR`; positive
  normal/fault builds with hostile shadow modules still record only canonical source
  paths. Sentinel negatives prove rejected routes preserve preexisting object,
  archive, and evidence bytes. Focused D1, E1B, X1, A0, E1, and I1 gates pass on the
  compatibility lane, as do the full `make test`, explicit `make build`, and
  `make smoke` gates. Supported exact-head CI remains required.
  After P1 merged at `b68ec9f9a64583800cfa888da2916f6ea99134b4`, the
  dependency-order rebase retained that hardened classifier and added P1 only as
  its exact repository trusted-root/bridge/rename/export tuple with no optional
  include directories and its fixed undefined-symbol manifest. P1 policy evidence,
  trusted-root depfile provenance, and a hostile `ESHKOL_PATH`/`ESHKOL_LIB_DIR`
  build are now explicit gates. Focused E1B, X1, P1, D1, A0, E1, and production
  Python-isolation gates pass, as do the integrated `make test`, explicit build,
  and smoke compatibility gates. Supported Ubuntu 22.04 / LLVM-Clang 21.1.8
  [CI run 33359439566](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33359439566)
  passed at exact independently approved head
  `ea8b6d1c87356aabbfce95e40726ca8983e5366f`. After merge, actual main passed
  `make test-d1`, including all 16 format/native-I/O tests and the strict compiled
  writer/validator boundary gates, on the documented local compatibility lane.
- **Dependencies / retest:** E1 issue #23 and E1B issue #33 are merged. The installed
  D1 facade imports only `transformer.error_consumer`; `transformer.error_internal`,
  `transformer.error_core`, the fixed raise seam, and all constructors are absent
  from its source dependency graph. Consumers must link the single completed D1/E1B
  artifact and must not combine it with another registry-owning E1B artifact.
  Compiler upgrades must rerun all source/object/AOT and crafted-link boundary checks.
  T1/D2/T2/DATA4 may now consume this accepted contract; D2
  retains all iteration, shuffle, packing, batching, and cursor behavior. A later
  streaming/I1 integration must rerun byte-determinism, publication, corruption,
  and scale tests.
- **Reference:** [issue #1](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1);
  [issue #18](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/18);
  [PR #29](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/29);
  [D1-R requested changes](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/29#issuecomment-5464425028);
  [D1-R2 requested changes](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/29#issuecomment-5471306523);
  [final independent review](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/29#issuecomment-5474526166);
  merge commit `98cf060b397ab00cd33c68521de952e24bda8704`.

## 2026-08-31 — C1 / issue #19

- **Decision:** accepted. The version 1.0 checkpoint contract below was independently
  reviewed, passed supported CI, merged, and passed an integration-owner retest of
  the actual merge commit.
- **Contract:** C1 format identity is `eshkol-checkpoint` version 1.0. The exact
  16-byte magic is `89 45 53 48 4b 4f 4c 43 4b 50 54 0d 0a 1a 0a 00`; all
  multibyte integers are unsigned little-endian and the header carries endianness
  marker 1. Version 1.0 requires zero feature bits and zero reserved fields and
  admits only SHA-256. A final 32-byte digest covers the domain
  `eshkol-checkpoint-container-v1\0` followed by every preceding file byte;
  each tensor also has a domain-separated SHA-256 over its canonical entry
  metadata and payload. Declared header, metadata, payload, tensor, and file
  lengths are exact, nonoverlapping, checked before allocation, and end exactly at
  EOF. Entries use P1 UTF-8 byte lexical path order and fixed kind, dtype, layout,
  shape, device, payload, and provider-identity encodings. Alias groups use sorted,
  disjoint parameter-entry indices. Version 1 hard limits are 1 TiB file bytes,
  256 MiB metadata bytes, 256 GiB per tensor, 4096 entries, 4096 alias groups and
  total members, 64 path segments/rank, 65536 UTF-8 bytes per segment, and 127
  provider-ID bytes; caller policy may only lower them.

  The serialized P1 provider spelling is inert canonical UTF-8 and is never
  interned as a symbol. It is compared only with trusted codec/provider symbol
  spellings. The constructed identity is
  `(transformer-tensor-provider 1 0 provider-id)`. Trusted load explicitly selects
  an already admitted exact provider and codec, compares the serialized identity,
  constructs detached tensors, creates one unpublished logical
  `transformer-state-dict` 1.0 with empty features, and invokes the reviewed P1
  binding helper. Checkpoint bytes never select or load callbacks, code, native
  libraries, or providers. The C1 codec seam is trusted/internal and requires exact
  provider identity plus unary encode/decode callbacks; it is not an A0 API or a
  capability claim. Current main has no reviewed production P1 tensor-byte codec,
  and I1 is not one. Production tensor save/load therefore remain unavailable and
  unsupported, and no capability is advertised; only the isolated inert P1 fixture
  may prove structural format/control-flow,
  ownership, and atomicity semantics. C1 does not broaden A0 `checkpoint-save!` or
  `checkpoint-load` from complete trainer state to P1 state dictionaries; C2 owns
  that later composition.

  Atomic writes use a narrow C11/Linux byte-I/O boundary only: an unpredictable
  `openat(O_CREAT|O_EXCL|O_NOFOLLOW)` same-directory temporary, checked
  short/EINTR/zero writes, temp `fsync`, successful close, atomic `renameat` or
  `renameat2(RENAME_NOREPLACE)`, then parent-directory `fsync`. Rename is the
  publication commit point. Pre-rename failure leaves the destination unchanged and
  attempts exact-temp cleanup; a crash or cleanup failure may leave an untrusted
  orphan that is never reused or swept. A directory-fsync or later directory-close
  failure after rename is reported as `io` with publication visible and crash
  durability conservatively unknown. Load
  follows no embedded path and rejects symlinks and non-regular inputs.
- **Evidence:** the unsupported CachyOS / LLVM-Clang 22.1.6 compatibility probe
  passed the focused C1 gate with the pinned Eshkol build: it reported 136 logical
  state, schema, provider, ownership, no-alias, error, and policy checks; two fresh
  AOT builds produced byte-identical executables and checkpoint bytes with reviewed
  SHA-256 `177a762eca5a535e01da4d740e676a3481c6617b632d4abe9fe9726dc6bb2769`;
  an independent parser plus the Eshkol validator passed 1012 cases: 820 deterministic
  structure-aware malformed, corruption, truncation, trailing-data, exact/one-over
  boundary, version, feature, checksum, overflow, UTF-8, path, alias, schema, and
  provider cases, plus 192 reproducibly seeded integrity mutations (which generally
  prove checksum rejection, not deep-validator reachability). Exact hard policy
  values and each one-over value execute in Eshkol without allocating those sizes;
  the independently checksummed 4096-entry/2048-group/4096-member wire boundary is
  parser-only because an attempted approximately 430-KiB Eshkol inspect exhausted
  the pinned runtime's fixed 1-GiB heap. The executable validator covers exact 64 and
  one-over 65 under its lowered policy. The native gate
  passed failpoint publication/cleanup, pre-existing destination, orphan, short/EINTR/
  zero I/O, C/C++ ABI, exact-symbol, deterministic-object, ASan, and UBSan checks.
  The earlier repository-wide run passed F0/Q0/A0/K1/E1/E1B/I1/X1/P1 and production
  Python-isolation gates, and a separate explicit unsupported-host `make smoke`
  rebuilt every artifact including C1 and printed `eshkol-transformer-smoke:v1`.
  LeakSanitizer execution is unavailable under the local ptrace-restricted executor;
  the supported CI ASan/UBSan gate remains required. Independent binary/security,
  atomic-I/O, P1-integration, adversarial-format, and documentation/evidence reviews
  report no blocker. Exact-head supported Ubuntu 22.04 / LLVM-Clang 21.1.8 run
  33398133014 passed configure, build, the complete integrated test matrix, smoke,
  and the reproducible benchmark. Independent C1-R approved exact head
  `bee6caf02d47d7da597b182cdb01bfbb5e3e1f1a` after reproducing the repaired
  zero-progress I/O, validation-before-codec, exact/one-over limit, and multi-alias
  evidence. PR #38 merged as
  `f5877b07cae3d18da198e5b146594de03c3a50a2`; the integration checkout then passed
  `make test-c1` on that merge commit with the pinned Eshkol build on the explicitly
  unsupported local LLVM-Clang 22.1.6 compatibility lane.
  C1 exposes no independently packaged facade because a second artifact would own a
  conflicting P1 registry; C2 must package C1 and its trainer schema into one
  prelocalized registry-owning E1B consumer before exposing A0 persistence operations.
- **Dependencies / retest:** T1 may consume only the reviewed internal persistence
  policy and atomic-I/O guarantees after C1 acceptance. C2 must supply the complete
  trainer-state schema and separately reviewed real tensor codecs before claiming
  exact resume. Any magic, layout, checksum, codec, limit, durability, or provider
  change requires issue #1 coordination and C1/T1/C2 retests.
- **Reference:** [issue #19](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/19);
  [integration issue #1](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1);
  [PR #38](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/38);
  [final independent review](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/38#issuecomment-5479830494);
  [supported CI run 33398133014](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33398133014);
  merge commit `f5877b07cae3d18da198e5b146594de03c3a50a2`.

## 2026-08-31 — T1 / issue #17

- **Decision:** accepted after independent exact-head review, supported CI, merge,
  and a merged-main retest. The integration owner accepted the Wave-1 aggregate,
  X1 baseline, C1 policy-shell, and I1 lifetime direction for implementation in
  issue #1 comment 5480447613; PR #40 implements and freezes the version-1
  tokenizer contract described below.
- **Contract:** T1 implements exactly the eight A0 tokenizer operations. Byte IDs are
  exactly `0..255`; specials are unique, contiguous from 256, and named by
  `[a-z][a-z0-9._-]{0,63}`. Encoding never recognizes special spellings. Ordered
  prefix and suffix lists may repeat only configured `omit` specials; decoding an
  `error` special or an unknown/out-of-range ID is an explicit error. Normalization
  is `none`. `raw` preserves every byte, including embedded NUL and malformed UTF-8;
  `strict` rejects malformed input/output without replacement or repair.

  The canonical artifact is the exact versioned seven-bit ASCII TSV grammar in
  `docs/TOKENIZER_FORMAT.md`: fixed record order and v1 limits, canonical unsigned
  decimal integers, exact payload length, no trailing bytes, and a lowercase SHA-256
  checksum over `"eshkol-byte-tokenizer-checksum-v1\n" || A`. Tokenizer identity is
  lowercase SHA-256 over `"sha256:eshkol-byte-tokenizer-v1\n" || H || P`; the public
  spelling is exactly
  `sha256:eshkol-byte-tokenizer-v1:<64-lowercase-hex>`. Version 1.0 has no features or
  optional fields. A higher v1 minor is readable only with no required feature and
  canonical inert optional fields. Artifact bytes are inert data and cannot select
  code, callbacks, providers, native libraries, paths, or capabilities.

  `tokenizer-byte` accepts only a same-aggregate validated X1 schema-1.0 resolved
  config with `model.vocabulary-size = 256` and constructs the raw/none/no-special/
  no-prefix/no-suffix baseline. Explicit specials, prefix/suffix behavior, and
  `raw|strict` policy come only from a validated artifact. T1 and the accepted E1,
  P1, D1, X1, and C1 trusted sources form one canonical aggregate, then localize
  once. Its exact public boundary is 46 globals: E1 6, P1 17, D1 8, X1 6,
  `persistence-policy` 1, and T1 8. Already-localized artifacts are not inputs.
  Installed module surfaces remain narrow, and `transformer.persistence` exposes
  only `persistence-policy` until C2.

  The public policy is an unforgeable observationally immutable same-aggregate shell
  containing copied C1-validated fields, and every use revalidates it. T1's effective
  file limit is `min(max-file-bytes, 1048576)` and its payload/metadata limit is
  `min(max-metadata-bytes, 1048576)`. The retained tensor limit/count fields do not
  apply to this tensor-free artifact; device is `cpu`. Forged, copied, mutated,
  foreign-aggregate, or malformed policies reject before I/O. T1 consumes C1 exact
  reads and atomic publication. Rename is the save commit point; a post-rename parent
  sync/close failure reports publication visible and durability unknown rather than
  claiming universal crash durability.

  `tokenizer-encode` returns a fresh sealed, rank-1, dense contiguous CPU I1 tensor
  with exact signed-`i64` storage and no alternate carrier, cast, transfer, or
  fallback. Its private fixed-arity shell operations are create, length, write,
  read, seal, and unpublished-only abort. Registry admission precedes every pointer
  dereference; writes are construction-only; abort ends the borrow and destroys and
  unregisters an unpublished tensor. Decode accepts only a sealed shell from the
  same aggregate. Native code implements transport and lifetime only; all tokenizer,
  UTF-8, special, parser, serializer, checksum, identity, policy, and error semantics
  remain Eshkol-authored.
- **Evidence:** the accepted implementation wires `make test-t1` to the
  development-only Python format oracle and frozen fixture, direct C/C++ shell ABI
  checks, ASan/UBSan execution, repeat builds/runs, exact 46-global admission, and a
  compiled Eshkol AOT public-runtime test. The public AOT path is authoritative;
  Python is neither linked nor executed by the production artifact. The independent
  documentation review ran all 26 development-oracle semantic, format, corruption,
  policy-limit, determinism, and hostile-environment tests successfully; the frozen
  artifact is 552 bytes with a 200-byte payload and SHA-256
  `93af83bf36428ff4b65b32a7e0a976e389fdf2b93bddaaa7c1d7b889460e8353`. The
  authoritative Eshkol/native gate passed twice on the compatibility host: 110 shell
  admission/exact-i64/lifetime/failpoint checks with C11, C++17, ASan, and UBSan; 9
  build-only semantic checks; 61 public adversarial AOT checks per repetition; and 32
  byte-identical generated adversarial artifacts per repetition. Boundary evidence
  fixes 46 global definitions, 40 non-E1 package exports, 144 undefined symbols, and
  the exact ten-source closure; it also passes ten flattened private-binding
  negatives, eleven crafted native-link negatives, reverse-import and hostile-path
  probes, copied/prelocalized/standalone-root rejection, cross-artifact registry
  collision, and production Python isolation. `make build`, `make test`, `make
  smoke`, and `make benchmark` pass on the explicitly unsupported CachyOS / LLVM
  22.1.6 compatibility lane.

  Independent T1-R review of head
  `02a3e419f237dce343fd93642515bddf0785e6e2` found that two public-AOT loads of one
  valid 472,645-byte combined-boundary artifact (4,096 specials with 64-byte names,
  4,096 prefix entries, and 4,096 suffix entries) reached at least 819 MiB of the
  pinned runtime's 1-GiB heap and emitted its 80-percent warning. That head proved
  syntax admission but did not provide acceptable operational-boundary evidence.
  The repair candidate preserves every exact v1 maximum, removes insertion-sort
  allocation amplification, and adds repeat public-AOT exact-max/one-over tests with
  bounded elapsed-time, RSS, heap-warning, canonical-save, and byte-determinism
  checks. Two frozen focused-gate runs completed the exact-max public AOT in 129,392
  KiB and 126,124 KiB peak RSS, below the explicit 512-MiB evidence budget, without
  the runtime heap warning; both made the canonical save byte-identical and rejected
  the three count-plus-one fixtures. The registry-lifetime AOT probe separately
  performs 16 baseline versus 128 growth cycles; every cycle discards one new policy,
  two tokenizer identities, and one sealed encode result, then verifies that the
  oldest values of every kind remain usable. Two frozen repetitions measured
  baseline/growth peak RSS of 48,652/57,080 KiB and 46,560/55,016 KiB, positive
  retained deltas of 8,428 KiB and 8,456 KiB. The gate requires the exact four
  oldest-identity checks, bounds each case at 30 seconds and 262,144 KiB RSS, and
  requires growth RSS to exceed baseline RSS.

  A later exact-head T1-R audit found that the same public-runtime claim also
  lacked exact optional-header evidence. The additive gate generates a deterministic
  98,891-byte version-1.1 artifact with 4,096 sorted unique inert optional-field
  records and one value of exactly 64 decoded bytes. Two bounded public-AOT
  repetitions load it, check its fixed fingerprint
  `sha256:eshkol-byte-tokenizer-v1:0831b9f3c303b182c9bc6bd83a3359875dd8156b8a340cdacd4ded012df8e53c`,
  save it byte-identically, reload it, and reject a separately checksummed 4,097-count
  artifact as `corrupt-data` / `tokenizer-load` with message
  `invalid optional-field count`; the existing adversarial public AOT retains the
  65-decoded-byte rejection. The combined exact-special/insertion and optional-limit
  executions measured 142,600 KiB and 142,864 KiB peak RSS, below the unchanged
  512-MiB evidence budget, without a heap-pressure warning. This changes no v1
  limit, API, fingerprint rule, package topology, or runtime implementation.

  The aggregate public `tokenizer-save!` AOT also repeats injected short-write,
  `EINTR`, zero-write, temporary-file sync/close, publication, directory sync/close,
  and cleanup-failure cases. It checks the exact E1 category, operation, native
  cause/domain/status/errno/stage, published and durability fields, reloads the old
  or new artifact as appropriate, rejects partial publication, and verifies orphan
  cleanup behavior. Independent implementation, format/security, lifetime/ABI,
  package-boundary, and documentation/evidence subreviews report no blocker. Final
  T1-R re-review approved exact head
  `fae9c35a754bdf735200d74de06d6013a35f4f58` with no findings after fresh focused
  and boundary runs. Supported Ubuntu 22.04 / LLVM-Clang 21.1.8 CI run 33445639643
  completed build, the integrated F0/A0/B0/C1/D1/E1/E1B/I1/K1/P1/Q0/T1/X1 test
  matrix, smoke, and the reproducible benchmark successfully in 1h12m14s; its
  combined-limit runs measured 142,496 KiB and 142,396 KiB peak RSS. PR #40 merged
  as `52ed785eabc7f1a6970fc5b42f1e98005ae0bcf7`. A fresh detached merged-main
  compatibility-lane retest then passed `make test-t1`, including the independent
  boundary gate; its combined-limit runs measured 142,856 KiB and 140,520 KiB.
- **Measured limitations / unsupported:** three strong append-only registries retain
  all successful tokenizer cores, persistence-policy shells and copied fields, and
  sealed encoded I1 shells/storage until process exit. Dropping caller references
  does not reclaim them. Registry lookup is linear in registered identities and
  cumulative memory grows monotonically; there is no registry-count ceiling,
  finalizer, weak identity, or public release operation. Concurrent or reentrant T1
  registry use is unverified and unsupported; callers serialize all T1 operations.
  Applications construct/load once and reuse identities. A bounded worker process
  and exit are the only current reclamation boundary where retention cannot be
  bounded. The shell is not a public raw pointer seam, K1 capability, general tensor
  interop surface, numerical kernel, device/dtype conversion, or performance claim.
  Exact v1 per-artifact maxima remain supported admission ceilings but do not promise
  bounded cumulative process use or indefinite repeated admissions. No
  power-loss, NFS, FUSE, hostile-parent, or concurrent-writer durability guarantee is
  made. T2/D2/training must not depend on repeated ephemeral encoding until a
  separately reviewed reclamation or reuse design exists.
- **Dependencies / retest:** T1 consumes the accepted A0, Q0, E1/E1B,
  I1, X1, P1, D1, and C1 contracts without changing their public names or arities.
  Any format, fingerprint domain, aggregate topology, public export, policy mapping,
  native shell ABI, admission, publication, lifetime, or reclamation change requires
  issue #1 coordination and complete T1 plus affected dependency retests.
- **Reference:** [issue #17](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/17);
  [accepted implementation contract in issue #1](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5480447613);
  [T1-R requested changes](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/40#issuecomment-5482913383);
  [PR #40](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/40);
  [final independent review](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/40#issuecomment-5486249037);
  [supported CI run 33445639643](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33445639643);
  merge commit `52ed785eabc7f1a6970fc5b42f1e98005ae0bcf7`.

## 2026-09-01 — O2 / issue #46

- **Decision:** review candidate. P1L issue #51 was independently approved and merged
  as `b72b9fa58042304a71e801415e53f280262edae2`; I2 issue #49 was independently
  approved and merged as `309de7262ebe33120e782ffc1c12f8cc10cbe74b`. Integration
  accepted O2's post-I2 contract with eight clarifications in issue #1 comment
  5557203111, accepted the exact 1,365-entry ceiling in comment 5557364807, and chose
  detached state release without an origin-optimizer backreference in comment
  5557373020. O2 may advance only to `review`; independent O2-R approval, exact-head
  supported CI, merge, merged-main retest, and acceptance-document follow-up remain
  pending.
- **Contract:** O2 preserves the five A0 optimizer names and adds only exact arity-1
  `optimizer-state-release!`. The public arities are 2/1/1/1/2/1. The one-member
  `build/o2/libeshkol_transformer_wave2.a` aggregate contains `o2_wave2.o`, preserves
  I2's 47 globals, and adds the exact six O2 wrappers for 53 globals total. It owns the
  single E1/P1/I2 registry universe and cannot be linked with another registry-owning
  aggregate. The fixed private release wrapper accepts no caller-selected provider or
  callback authority and is statically bound to provider
  `(transformer-tensor-provider 2 0 i2-dense-cpu-f32-v1)` plus the exact O2 ledger.
  It cannot release P1 state, parameters, gradients, live optimizer moments, or
  another owner.

  O2 does not change X1 schema 1.0. `optimizer-create` accepts the strict data-only
  `transformer-optimizer-config` 1.0 value in [O2_OPTIMIZER.md](O2_OPTIMIZER.md).
  Version 1 admits dense CPU-f32 AdamW, complete canonical parameter groups, optional
  global-L2 clipping, and constant or successful-update-indexed linear schedules.
  One step reads I2's exact present numerator/contribution-count/normalization-weight
  state, normalizes and clips once, and updates each unique protected storage identity
  plus its two moments through one 3N-assignment I2 commit. Conflicting tied-group
  assignments reject before mutation. Only a completed commit advances
  `completed-updates`; step never clears gradients, and `optimizer-zero-grad!` is the
  sole exact, idempotent clear operation.

  Version 1 admits at most 1,365 unique live parameter allocations and at most 1,365
  nonempty groups. Aliases do not count again and each unique allocation belongs to
  exactly one group. The same ceiling and canonical ordering govern receiver and
  logical-state validation. A 1,366th entry is `invalid-argument` before provider
  work, retained native allocation, or mutation.

  `transformer-optimizer-state` 1.0 is a byte-independent logical projection behind
  an opaque registered public owner; it is not a C2 encoding or file format. Snapshot
  and load require an idle optimizer and absent I2 gradients. Load validates the
  complete protected state and all finite moments, borrows without adoption, commits
  all destination moments once, and only then publishes configuration/counters; it
  never changes parameters or gradients. Release depends only on exact state/provider/
  ledger liveness and no active state borrow. It remains valid after the source
  optimizer later accumulates or steps, closes handles with `live -> releasing`,
  releases every clone exactly once, and publishes `dead`. Exact-dead repeat succeeds;
  every other dead-state operation is `invalid-state`.
- **Error and lifetime contract:** malformed/non-state/wrong-kind/forged/copied/
  unregistered/cross-aggregate release inputs are `invalid-argument`; recognized
  live busy/reentrant/releasing/owner-conflict is `invalid-state`; unavailable
  capability is `unsupported` only before construction/load; a post-admission
  provider defect is `internal`. An unavailable required binary32 execution mode is
  `determinism-unavailable`. O2 v1 has no RNG and no optimizer-destroy operation.
  Live optimizer receivers and their two moment tensors per unique parameter remain
  process-local until exit. Released snapshots retain no moment carriers, but dead
  state/handle and ended-borrow tombstones are cumulative; identity registries are
  serialized and linear-time.
- **Evidence:** The development-only PyTorch oracle contains 15 deterministic cases
  and 99 tensors covering later-step bias correction, present-zero decay, multiple
  groups, clipping, unequal-weight accumulation, schedules, explicit-only clearing,
  and state continuation. Its canonical 27,774 bytes have SHA-256
  `db075ceb47d121d4a13537cd2d0c2457f5596b4c4bb739766a9652b6bc66e561`.
  Runtime implementation and focused native/Eshkol/package tests are present; final
  consolidated exact-head local results, PR reference, supported Ubuntu 22.04 /
  LLVM-Clang 21.1.8 CI URL, and independent O2-R disposition are pending.
- **Dependencies / retest:** C2 may consume only the logical projection and scoped
  synchronous state/I2/K1 borrows; it must not serialize owner tokens, callbacks,
  executable provider authority, or capability evidence. C2 remains blocked on O2
  acceptance and D2. TR3 owns when to step and all token counters/token schedules.
  Before merge, rerun P1L/I2/O2/Q0 and repository gates, strict AOT, sanitizers/LSan,
  smoke, benchmark, exact aggregate/authority/isolation checks, supported exact-head
  CI, and independent O2-R. Do not merge from this workstream.
- **Reference:** [issue #46](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/46);
  [I2 issue #49](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/49);
  [binding integration decision](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5487253506);
  [proposed logical contract](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5487546881);
  [O2 evidence update](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/46#issuecomment-5487548739);
  [I2 lifetime blocker](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/49#issuecomment-5487674367);
  [binding P1L lifetime decision](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/51#issuecomment-5487927479);
  [O2 optimizer-state lifetime requirement](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/51#issuecomment-5493245836);
  [O2 dependency update](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/46#issuecomment-5493252239);
  [proposed O2 release operation](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5493291780);
  [independent-review addendum](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5493340320);
  [accepted lifecycle verdict](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5493412398);
  [O2/I2 shared-seam disposition](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/46#issuecomment-5493456203);
  [dead-state category proposal](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5493503924);
  [accepted dead-state taxonomy](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5493574404);
  [P1L PR #55](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/55);
  [accepted post-I2 freeze](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5557203111);
  [issue #46 post-I2 freeze record](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/46#issuecomment-5557203114);
  [accepted O2 limit](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5557364807);
  [issue #46 O2 limit record](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/46#issuecomment-5557364804);
  [accepted detached release](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5557373020);
  [issue #46 detached-release record](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/46#issuecomment-5557373040);
  P1L merge commit `b72b9fa58042304a71e801415e53f280262edae2`;
  I2 merge commit `309de7262ebe33120e782ffc1c12f8cc10cbe74b`.

## 2026-08-31 — L2 proposed contract / issue #44

- **Status:** proposed before implementation; no A0 public name, arity, or
  persistent format changes.
- **Native ABI direction:** L2 will provide an isolated version-1.0 native
  implementation of the existing `kernel.indexed-cross-entropy` capability with
  exact deterministic CPU-`f32` operations
  `indexed-cross-entropy.forward` and `indexed-cross-entropy.backward`. Requests
  have shape `[N,T,V]` with all extents positive. Forward consumes borrowed dense
  `f32[N,T,V]` logits and `i64[N,T]` targets and writes disjoint caller-owned
  `f32[N,T]` per-token losses. Backward additionally consumes borrowed
  `f32[N,T]` upstream gradients and writes disjoint caller-owned
  `f32[N,T,V]` logit gradients. L2 defines no mask or reduction; later model and
  trainer composition owns A0's weighted mean.
- **Numerical/error contract:** forward uses max-subtracted log-sum-exp in fixed
  row-major vocabulary order. Backward directly computes
  `upstream * (softmax - one_hot(target))` without allocating one-hot storage,
  invoking runtime autodiff, or using finite differences. Negative or
  out-of-range targets are `shape-mismatch`; zero extents are rejected. NaN/Inf
  operands and finite inputs whose forward result is not finite `f32` reject
  during validation before commit. There is no allocation, cast, copy, transfer,
  device substitution, hidden precision, scalar fallback, approximate gradient,
  or recoverable commit failure. Inputs and outputs remain byte-identical after a
  validation failure.
- **Discovery and Eshkol boundary:** a versioned L2 accessor supplies the provider
  descriptor only to an explicit caller-owned K1 resolver. L2 will not define the
  global `eshkol_transformer_kernel_provider_v1` symbol, search/load providers,
  or modify the provider-free K1 baseline. A private fixed-arity opaque transport
  context will prove real Eshkol AOT calls into the same provider; it is not an A0
  tensor API, owned f32 carrier, autodiff object, parameter store, or downstream
  training contract.
- **Cross-workstream coordination:** A2 requested the L2 ownership/ABI proposal.
  L2 will remain carrier-neutral and consume only accepted K1 borrowed tensor
  views; A2/N2 must not depend on the private L2 proof shell. K1 v1 permits one
  provider callback pair, so a unified N2/A2/L2 provider plus an owned f32/autodiff
  carrier remains an explicit composition decision before M3 rather than an API
  any Wave-2 primitive may invent independently.
  Integration subsequently created the shared I2 substrate at issue #49. L2 sent
  I2 its exact natural-alignment, immutable-borrow, pairwise-disjoint-input,
  caller-owned-output, two-phase-lifetime, and accessor-only provider-composition
  requirements. L2 will remain interoperable with accepted I2 dense f32 K1 views
  but does not treat I2 as an accepted dependency or production carrier until I2
  review and integration complete.
- **Planned evidence:** frozen Q0/PyTorch forward and direct-backward parity,
  development-only central finite differences, extreme finite logits, exhaustive
  schema/target/zero/nonfinite/alias/failure-atomicity negatives, repeated
  deterministic AOT, canonical capability report, sanitizers, production Python
  isolation, full repository gates, independent reviews, and supported exact-head
  Ubuntu 22.04 / LLVM-Clang 21.1.8 CI.
- **Reference:** [issue #44](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/44);
  [issue #1 proposal](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5487177062);
  [I2 coordination update](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5487282264).

## 2026-08-31 — T2 / issue #43

- **Decision:** proposed for integration-owner review; a local implementation
  candidate now exercises the contract but remains unfrozen and unaccepted.
- **Contract proposal:** preserve the accepted T1 `eshkol-byte-tokenizer` 1.x byte
  grammar, fingerprint domain, eight public names/arities, C1 policy mapping, exact
  I1 result carrier, and process-lifetime rules byte-for-byte. Add a distinct
  `eshkol-bpe-tokenizer` 1.0 data format and
  `sha256:eshkol-bpe-tokenizer-v1:<digest>` identity domain rather than treating BPE
  semantics as an inert T1 minor extension. Byte IDs remain `0..255`. Learned merge
  IDs are contiguous from 256 in rank order; configured special IDs follow the
  merge range contiguously. Each merge references only earlier IDs. Training chooses
  the greatest adjacent-pair count, breaks ties by ascending `(left-id,right-id)`,
  applies the chosen pair left-to-right without overlap inside each document, never
  crosses an explicit document boundary, and stops when the requested merge bound is
  reached or no pair meets the minimum frequency. Admitted document order and chunk
  partition within a document do not affect learned bytes.

  The proposed v1 operational ceilings are 1,048,576 artifact/payload bytes under
  the existing lowering persistence policy, 256 learned merges, 256 decoded bytes
  per learned token, 4,096 specials, 4,096 prefix entries, 4,096 suffix entries,
  65,536 aggregate training bytes, 4,096 documents, 4,096 training chunks, and
  65,536 bytes per encode/decode stream and 73,728 decoder token IDs (589,824
  i64-le staging bytes), closing the maximum 65,536 byte tokens plus 4,096 prefix
  and 4,096 suffix insertions under round trip. Exact ceilings and one-over rejection are
  part of the compiled gate; RSS/time thresholds are evidence budgets, not hidden
  lower admission limits.

  T2 adds no installed public procedure. Existing `tokenizer-load`,
  `tokenizer-save!`, encode/decode, vocabulary, fingerprint, and special lookup
  wrappers operate on either accepted T1 byte artifacts or validated T2 BPE
  artifacts without changing their contracts. BPE training and stateful streaming
  encode/decode are fixed-arity build-only Eshkol contracts for trusted later CLI/data
  composition. Streaming uses private i64-le staging chunks and a pipeline of one
  left-to-right transducer per merge rank, with at most one pending token per rank;
  it does not publish one retained T1 I1 shell per input chunk. Prefix and suffix
  specials are applied once per logical stream. No new native ABI,
  tensor carrier, dtype/device conversion, callback selected by data, or fallback is
  proposed.

  A canonical Wave-2 successor aggregate is rebuilt from the accepted trusted
  E1/P1/D1/X1/C1/T1 sources plus T2 and localized once. It retains the accepted
  upstream public surface and cannot be combined with the already-localized Wave-1 aggregate.
  The Wave-1 archive and all focused T1 evidence remain independently reproducible.
  To prove D1 round trips without changing D1 v1 bytes or its eight-name facade, the
  Wave-2 trusted closure adds a bounded internal corpus-token read operation. After
  accepted P1L integration, the successor aggregate carries its public unary
  `state-dict-release!` unchanged: production is exactly 47 globals/41 non-E1
  wrappers and the D1-test-only aggregate is exactly 48/42. T2 itself still adds no
  installed public procedure or native ABI. It
  fully validates the manifest and shards, then compares the supplied tokenizer
  fingerprint and vocabulary before returning any tokens; a self-consistent corpus
  paired with the wrong tokenizer is `invalid-argument`, not `corrupt-data`.
- **Status / evidence:** local candidate implementation. The distinct v1 artifact,
  Eshkol trainer, rank-stage whole/stream runtime, strict/raw policy, D1 bridge,
  unchanged T2 API with a 47-global production aggregate and 48-global test-only
  aggregate, frozen
  Python oracle, deterministic fixture generators, exact-limit/adversarial AOTs,
  and production-oracle isolation gates are checked in for review. After the
  independent T2-R request-changes review, decoder push now validates and sizes its
  current chunk before state mutation and allocates exactly the current decoded-byte
  count rather than the remaining 65,536-byte budget. The compiled gate exercises
  the exact maximum partition of 73,728 one-ID omit chunks and the single-chunk
  equivalent. Compiled public AOT coverage now includes 22 frozen parser rejection
  checks and four D1 seam negatives; malformed, truncated, and checksum-corrupt D1
  shards retain `corrupt-data/token-corpus-validate`, while vocabulary-only mismatch
  is `invalid-argument/t2-token-corpus-read`. Strict UTF-8 evidence uses a real
  four-byte scalar at every split and pins F0/F4 lower/upper boundary cases.

  Local compiled evidence includes 91 training/stream checks, 6 core checks, 17
  delivered-public checks, 6 D1-runtime checks, 22 parser negatives, 4 D1 negatives,
  60 UTF-8 split/boundary checks, and 25 Python oracle tests. Two fresh-cache
  training/stream runs measured 142,816 and 142,932 KiB peak RSS; the 73,728 one-ID
  decoder runs measured 10,632 and 10,580 KiB; UTF-8 runs measured 9,144 and 8,564
  KiB; two public-runtime runs measured 54,984 and 7,832 KiB; the parser-negative,
  D1 setup, and D1-negative runs measured 57,256, 10,352, and 9,368 KiB. Every
  bounded process completed within 60 seconds, below 524,288 KiB, with no
  heap-pressure warning. The T2-specific boundary suite independently retained and
  byte-compared two localized production and D1-test objects, archives, evidence
  directories, and public-caller AOT binaries. After merging accepted main
  `b72b9fa58042304a71e801415e53f280262edae2`, the complete compatibility gate
  passed with exact 47/41 production and 48/42 D1-test surfaces, exact public-string
  and public-source manifests, inherited P1L private-capability negatives, hostile
  path/tuple rejection, private/native localization and archive index closure,
  crafted-link isolation, and the Wave1+Wave2 duplicate-E1 rejection. The two
  training/stream runs measured 142,936 and 144,936 KiB; the exact 73,728 one-ID
  partitions measured 10,576 and 12,620 KiB; UTF-8 measured 8,972 and 5,824 KiB;
  public runtime measured 9,540 and 10,564 KiB; parser negatives measured 65,192
  KiB; D1 setup and negatives measured 2,532 and 3,760 KiB. Supported Ubuntu 22.04
  / LLVM-Clang 21.1.8 run 33697384015 then passed Build, the full Test matrix, and
  Smoke, but its benchmark was cancelled when the job reached the exact 150-minute
  outer ceiling. The measured run spent 14m02s in Build, 1h55m34s in Test, and
  13m40s in Smoke before the benchmark received only 3m38s. The workflow outer
  ceiling therefore rises narrowly from 150 to 180 minutes; no compiler timeout,
  test, sanitizer, Smoke assertion, or benchmark sample is removed. A new supported
  exact-head run, exact-head T2-R rereview, and the unmerged PR remain required. T2 stays no
  further than `review` until independent acceptance, merge, merged-main retest, and
  acceptance-document follow-up.

  A subsequent authoritative T2-R rereview of exact head
  `e667523adfcde28b88f3238ff336824c04ad166e` accepted the decoder, D1, package,
  UTF-8, API, format, aggregate, and supported-run evidence but required exhaustive
  production-linked coverage for every frozen parser invariant. The bounded repair
  expands deterministic generated evidence from 32 to 365 artifacts: 360
  byte-distinct malformed models, four valid exact-count-ceiling models, and the
  retained alternate D1 model.
  The public Eshkol AOT asserts 362 exact rejection category/operation tuples including
  two lowered-policy cases, performs a frozen-identity valid load after every failure,
  admits exact 613-byte/295-byte policy and merge/special/prefix/suffix count ceilings,
  and performs a canonical post-matrix save, for 730 checks per repetition. The new
  topology cases exposed boolean-valued one-over field counts escaping through numeric
  `=`; all parser record-shape guards now use a bounded boolean-safe predicate and map
  malformed records to the promised structured `corrupt-data/tokenizer-load` result.
  Python remains a deterministic byte generator only. Focused, full, supported, and
  independent exact-head evidence remain required for this repair; T2 remains `review`.
- **Dependencies / retest:** any accepted change to the aggregate source closure or
  D1 trusted internals requires complete T1 and D1 regression/boundary gates. Any
  public name, native ABI, T1 grammar/fingerprint, D1 byte-format, I1 lifetime, or
  persistence-policy change requires a new issue #1 decision and affected downstream
  retests.
- **Measured limitation:** the inherited D1 summary registry retains every
  successfully validated internal corpus summary until process exit, even if the
  later tokenizer identity comparison fails. T2 private cores/states are uniquely
  owned trusted-build values and are not safe for arbitrary representation mutation;
  registry operations are serialized.
- **Reference:** [issue #43](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/43);
  [integration issue #1](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1);
  [T2-R request changes](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/54#issuecomment-5514480850).

## 2026-09-01 — P1L / issue #51

- **Decision:** proposed; implementation and independent exact-head re-review are in
  progress. PR #55 remains open, and `docs/ROADMAP.md` remains at `review`. Nothing
  in this entry accepts, merges, or authorizes downstream integration of P1L.
- **Contract under review:** the P1 provider interface advances to 2.0 with exactly
  eight unary callbacks. The eighth, `release-owned!`, consumes one P1-owned carrier
  through a preallocated request envelope, is admitted only after validation as
  nonallocating/nonraising/semantically infallible, and acknowledges by returning
  that exact envelope identity. Under a conforming admitted provider, every clone
  has exactly one owner and is either transferred into one live state dictionary or
  released exactly once. Provider
  identity is inert `(transformer-tensor-provider 2 0 provider-id)` data; no
  serialized name selects code or upgrades a 1.x provider.

  `transformer.module` adds only public unary `state-dict-release!`, for 18 public P1
  operations total. Exact registered dead-state release is idempotent. Every other
  recognized dead-state or dead state-backed-handle operation fails `invalid-state`
  before native dereference; malformed, forged, copied, unregistered, and wrong-kind
  inputs remain `invalid-argument`. `state-dict-tensor` returns a cached read-only
  state-backed identity, not a fresh native clone. Trusted consumers may resolve and
  borrow its carrier only synchronously inside one serialized call and retain no raw
  pointer. Release rejects while such a call is active, invalidates the state and
  dependent handles before carrier destruction, and may retain only inert identity
  tombstones without tensor storage or released lifecycle graphs. Native identity
  ABI 1.1 preserves existing record sizes and offsets. No generic dispatcher,
  caller-selected provider, second registry, finalizer assumption, numerical
  provider, module release, or O2 public release operation is added.

  The existing unary `storage-identical?` callback is the authoritative same-
  provider physical-storage equivalence relation, not wrapper or raw data-pointer
  equality. After carrier/provider/liveness prevalidation it is total,
  deterministic, nonallocating, nonraising, nonretaining, side-effect-free, and
  semantically infallible; true means release through either carrier invalidates the
  other, false means disjoint releasable allocations, and the relation is reflexive,
  symmetric, and stable for the serialized call. It grants no release authority.
  P1 disarms candidate envelopes before comparison and restores only proved-
  independent owners. Callback raise, mutation, malformed result, or semantic lie
  fails closed without releasing the ambiguous carrier. Therefore exact cleanup and
  baseline restoration are claims only for a conforming provider, never for a
  trusted provider that violates this obligation.

  C1 container format remains 1.0 because its canonical schema already carries the
  provider interface major/minor; the proposed implementation admits only provider
  2.0 before callback use. The single Wave-1 aggregate remains one registry-owning
  object with 47 global definitions and 41 non-E1 exports. P1 is one
  24-definition/18-export package. Exact checked-in P1, C1, and T1 trusted source
  closures and public/symbol/string/undefined/rename manifests gate these boundaries.
- **Evidence to date:** reviewed PR head
  `d0afa985d4390cc79895e4e8eb4f3a34af84094f` reported 303 P1 structural checks,
  405 P1 native checks, 138 P1 registry-atomicity checks, 202 C1 logical checks,
  1,012 C1 adversarial cases, and exact 24/18 P1 plus 47/41 T1 boundaries. Supported
  Ubuntu 22.04 / LLVM-Clang 21.1.8 run 33554345381 passed build, the full test matrix,
  smoke, and benchmark for that exact head/base tree. That green run is not
  acceptance evidence: independent P1L-R requested changes for protected-storage
  aliasing, supported C1 leak detection, missing callback-defect combinations,
  retained entry owner graphs, and the absent exact P1 source-closure manifest.
  The repaired implementation has local compatibility-lane passing evidence for 419 P1
  structural checks, 405 P1 native checks, and 169 registry checks including the
  bounded eight-shell released-owner-graph probe and protected/ordinary comparator
  reentrancy attacks against state release and parameter/buffer registration. C1
  passes 245 logical checks plus 1,012 adversarial cases, and the exact T1 47-global
  aggregate/boundary gate passes.
  The current bounded repair passes the complete P1, C1, and T1 focused gates on
  CachyOS/LLVM 22. Before this test-and-documentation-only correction, the unchanged
  production implementation also passed `make build`, `make test`, `make smoke`, and
  `make benchmark` on that host; the benchmark truthfully reports
  `compatibility-only`. The repair also adds the exact four-source P1 trusted closure
  manifest and enables `P1_LSAN=1` plus `C1_LSAN=1` on the supported job.

  Two supported-run time-budget failures produced two explicit, bounded budget
  relaxations. Run 33597586986 showed that the larger repaired P1 trusted package no
  longer fit the generic E1B builder's 120-second inner compile limit; the successful
  supported invocation later needed about 176.7 seconds. Commit
  `d5f77fa5bf29d1f1a80c1111fda2b758aea17a14` therefore gives only the P1 package
  wrapper the already validated `P1_COMPILER_TIMEOUT_SECONDS` default of 360 seconds.
  The generic E1B default remains 120 seconds. Run 33601354539 then completed Build
  and the entire Test step, including supported P1/C1 leak detection, but the
  90-minute outer job budget cancelled Smoke and skipped the benchmark. Commit
  `937dc5b12962415413dfb0ce7ae2e16909ffa4f1` raises that bounded outer budget to 120
  minutes.

  Integration admits these bounded build-budget relaxations: one P1-specific inner
  compile ceiling rises from 120 to 360 seconds and the workflow's outer ceiling
  rises from 90 to 120 minutes. This disposition neither accepts P1L nor removes a
  compiler invocation, correctness assertion, test or sanitizer selection, Smoke,
  or the reproducible benchmark. On prior repair head
  `937dc5b12962415413dfb0ce7ae2e16909ffa4f1`, supported Ubuntu 22.04 / LLVM-Clang
  21.1.8 run 33609401595 completed Build, the full test/sanitizer aggregate, Smoke,
  and benchmark in 1h43m03s with `P1_LSAN=1` and `C1_LSAN=1`; it validates both
  admitted budgets but predates the exhaustive eight-position callback-allocation
  evidence. New exact-head supported CI and independent approval remain pending
  after the bounded findings in review comment 5508808267. The P1L decision remains
  proposed.
- **Measured limitations / unsupported:** pinned Eshkol has no proved finalizer, and
  the local ptrace-restricted compatibility executor cannot supply authoritative
  LeakSanitizer evidence. The repository ships no production numerical tensor
  provider or codec, so fixture callbacks prove control flow and ownership only.
  Concurrency and reentrancy remain unsupported except for the explicit rejection
  paths under test. Checkpoint format 1.0 is unchanged; C1 still does not expose the
  later complete trainer-state API.
- **Dependencies / retest:** I2 may implement only against the proposed provider-2.0
  and ABI-1.1 shape after P1L independent approval and merge. I2 has independently
  confirmed that it can canonicalize wrappers to the live owning `et_f32_tensor`
  resource, including keeping distinct zero-element owners disjoint despite null
  data pointers, but has made no runtime change, rebase, or integration. That design
  compatibility is not P1L acceptance. O2 may later reuse only
  the narrow trusted exact-one-carrier release mechanism behind its own separately
  reviewed owner ledger; P1L adds no optimizer public operation or generic release
  authority. N2, O2, C1, T1, C2, and the complete Wave-1 aggregate remain blocked on
  the repaired exact-head P1/C1/T1 gates, supported CI, and independent P1L-R
  approval.
- **Reference:** [issue #51](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/51);
  [integration issue #1](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1);
  [PR #55](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/55);
  [P1L-R requested changes](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/55#issuecomment-5501171044);
  [frozen storage-equivalence decision](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/51#issuecomment-5502247288);
  [I2 compatibility confirmation](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/49#issuecomment-5502269486);
  [issue #51 downstream disposition](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/51#issuecomment-5502293490);
  [integration issue #1 disposition](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5502293552);
  [timeout-evidence review correction](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/55#issuecomment-5508808267);
  [inner-timeout run 33597586986](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33597586986);
  [outer-budget run 33601354539](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33601354539);
  prior-head supported [CI run 33609401595](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33609401595).
## 2026-08-31 — A2 / issue #45

- **Decision:** accepted for implementation with required cache-view corrections in
  integration comment `5487284582`; independent exact-head review, supported CI,
  merge, local retest, and acceptance follow-up remain pending.
- **Contract:** A2 changes no A0 public name or arity. It proposes a
  carrier-neutral deterministic CPU-f32 K1 provider obtained only through the
  versioned `et_a2_kernel_provider_v1` accessor, plus a separate fixed-capacity KV
  cache ABI 1.0 and a private Eshkol AOT transport. The provider does not export the
  generic K1 resolver symbol or alter provider-free baseline discovery.

  `kernel.causal-attention` exposes explicit forward and analytic-backward
  operations with semantic request shape `[N,Hq,Hkv,Tq,Tk,Dh]`. K1 v1 range
  records are disjunctive, so the one uniquely named capability advertises only
  ten exact min=max rows: `[1,2,2,1,1,1]`, `[1,2,2,2,2,1]`,
  `[1,2,2,1,2,1]`, `[1,2,2,1,2,2]`, `[1,2,2,2,2,2]`,
  `[1,4,2,1,1,2]`, `[1,4,2,3,3,2]`, `[2,4,2,2,3,4]`,
  `[2,4,2,3,3,2]`, and `[2,4,2,1,3,2]`. Broader shapes admitted by the
  defensive provider validator are unverified and rejected by K1 capability
  resolution. Q is
  `f32[N,Hq,Tq,Dh]`; K and V are distinct `f32[N,Hkv,Tk,Dh]`; query and key
  positions are `i64[N,Tq]` and `i64[N,Tk]`; the exact nonbroadcast keep mask is
  `bool[N,Tq,Tk]`; and the output/upstream is `f32[N,Hq,Tq,Dh]`. Query head `h`
  maps to KV head `floor(h/(Hq/Hkv))`, with exact divisibility. Version 1 admits
  proved MHA and GQA with `Hkv >= 2`; MQA remains MOD5 scope. Admission requires a
  true keep-mask element and `key-position <= query-position`. Stable f32 softmax
  serially accumulates the dot product, computes `root=sqrtf((float)Dh)`, then
  `scale=1.0f/root`, and multiplies the completed sum by that scale. A fully masked
  row returns positive-zero output and zero
  adjoints. Floating operands are finite; positions are nonnegative, strictly
  increasing per row, and at most `16777215`. Masks and positions have no gradient.

  `kernel.rope` exposes forward and analytic backward over
  `f32[N,H,T,Dh]`, exact `i64[N,T]` positions, and positive finite
  `inv-freq f32[Dh/2]`, with exact even `Dh >= 2`. Its one capability advertises
  only `[1,1,1,2]`, `[1,1,2,2]`, `[1,1,2,4]`, `[2,2,3,4]`,
  `[2,4,3,2]`, and `[2,2,3,2]`. Adjacent pairs rotate by
  `position * inv-freq[i]`; backward applies the inverse rotation. The inv-frequency
  input avoids freezing an unaccepted model-level base or scaling policy.

  The opaque cache owns distinct, finite-zero-initialized preallocated keys and values
  `[L,N,Hkv,C,Dh]` and shared exact `i64[N]` logical lengths. Capacity and storage
  identities are fixed. Append width `A` is positive. A transaction validates
  `0 <= count[i] <= A`, requires at least one positive count, and proves
  `length[i] + count[i] <= C` without overflow before it stages each
  layer exactly once outside committed logical lengths, exposes only a
  transaction-scoped full-capacity dense `[N,Hkv,C,Dh]` K/V view plus immutable
  effective `i64[N]` lengths and a dense bool `[N,C]` mask whose bytes are one
  below each effective length and zero otherwise for a staged layer, and
  advances shared lengths only after every layer is staged. Abort or precommit
  failure preserves every observable prefix, length, identity, and source; tail
  bytes outside logical lengths are deterministically positive zero and excluded
  by the canonical mask. One live
  nested layer view may begin only after that layer is staged; while live it blocks
  stage, commit, abort, and destruction and must be explicitly ended. Commit requires
  all layers staged and no live view. A caller must use the immutable effective
  lengths and the exact false-outside-length mask; A2 never exposes a shorter
  `Tk < C` canonical dense view over capacity-strided storage. Registry admission
  precedes native-handle dereference.

  K1 validation performs every fallible schema, alias, range, finite-value, and
  numerical check without mutation. Invoke allocates nothing, cannot fail, and
  fully writes disjoint caller-owned outputs in a fixed order using explicit serial
  CPU-f32 arithmetic with contraction disabled. This is a reviewed baseline native
  kernel, not an accelerated, fused, compiler-reverse-AD, P1, or general tensor
  capability. There is no core, scalar, dtype, device, cast, transfer, allocation,
  precision, or cache fallback.
- **Exact-head review corrections:** Independent A2-R review of
  `f85a05de077092dcb29bebdbbdb3d9ff81cde111` requested corrections. The repair
  leaves K1 v1 unchanged and narrows capability metadata to the exact disjunctive
  rows above; it adds K1 require negatives for `Hq < Hkv`, nondivisible heads,
  odd RoPE `Dh`, and otherwise valid shapes outside the published rows. The
  supported-platform arithmetic regression distinguishes reciprocal-then-multiply
  from divide-after-sum, while the frozen PyTorch fixture remains a tolerance-based
  mathematical oracle and regenerates byte-identically.

  KV-cache ABI 1.0 now maps a wrong major to
  `ABI_MAJOR_MISMATCH` and a too-new minimum minor to
  `UNKNOWN_REQUIRED_FEATURE`. Every creator or descriptor output slot must be
  pointer-aligned, disjoint (including from logical dtype/device text spans), and
  NULL on entry; destructive handle slots are nulled only on success;
  every output remains unchanged on failure. All caller-declared spans require a
  representable exclusive end before dereference. The entire staged physical
  `[N,Hkv,A,Dh]` K/V source, including unused padding, must be finite; unused values
  are validated but not copied or exposed. New high-address, output-slot,
  finiteness, failpoint-continuation, and two-batch numerical tests close those
  review findings without changing the accepted full-capacity cache-view contract.
- **Evidence:** Initial canonical-pin AOT probes prove only narrow scalar/vector
  gradient behavior. A separate built-in attention probe is noncausal without an
  explicit mask and compiler/source inspection finds double scalar loops and
  incomplete fallback/backward behavior, so pinned-core attention/RoPE are rejected.
  K1 continues to report f32, reverse AD, matmul, and causal attention unverified.
  Independent contract/capability, implementation/test-design, documentation/
  packaging, and adversarial/numerical reviews approve the narrow provider/cache
  boundary after their findings were resolved. The focused gate compile-checks the
  ABI and passes 348 attention/RoPE provider checks, 780 cache checks, and 407
  cached-attention integration checks in both optimized and ASan/UBSan builds.
  It also verifies frozen Q0/PyTorch forward and gradient bits, direct finite
  differences, MHA/GQA admission, causal and fully-masked boundaries, RoPE boundary
  positions, transactional failure atomicity and failpoints, archive manifests,
  provider-free K1 baseline preservation, deterministic repeatability, and a
  private Eshkol AOT forward/backward/cache path. The private transport remains test
  evidence only. The complete local repository test gate also passes on the
  documented unsupported CachyOS/LLVM 22 compatibility host. Supported Ubuntu
  22.04/LLVM 21 exact-head CI remains pending. The corrected local focused gate
  passes 721 provider, 961 cache, and 677 cached-attention checks in deterministic
  optimized and ASan/UBSan runs, all four frozen-oracle tests, and the private AOT
  path on the explicitly unsupported compatibility host. A subsequent exact-head
  review found that the `N=2` finite-difference and cached incremental/full checks
  were correlated with the provider and therefore did not independently prove
  batch indexing. The corrected frozen PyTorch fixture now adds full `N=2`
  attention output and dQ/dK/dV with distinguishable batch values, positions, and
  masks, plus RoPE output/dX and all cached incremental outputs with distinguishable
  batch values and positions. The header checksum binds the complete fixture; every
  frozen expected output/gradient word is copied into the C header and compared
  elementwise. The focused gate now passes
  1,041 provider, 961 cache, and 869 cached-attention checks plus five oracle-format
  checks, and it compiles and rejects the reviewer-specified Q/K/V batch-zero and
  RoPE position-batch-zero source mutations at independent reference assertions.
- **Dependencies / retest:** M3/G3 cannot treat the private A2 transport as a shared
  tensor API. They remain blocked on a separately accepted f32 carrier, P1 provider,
  provider aggregation, and production Eshkol ownership/lifetime boundary. N2 and
  L2 currently define no shared carrier. Any public attention module, P1 binding,
  general tensor shell, configurable RoPE, MQA, additive mask, accelerator, or cache
  serialization requires a new issue-#1 decision and affected retests.
- **Reference:** [issue #45](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/45);
  [integration proposal](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5487219486);
  integration verdict `5487284582` on issue #1 and issue #45;
  [A2-R requested changes](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/52#issuecomment-5494434343);
  [correction direction](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/52#issuecomment-5494455136);
  [cache-bound ledger correction](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5494455138).

## 2026-09-04 — L2 current-main integration / PR #56

- **Decision:** merge accepted-main parent
  `b72b9fa58042304a71e801415e53f280262edae2` into reviewed L2 parent
  `264cfa7f3d6e33d140f264f2077b91d8ff66f9b6` without rewriting either history.
  The five shared-file conflicts are resolved as a strict union: A2 and L2 remain
  separately built and tested, both isolated provider/archive contracts remain
  documented, P1L dependency edges are retained, and A2, P1L, and L2 remain at
  `review`.
- **Boundary:** no L2 implementation, ABI header, export manifest, focused test,
  fixture, oracle generator, or private AOT proof source changes in this integration.
  No A2 or P1L implementation tree from the accepted-main parent is modified. L2
  still supplies only its versioned explicit provider accessor and does not define
  K1's canonical provider symbol, a registry, an owned carrier, or a shared private
  AOT transport.
- **Required evidence:** byte-for-byte parent comparisons, focused L2/K1/A2/P1
  gates, full repository test/smoke/benchmark gates, supported Ubuntu 22.04 / LLVM
  21 exact-head CI, and a separate independent exact-head L2-R review are required
  before integration acceptance. PR #56 remains open and must not be merged here.

## 2026-09-04 — L2/T2 current-main refresh / PR #56

- **Decision:** merge accepted T2 main
  `3006c5c90d1ee40647cababac004f2c75d46fa65` into independently reviewed L2
  integration head `cab8b790ec8b98a47b064a37c16893979622a265` without rewriting either
  history. The three content conflicts are strict registry unions: the CI test label
  names both L2 and T2, the Makefile retains both focused gates, and this log retains
  both complete workstream records. T2's 180-minute supported-job budget is
  unchanged. L2 and T2 remain at `review`.
- **Boundary:** no L2 implementation, ABI, export manifest, focused evidence, oracle,
  fixture, or private AOT source changes. No accepted T2 implementation, format,
  aggregate, fixture, oracle, build/test script, or boundary evidence changes. T2
  adds no tensor/provider contract and L2 adds no tokenizer/aggregate contract; the
  workstreams interact only through explicit build/test orchestration.
- **Required evidence:** exact parent byte/mode comparisons, focused L2 and T2 gates,
  the complete affected repository test/smoke/benchmark gates, new supported Ubuntu
  22.04 / LLVM-Clang 21.1.8 exact-head CI, and a separate bounded L2-R integration-
  delta review are required. PR #56 remains open and must not be merged here.

## 2026-09-03 — N2 / issue #47

- **Decision:** frozen for review with the three binding clarifications below after
  I2 PR #53 exact approved head `c4522b2e5473ac46d5d45e889755076ffdb89cfa`
  merged into main as `309de7262ebe33120e782ffc1c12f8cc10cbe74b`. N2 is
  `review`; no Wave-2 provider aggregate is claimed.
- **Contract frozen:** one registry-free deterministic CPU-f32 provider archive,
  `build/n2/libeshkol_transformer_n2.a`, contains only
  `n2_primitives_provider.o` and exposes only
  `et_n2_kernel_provider_v1` through
  `include/eshkol_transformer/n2_primitives_abi.h`. It defines no canonical K1
  symbol and no A0 Eshkol name. Whole-capability ownership is unique: a future
  single registry-owning Wave-2 aggregate rejects duplicate capability names and
  forwards unchanged K1 calls, tensor descriptors, and storage identities to one
  accessor-selected owner.

  The exact capabilities are distinct `kernel.embedding-forward` and
  backward-only `kernel.embedding-backward`, plus `kernel.matmul`, `kernel.norm`,
  `kernel.activation`, `kernel.dropout`, and `kernel.residual`. Every advertised
  K1 row is min=max and every operation/row Cartesian combination must execute in
  the focused gate; zero and absent rows remain unsupported. Exact ordered tensor
  schemas, unnormalized analytic VJPs, input-alias rules, serial binary32 reduction
  order, two-pass biased LayerNorm, exact-erf GELU, positive-zero ReLU kink policy,
  failure-atomic dry preflight, and residual per-edge/I2 accumulation semantics are
  fixed by the accepted corrected proposal.

  Dropout uses a self-contained signed-`i64[4]` Philox4x32-10 state with numeric
  version `+1`, exact signed/unsigned mappings, published constants/equations/lane
  order, overflow-safe block counting, tail discard, and an exhausted maximum
  counter sentinel. Train backward receives the original state, regenerates the
  mask, and emits no successor. Both signed p zeros and eval are byte-copy paths
  with no RNG work. Positive-p inverted scaling is explicitly binary32. The
  supported environment is RNE, no contraction/excess precision, and FTZ/DAZ off;
  dry preflight restores the complete incoming FP exception status.

  Every embedding ID is signed exact i64 and must satisfy numeric `0<=id<V`;
  negative and `id>=V` values reject as `invalid-argument` before row access.
  LayerNorm epsilon must be finite and numerically `>0.0f`; both signed zeros,
  negatives, infinities, and NaN reject before evaluation or mutation.
  Provider input/input overlap rejects with `ET_KERNEL_ERROR_INVALID_ARGUMENT` and
  `ET_KERNEL_CODE_PROVIDER_REJECTED`, never K1's `ALIASING_OUTPUT`, which remains
  reserved for K1-detected output overlap.
- **Carrier-neutral evidence to date:** exact-pin compiled probes show that generic
  Eshkol tensors retain f64-distinguishing values, tensor/LayerNorm reverse AD is rejected or
  wrong, built-in dropout has hidden/global divergent RNG behavior, and malformed
  reshape/broadcast/dimension operations may return garbage, crash, or accept
  invalid input. These observations reject generic tensor, compiler-AD, and built-in
  dropout paths; they do not establish production f32/device/ownership support.
  The carrier-neutral Q0 fixture contains 26 cases and 82 tensors against exact
  PyTorch `2.13.0+cpu`; all 31 reference, Philox, fixture-integrity, and independent
  numerical-gradient tests pass under that pin. Under the system interpreter all
  17 runnable tests pass and the 14 PyTorch-bound tests skip explicitly. The review
  provider passes 1,892 optimized and ASan/UBSan/LSan native checks, including
  direct scaled central differences over all 79 differentiable input/parameter
  coordinates, plus 59,161 all-operation descriptor, alias, domain, Philox
  multi-block overflow, failure-atomicity, and independent x87/MXCSR status checks
  in both builds. A separate 449-check optimized and sanitized integration harness
  dispatches every primitive through real I2 f32 and I1 exact-i64 borrows, proves
  active-borrow mutation/destruction blocking and release, and binds backward
  contributions to I2 gradient metadata and atomic plans. The gate binds all 32
  bits of three official Random123 vectors to the native implementation through a
  19-check test-only hook, passes 76 separate exact-bit/hash checks across every
  primitive and VJP output, rejects 14 compiled source mutations through the bit
  and frozen suites, and verifies exact archive/symbol/source/IR isolation, the
  unchanged K1 baseline, deterministic fresh builds, and all 68 operation/row
  combinations.
  The negative harness exposed separate x87/MXCSR exception-status contamination
  in the initial `fexcept_t` restoration; complete `fenv_t` snapshot/restore fixes
  it without changing arithmetic. A fixed private transport compiles with
  the pinned Eshkol compiler and produces byte-identical binaries and output across
  two runs; its native bridge is also covered by ASan/UBSan/LSan. These are local
  implementation results on the explicitly unsupported CachyOS/LLVM 22
  compatibility host, not published production capability evidence. N2 exact-head
  Ubuntu 22.04 / LLVM-Clang 21.1.8 CI and independent N2-R review remain pending.
- **Dependencies / retest:** merged I2 supplies physical owned dense zero-offset
  CPU f32 storage, synchronous K1 borrows, exact P1 handle/tie identity, gradient
  metadata, and atomic plans without an N2-facing contract deviation. Accepted I1
  supplies exact i64 views for embedding IDs and dropout state. N2's focused gate
  must retain I1/I2 borrow/lifetime, gradient-plan, K1/Q0 regression, and sanitizer
  evidence at every review head. RMSNorm and SwiGLU remain MOD5.
- **Reference:** [issue #47](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/47);
  [change-required decision](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5526909956);
  [corrected integration proposal](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5527059596);
  [corrected issue proposal](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/47#issuecomment-5527059832);
  [accepted contract and binding clarifications](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5527945682);
  [I2 PR #53](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/53);
  [I2-R exact-head approval](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/53#issuecomment-5556348520);
  [I2 supported CI run 33988137736](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33988137736);
  [I2-R requested changes](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/53#issuecomment-5526955464);
  [I2 supported CI run 33711937185](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33711937185);
  [upstream LayerNorm AD evidence](https://github.com/tsotchke/eshkol/issues/551#issuecomment-5526758405).

## N3K — bounded diagnostic numerical provider

- **Status:** accepted and complete for the bounded N3K workstream. Tracking [#66](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/66), parent #1; implementation base `231f9354f14389db15faac7820ef23dc038ae941`.
- **Direction and contract:** [accepted profile/prerequisite split](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5584217290), [exact proposal](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5584346444), [clarifications](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5584381218), and [explicit acceptance/corrections](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5584383610). Audit #65 at `a97c1095b542f1f86f3273139d39af2d8ae47e93` was fetched read-only, not merged. `docs/N3K_PRIMITIVES.md` records the exact accepted interface and reproduction commands.
- **Scope:** sole `et_n3k_kernel_provider_v1`, ABI 1.0, one-member explicitly selected archive, nine distinct capabilities and 31 exact operation/row pairs. Includes bounded embeddings, four bias-free linears, GELU, residuals, materialized layouts/inverse VJPs, ordered sums, and per-matrix Philox uniform initialization. Existing N2/A2/K1 source/header/schema/evidence and artifacts remain unchanged. No D2/O2 APIs, public model/lifecycle/transport, ingress, scalar reduction, checkpoint/trainer, upstream compiler changes, or provider composition.
- **Validation:** full focused gate passes 23 Python oracles, 15,227 primitive checks/3,248 actual-native central derivatives, 12,944 composition/initializer checks, 8,328 negative/atomicity checks, and 31,458 owned I1/I2 integration checks. All 67,957 native checks plus the private bridge pass ASan/UBSan/LSan with `detect_leaks=1`. Actual compiler dependency closure, exact symbols/archive, immutable predecessor hashes/K1 baseline, fresh artifact determinism, C/C++ ABI and two fresh pinned-Eshkol private AOT binaries/stdout pass.
- **Independent review:** three Astra-high reviewers found no unresolved numerical/schema, initializer/composition, or ownership/fenv/packaging blocker. Added invalid-ID, ordinary-input-alias, and repeated-scatter overflow tests in response to negative-review findings; independently sanitized again. Maximum primitive reference absolute error was `7.62939453e-06` on the local compatibility host.
- **Integration acceptance (2026-09-08):** [N3K-R approved](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/67#issuecomment-5587426332) exact head `39a79d2d6a87f9759b828082ac86fd66c67e2cf0` after primary and three independent Astra-high reviews. [Supported run 34222643674](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/34222643674) passed the full matrix, build, smoke and benchmark. CI checked out synthetic merge `b8c1768b75f54b4ac399d1d4cef0e05f68ae9f73`; its exact base/head parents and tree were verified. [PR #67](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/67) merged as `15ab04d4be6df11ca279c3eeb50b582e8163d535`. Reviewed head, CI checkout and actual merge all have tree `07cf8f8c3d9529a4df4c2f4a81e649fcc9e4bc49`.
- **Merged-main retest:** integration revalidated pinned Eshkol, configured and rebuilt K1/I1/I2/N3K, then ran `N3K_ASAN_DETECT_LEAKS=1 scripts/test-n3k.sh` with explicit non-login `/usr/bin/bash`: PASS. The 23 oracle tests, 67,957 native assertions, 3,248 primitive central derivatives, reference error `7.62939453e-06`, owned carrier checks, fresh private AOT, predecessor/ABI/source/archive isolation and ASan/UBSan/LSan match the approved evidence. This retest used explicit CachyOS/LLVM22 compatibility settings and the revalidated compiler under the `41a1` worktree; supported Ubuntu22/LLVM21 evidence is supplied by the CI run above.
- **Limits and downstream obligations:** private AOT establishes no public Eshkol model. Numeric RNG words neither authenticate initializer identity nor domain-separate streams; typed transport must validate identity. Reference unique-storage/path scheduling is test-only; production caller enforcement, whole-model publication, model wiring, and training remain downstream. No first-release, performance, GPU, general AD, JIT, or resume claim.

## 2026-09-10 — O2 current-main CI-topology refresh / PR #58

- **Decision:** merge current main `db70fe9c9bbbc6fbe221d65e5737b7486a75d77c`
  into the preserved O2 parent without rewriting either history. Accepted O2, N2,
  I2, T2, L2, and D2 implementation/evidence trees remain byte-identical to their
  respective parents. Shared build and CI metadata is a strict union: O2 remains in
  the native-numerics shard with its exact I2/T2/D2 aggregate prerequisites, all
  eight blocking suites and every full test command remain, and exhaustive
  acceptance retains the 240-minute outer timeout.
- **Oracle-host correction:** the first exact-head exhaustive attempt exposed a
  development-only N2 frozen-byte mismatch on a newer hosted runner image although
  the N2 source, fixture, lock, native checks, and independent numerical-gradient
  checks were unchanged. PyTorch 2.13 CPU capability dispatch is therefore pinned
  to `DEFAULT` only through an ephemeral N2 oracle-interpreter wrapper. O2, A2, and
  Q0 retain the direct pinned interpreter; no accepted N2 path, frozen byte, native
  runtime, ABI, provider, or gate is changed or skipped. CI verifies the wrapper's
  effective capability before running the unchanged byte-equality assertion.
- **Required evidence:** exact-head native-numerics and 240-minute exhaustive runs
  must both pass on Ubuntu 22.04 / LLVM-Clang 21.1.8, including unskipped N2 and O2
  oracle regeneration, O2's 6,201 adversarial checks, sanitizers/LSan, full smoke,
  and benchmark. A new independent O2-R exact-head review is required after green
  evidence. O2 and the F0 topology remain `review`; PR #58 must not be merged here.
