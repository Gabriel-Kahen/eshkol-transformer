# Integration log

This repository-side ledger mirrors contract decisions recorded in
[issue #1](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1).
Only the integration owner changes a proposed decision to `accepted` after review.

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

## 2026-08-28 — D1 / issue #18

- **Decision:** proposed; the binary v1.0 format direction is accepted with conditions
  by the integration task, while repository acceptance remains pending review.
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
  whole-directory atomicity, `fsync`, or power-loss durability claim.
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
  D1-R re-review remains pending. The historical pre-correction run is not
  acceptance evidence for the corrected implementation.
- **Dependencies / retest:** E1 issue #23 is merged through PR #27. D1 imports only
  `transformer.error_public` plus the accepted E1 core; `transformer.error_internal`
  and its three named construction helpers are absent from the data dependency graph.
  Because pinned `provide` declarations are informational, E1's already accepted
  `e1-internal-dispatch` core bridge remains technically name-reachable and requires
  upstream module-opacity retesting. T1/D2/T2/DATA4 may consume this contract only
  after integration acceptance; D2 retains all iteration, shuffle, packing,
  batching, and cursor behavior. A later streaming/I1 integration must rerun byte-
  determinism, publication, corruption, and scale tests.
- **Reference:** [issue #1](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1);
  [issue #18](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/18);
  [PR #29](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/29);
  [D1-R requested changes](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/29#issuecomment-5464425028).
