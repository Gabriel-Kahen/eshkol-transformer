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
