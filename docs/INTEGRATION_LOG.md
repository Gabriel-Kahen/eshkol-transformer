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

## 2026-08-28 — R0 complete-suite supplement / issue #4

- **Decision:** proposed; integration owner acceptance is pending.
- **Contract:** Capability discovery is execution-based and limited to the exact
  observed shape, dtype, backend, and differentiation case. A `gpu-*` symbol that
  executes while GPU support is disabled is CPU evidence, not device proof. Broken,
  unsupported, and untested cases must fail explicitly at the project boundary; no
  hidden CPU/scalar fallback is authorized.
- **Evidence:** The canonical pinned compatibility-lane suite retained 183 command
  records and completed with 38 intentional audit failures. Eight supplemental
  probes retained 48 records; seven passed and corrupt tensor/model rejection failed
  all four executions. Every runnable positive ran twice under AOT and JIT. An
  independent fresh-scratch sample corroborated supported and broken cases. See
  `docs/audits/R0_ESHKOL_CAPABILITY_AUDIT.md` and
  `docs/evidence/r0/2026-08-28/`.
- **Dependencies / retest:** K1 may consume only exact observed rows and must reject
  absent capabilities. P1/N2/A2 remain gated on per-operation gradient evidence; C1
  remains gated on a versioned, checksummed persistence format. GPU, physical
  reduced-precision storage, and the supported Ubuntu/LLVM-21 lane remain untested.
- **Reference:** issue #4; proposed contract coordination in issue #1; Eshkol-core
  issues tsotchke/eshkol#549–#553;
  [PR #16](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/16).
