# Development roadmap

Status values: `planned`, `active`, `blocked`, `review`, `complete`.

## Parallel execution strategy

Work proceeds in dependency-aware waves. Within a wave, workstreams may run in
parallel in isolated worktrees. Contracts merge before downstream implementation.

## Wave 0 — contracts and verification foundation

| ID | Workstream | Depends on | Acceptance evidence | Status |
|---|---|---|---|---|
| F0 | Package layout, build/test entry points, CI matrix | — | Eight full-coverage parallel suites, separate T2 runtime/boundary jobs, explicit tokenizer/D2 prerequisites, test-only prose selection, and early Python-isolation checks; corrected hosted run pending, serial nightly/manual exhaustive acceptance retains one outer build | review |
| A0 | Public API, shapes, dtype/device, error and ownership contracts | — | Reviewed specification and compile-only API fixtures | complete |
| R0 | Audit Eshkol tensor/autodiff/runtime capabilities | — | Executable capability probe and gap report with no inferred support | complete |
| Q0 | Test harness and frozen reference-oracle format | — | [Deterministic harness, frozen fixture, and passing compiled parity](Q0_VALIDATION.md) | complete |
| B0 | Benchmark and memory-measurement harness | F0 | [Versioned/checksummed definition, report schema, and smoke benchmark](BENCHMARK_FORMAT.md) | complete |

## Wave 1 — independent foundations

| ID | Workstream | Depends on | Acceptance evidence | Status |
|---|---|---|---|---|
| E1 | [Shared structured-error construction and accessors](ERROR_CONTRACT.md) | A0, Q0 | 112 construction/accessor/ownership/forgery/negative checks, repeated AOT/JIT parity, and declared-public-boundary gates | complete |
| E1B | [Separately compiled fixed-arity raise-only consumer boundary](E1B_CONSUMER_BOUNDARY.md) | E1 | Fresh-cache mixed-facade AOT identity plus closure, arity, localization, exact defined/undefined-symbol allowlists, determinism, and artifact evidence | complete |
| P1 | [Named parameter tree, buffers, modules, state dictionaries](P1_MODULE_STATE.md) | A0, R0, Q0, E1, E1B | One 24-definition/18-export registry-owning package, 419 structural/ownership/atomicity checks, 405 native identity/failpoint/cross-role checks, 169 registry/error-mapping/topology/name-bound checks, exact manifests, import-both/fresh-cache negatives, sanitizers, deterministic AOT, and supported CI | complete |
| T1 | [Byte tokenizer, special tokens, fingerprints and format](TOKENIZER_FORMAT.md) | A0, Q0, E1/E1B, I1, X1, P1, D1, C1 | Exhaustive byte/UTF-8 and special-token semantics, canonical-format/fingerprint negatives, all-registry retention/complexity evidence, exact-max and exact/one-over optional-header public-AOT RSS/time checks, exact-i64 shell lifetime and sanitizers, C1 policy/atomic-I/O checks, exact 47-global aggregate boundary, deterministic fresh-AOT evidence, production Python isolation, independent approval, supported CI, and merge-commit retest | complete |
| D1 | [Versioned token-shard format and corpus writer](TOKEN_SHARD_FORMAT.md) | A0, Q0, E1 | Deterministic/reference/corruption/boundary groups; checked partial-write/ENOSPC/EIO/close-failure cleanup with no visible manifest; summary mutation/vector-copy/constructor/receiver-forgery; fresh-cache source/object/AOT plus symbol/depfile/crafted-link public-boundary gates; and supported CI | complete |
| K1 | Native kernel ABI/capability layer | A0, R0, Q0 | [Versioned ABI, canonical unverified baseline, and 596 conformance/unsupported/malformed-call checks](K1_KERNEL_ABI.md) | complete |
| I1 | Exact signed-i64 dense CPU tensor container and bounded K1 storage-copy provider | A0, R0, Q0, E1, K1 | [ABI 1.0 ownership/layout contract, exact boundary round trips, malformed/failure-atomic checks, sanitizers, and canonical-pin AOT interop](I1_I64_TENSOR.md) | complete |
| C1 | [Versioned checkpoint container and atomic I/O](CHECKPOINT_FORMAT.md) | A0, P1, Q0, E1/E1B | 245 logical-state/ownership/error checks, 1012 bounded adversarial parser/validator cases, deterministic repeated AOT bytes, native failpoint/ABI/sanitizer gates, atomic old-or-new publication evidence, independent approval, supported CI, and merge-commit retest | complete |
| X1 | [Declarative configuration and resolved-run manifests](CONFIG_FORMAT.md) | A0, Q0, E1, E1B | 111 native semantics, 11 reference/isolation checks, source-before-overlay admission, deterministic fresh object/AOT builds, both E1 import orders, exact 12-export/95-undefined artifact admission, private-source/symbol leakage negatives, and supported CI | complete |

## Wave 2 — model and training primitives

| ID | Workstream | Depends on | Acceptance evidence | Status |
|---|---|---|---|---|
| P1L | [Release-capable provider/state ownership correction](P1_MODULE_STATE.md) | P1, E1B, C1, T1 | Provider 2.0 exact-once clone ownership, explicit idempotent state release, scoped read-only state-backed handles, callback-defect/failure cleanup, exact P1/C1/T1 aggregate manifests, sanitizers, deterministic fresh-AOT negatives, and supported CI | complete |
| I2 | Shared dense CPU-f32 tensor, P1 value/gradient, and atomic-mutation substrate | P1L, K1, Q0, E1/E1B, R0 | [ABI 1.0 exact-bit storage, borrowed K1 views, P1-bound accumulated gradients, whole-batch preflight/commit, sanitizers, packaging negatives, and deterministic AOT evidence](I2_F32_TENSOR.md) | complete |
| K2 | [Process-local A0 capability facade over exact K1/I2 discovery](K2_CAPABILITY_FACADE.md) | A0, K1, E1/E1B, I2 | Twelve unchanged A0 operations; exact 11-row audited report; D2-style caller-region identities; 59-global/53-export aggregate; native diagnostic, allocation, AOT, manifest, fork and retention gates | review |
| T2 | [Deterministic BPE training and streaming encode/decode](BPE_TOKENIZER_FORMAT.md) | T1, D1 | Canonical repeated artifacts and order/partition-invariant merges; all-byte/raw/strict/F0-F4/special whole-stream parity; exact 65,536-byte/73,728-ID ceilings including 73,728 one-ID chunks and one-over negatives; compiled parser/D1 corrupt-data negatives; deterministic localized object/archive/evidence/AOT boundary suite; exact aggregate manifests; production Python isolation; independent review; supported CI; and merged-main retest | complete |
| D2 | [Memory-bounded shard loader, batching, packing and cursor state](D2_SHARD_LOADER.md) | D1, T1, Q0 | Accepted ten-key config and 11-operation surface; exact CPU `17*N*T` carrier/lifetime; packed/unpacked shift and bool masks; unbiased bounded-window shuffle; `ESHKDCU1` exact resume; corrupt/resource/sanitizer/AOT/58-global/52-export gates; independent D2-R, supported CI, identical-tree merge, and merged-main retest | complete |
| N2 | [Embedding, linear, normalization, activations, dropout, residuals](N2_PRIMITIVES.md) | P1L, K1, Q0, I1, I2 | Exact-row carrier-backed forward/VJP parity, scaled numerical gradients, Philox bit determinism, failure atomicity, native lifetime, private AOT, ABI/isolation manifests, sanitizers, independent review, and supported CI | complete |
| A2 | Causal attention, masks, RoPE and KV-cache primitives | P1, K1, Q0 | Masking, forward/backward and cache parity tests | complete |
| L2 | [Fused indexed token cross-entropy](L2_INDEXED_CROSS_ENTROPY.md) | K1, Q0 | Explicit carrier-neutral K1 provider, stable per-token f32 loss, direct-backward/oracle/finite-difference parity, adversarial failure atomicity, deterministic Eshkol AOT, sanitizer and isolation gates | complete |
| O2 | [AdamW, parameter groups, clipping, accumulation and schedules](O2_OPTIMIZER.md) | P1L, Q0, I2 | Exact 1,365-parameter/group boundary; accumulated-gradient, clipping, AdamW and schedule parity; atomic step/load; releasable logical-state continuation; exact 53-global/six-wrapper aggregate; 6,201 adversarial checks, sanitizers, independent review, supported blocking/exhaustive CI, identical-tree merge and bounded merged-main retest | complete |
| C2 | Full training-state checkpoint schema | C1, D2, O2, X1, K2 | Pre-freeze contract correction; blocked on K2 approval/merge. Compiled model/optimizer/scheduler/RNG/cursor component continuation and atomic file publication; TR3 retains joint live restore/full trajectory and G3 generation proof | blocked |

## Wave 3 — first complete language model

| ID | Workstream | Depends on | Acceptance evidence | Status |
|---|---|---|---|---|
| N3K | [Bounded diagnostic numerical primitives and initializer](N3K_PRIMITIVES.md) | N2, A2, K1, I1, I2, Q0 | 31 exact operation/row pairs; independent VJP/bit/negative/carrier tests, private AOT and sanitizers; independent N3K-R approval, supported CI 34222643674, PR #67 merge and focused merged-main retest | complete |
| M3 | Decoder-only GPT configuration, blocks and tied LM head | N2, A2, P1 | Full forward/gradient oracle parity | planned |
| E3 | Loss, perplexity, token accuracy and validation runner | M3, L2, D2 | Deterministic held-out metrics | planned |
| G3 | Greedy, temperature, top-k/top-p generation with KV cache | M3, T1, A2 | Seeded sampling and cache/no-cache parity | planned |
| TR3 | Trainer state machine and exact resume | M3, L2, O2, D2, C2, E3 | One-batch overfit and interrupted/resumed equivalence | planned |
| CLI3 | Corpus, tokenizer, pretrain, evaluate, generate and inspect CLIs | T2, TR3, G3, X1 | End-to-end command tests and actionable diagnostics | planned |

## Wave 4 — practical pretraining and performance

| ID | Workstream | Depends on | Acceptance evidence | Status |
|---|---|---|---|---|
| MP4 | True f32 plus f16/bf16 policy, master weights and loss scaling | TR3, K1 | Numerical and storage-size evidence; overflow recovery tests | planned |
| AC4 | Activation checkpointing and memory-bounded training | TR3 | Same update within tolerance and measured memory reduction | planned |
| AMD4 | AMD HIP/rocBLAS backend and device observability | K1, TR3, B0 | Direct device execution, parity and benchmark evidence | planned |
| PA4 | Optimized/fused attention path | A2, B0 | Parity plus measured speed/memory improvement | planned |
| DATA4 | Filtering, deduplication, provenance and dataset reports | D1, T2 | Deterministic reports and contamination/provenance metadata | planned |

## Wave 5 — ecosystem expansion

| ID | Workstream | Depends on | Acceptance evidence | Status |
|---|---|---|---|---|
| IO5 | Safe import/export for safetensors and GGUF | C1, M3 | Cross-format round trips and malformed-file tests | planned |
| MOD5 | RMSNorm/SwiGLU/GQA/MQA/ALiBi/sliding-window model variants | M3 | Per-variant parity and training smoke tests | planned |
| DIST5 | Data-parallel training and collective abstraction | TR3, K1 | Multi-process deterministic smoke and failure handling | planned |
| EVAL5 | External evaluation adapters and contamination checks | E3, T2 | Reproducible benchmark manifests | planned |
| DOC5 | Tutorials, API reference and from-scratch corpus-to-model guide | CLI3 | Fresh-environment walkthrough succeeds | planned |

## Orchestrator rules

- Maintain one integration owner and a contract-change log.
- Create implementation tasks only when their declared dependencies are merged.
- Require each task to use subagents for at least independent testing/review when the
  work is non-trivial.
- Prefer several bounded tasks over one cross-cutting task.
- Review and integrate in dependency order; rerun affected downstream gates.
- Treat Eshkol-core defects as explicit upstream issues or isolated native-extension
  work, never as silent library fallbacks.
