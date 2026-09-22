# Development roadmap

Status values: `planned`, `active`, `blocked`, `review`, `complete`.

## Parallel execution strategy

Work proceeds in dependency-aware waves. Within a wave, workstreams may run in
parallel in isolated worktrees. Contracts merge before downstream implementation.

## Wave 0 — contracts and verification foundation

| ID | Workstream | Depends on | Acceptance evidence | Status |
|---|---|---|---|---|
| F0 | Package layout, build/test entry points, CI matrix | — | [Accepted CI-E2 full-coverage engine](CI_EFFICIENCY.md): all 16 suites and 23 commands, separate native optimizer, six disjoint C2 groups, canonical build/smoke/benchmark, audited prerequisite plans, strict original-evidence reuse, supported exact-tree CI 35407378830, independent approval, PR #82 merge, and successful 18-second main reuse | complete |
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

The accepted [CI-E2 follow-up](CI_EFFICIENCY.md#ci-e2--native-critical-path-and-main-push-reuse)
is tracked separately in issue #81. It changes CI scheduling/evidence selection,
not the accepted Wave 2 runtime scope, and starts no Wave 3 task.

Wave 2 is complete within the bounded component contracts below. Wave 3 resumed
on September 19 at 04:30 EDT by user direction; M3T and bounded M3 model composition
are accepted. Downstream evaluation, generation and training require the remaining
composition contracts. C2 completion does not prove a live trainer trajectory or
generation, and Wave 3 remains incomplete.

| ID | Workstream | Depends on | Acceptance evidence | Status |
|---|---|---|---|---|
| P1L | [Release-capable provider/state ownership correction](P1_MODULE_STATE.md) | P1, E1B, C1, T1 | Provider 2.0 exact-once clone ownership, explicit idempotent state release, scoped read-only state-backed handles, callback-defect/failure cleanup, exact P1/C1/T1 aggregate manifests, sanitizers, deterministic fresh-AOT negatives, and supported CI | complete |
| I2 | Shared dense CPU-f32 tensor, P1 value/gradient, and atomic-mutation substrate | P1L, K1, Q0, E1/E1B, R0 | [ABI 1.0 exact-bit storage, borrowed K1 views, P1-bound accumulated gradients, whole-batch preflight/commit, sanitizers, packaging negatives, and deterministic AOT evidence](I2_F32_TENSOR.md) | complete |
| K2 | [Process-local A0 capability facade over exact K1/I2 discovery](K2_CAPABILITY_FACADE.md) | A0, K1, E1/E1B, I2 | Twelve unchanged A0 operations; exact 11-row audited report; 59-global/53-export aggregate; independent approval, supported blocking/exhaustive CI, identical-tree merge and focused merged-main retest | complete |
| T2 | [Deterministic BPE training and streaming encode/decode](BPE_TOKENIZER_FORMAT.md) | T1, D1 | Canonical repeated artifacts and order/partition-invariant merges; all-byte/raw/strict/F0-F4/special whole-stream parity; exact 65,536-byte/73,728-ID ceilings including 73,728 one-ID chunks and one-over negatives; compiled parser/D1 corrupt-data negatives; deterministic localized object/archive/evidence/AOT boundary suite; exact aggregate manifests; production Python isolation; independent review; supported CI; and merged-main retest | complete |
| D2 | [Memory-bounded shard loader, batching, packing and cursor state](D2_SHARD_LOADER.md) | D1, T1, Q0 | Accepted ten-key config and 11-operation surface; exact CPU `17*N*T` carrier/lifetime; packed/unpacked shift and bool masks; unbiased bounded-window shuffle; `ESHKDCU1` exact resume; corrupt/resource/sanitizer/AOT/58-global/52-export gates; independent D2-R, supported CI, identical-tree merge, and merged-main retest | complete |
| N2 | [Embedding, linear, normalization, activations, dropout, residuals](N2_PRIMITIVES.md) | P1L, K1, Q0, I1, I2 | Exact-row carrier-backed forward/VJP parity, scaled numerical gradients, Philox bit determinism, failure atomicity, native lifetime, private AOT, ABI/isolation manifests, sanitizers, independent review, and supported CI | complete |
| A2 | Causal attention, masks, RoPE and KV-cache primitives | P1, K1, Q0 | Masking, forward/backward and cache parity tests | complete |
| L2 | [Fused indexed token cross-entropy](L2_INDEXED_CROSS_ENTROPY.md) | K1, Q0 | Explicit carrier-neutral K1 provider, stable per-token f32 loss, direct-backward/oracle/finite-difference parity, adversarial failure atomicity, deterministic Eshkol AOT, sanitizer and isolation gates | complete |
| O2 | [AdamW, parameter groups, clipping, accumulation and schedules](O2_OPTIMIZER.md) | P1L, Q0, I2 | Exact 1,365-parameter/group boundary; accumulated-gradient, clipping, AdamW and schedule parity; atomic step/load; releasable logical-state continuation; exact 53-global/six-wrapper aggregate; 6,201 adversarial checks, sanitizers, independent review, supported blocking/exhaustive CI, identical-tree merge and bounded merged-main retest | complete |
| C2 | [Detached full training-state checkpoint schema](C2_TRAINING_STATE.md) | C1, D2, O2, X1, K2 | Accepted C2 1.0 component continuation and atomic-file publication: exact 81-global/75-export/81-string boundary, authenticated carriers, exact-bit LOAD/SAVE ownership, failure atomicity, deterministic packaging, flat 1,024/8,192 root retention, and final joint 64-tensor runs at 521,500/521,576 KiB below 524,288 KiB. PR #79 merged as `cbd0929`; PR #77 merged to main as `913cdf4097db09d6c33769e9b0968c01ce0e1f55`; reviewed, tested, and merged states share tree `512a3355cea79583d73b49f690a46478fb1c772c`. Supported run 35393213200 passed all 15 suites/23 commands plus smoke/benchmark, and acceptance 35401088338 reused the verified exact tree successfully. TR3 retains joint live restore/full trajectory and G3 generation proof | complete |

## Wave 3 — first complete language model

| ID | Workstream | Depends on | Acceptance evidence | Status |
|---|---|---|---|---|
| N3K | [Bounded diagnostic numerical primitives and initializer](N3K_PRIMITIVES.md) | N2, A2, K1, I1, I2, Q0 | 31 exact operation/row pairs; independent VJP/bit/negative/carrier tests, private AOT and sanitizers; independent N3K-R approval, supported CI 34222643674, PR #67 merge and focused merged-main retest | complete |
| M3 | [Fixed-profile decoder-only model composition](M3_MODEL.md) | M3T, N2, A2, P1 | #85; bounded immutable-output/graph and 14-parameter contribution contract; independent M3-R approval 5747957422, supported full CI 35465971406 (18 suites / 25 commands), PR #88 merge `aa9e78f` at reviewed tree `7d81855be5ad35a44e74a708642668f8d789ebff`, verified main evidence reuse and focused merged-main native/oracle/package/public-AOT/lifetime checks. [Retention evidence](M3_RETENTION_GATE.md) preserves the original timeout and cumulative costs; no full-training or flat-memory claim | complete |
| M3T | [Bounded production Eshkol transport/lifetime contract](M3T_TRANSPORT_PROPOSAL.md) | N3K, N2, A2, I1, I2, P1L, E1B | #68; [accepted implementation and evidence](M3T_TRANSPORT.md); restricted 85-global sibling; independent M3T-R approval, supported full CI 35441033357, identical-tree PR #84 merge `f501609`, and focused merged-main retest; no model-completion claim | complete |
| L3S | [Bounded masked-objective reduction and explicit seeds](L3S_MASKED_OBJECTIVE.md) | L2, K1, Q0; I1/I2 for evidence | #87; accepted six-operation CPU-f32 `[1,2]` contract; independent L3S-R approval 5770338523, supported full CI 35531132634 (18 suites / 26 commands), PR #91 merge `fe249316`; only intervening accepted G3 Markdown differs from the tested candidate; focused merged-main gate and leak checks passed; fresh combined-main CI 35680925863 passed all 18 suites / 26 commands at exact merge/tree `fe249316`/`902d4645` | complete |
| E3 | [Loss, perplexity, token accuracy and validation runner](E3_EVALUATION_PROPOSAL.md) | M3/M3T, D2, L2/L3S; shared guard implementation, E3 frame, P1/D2 restore and bool metric seams pending | #99; [E3-CG #101 exact shared guard/fixed14 design](E3_SHARED_CALL_CONTRACT.md) accepted by [root decision 5771662657](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/102#issuecomment-5771662657); contract merged in PR #102; shared implementation authorized separately, runtime unaccepted; [exact E3 private consumer design](E3_PRIVATE_CONTRACT.md), including canonical P1 69-slot prerequisite, accepted by [root verdict5771845949](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/104#issuecomment-5771845949); contract merged in PR #104; independent prerequisites dispatched, implementation acceptance pending; private-first architecture accepted with conditions 5771430755/5771464954; consolidated exact design and explicit E3-PRIVATE/E3-PRIVATE-R milestones, genuine I2 outputs; optional public/weighted paths deferred, no runtime/public evaluator or completion claim | active |
| E3-D2 | [Canonical-byte identity and native idle prerequisite](e3/E3_D2_PREREQUISITE.md) | T1, D1, D2; merged PR #104 contract | #106; exact one-form hash-pinned source variant, authentic T1 admission and native idle 0/1/2; private compiled closure, lifecycle, provenance and sanitizer gates; independent review and supported exact-candidate CI required; frame-bound restoration/staging remains deferred | review |
| G3 | Greedy, temperature, top-k/top-p generation with KV cache | M3, T1, A2 | #89 [bounded design and compiled reachability](G3_GENERATION_PROPOSAL.md), decision 5748532295; exact G3-N/S ABIs accepted for implementation by decision 5751933660; [exact G3-T private-seam contract](g3/G3_T_PRIVATE_CONTRACT.md) accepted by decision 5752756205, implementation blocked on both independently approved N/S merges; end-to-end generation/save-reload acceptance remains open | active |

| G3-N | [Exact T1 forward numerical provider](G3N_PRIMITIVES.md) | K1, Q0, I1, I2, N2, N3K, A2, PR #92 | #93; ABI 1.0 six capabilities/seven operations/eleven pairs; independent G3-N-R approval 5771670331, supported full CI 35681252651 attempt 2 (19 suites / 27 commands), PR #97 merge `ca3880f` and root focused merged-head retest 5771719405. [Provenance](G3N_PRIMITIVES.md#integration-provenance) retains the initial P1 timeout/sole retry and distinct candidate/merged trees; no model/generation claim | complete |
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
