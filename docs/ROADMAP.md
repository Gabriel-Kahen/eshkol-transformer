# Development roadmap

Status values: `planned`, `active`, `blocked`, `review`, `complete`.

## Parallel execution strategy

Work proceeds in dependency-aware waves. Within a wave, workstreams may run in
parallel in isolated worktrees. Contracts merge before downstream implementation.

## Wave 0 — contracts and verification foundation

| ID | Workstream | Depends on | Acceptance evidence | Status |
|---|---|---|---|---|
| F0 | Package layout, build/test entry points, CI matrix | — | Clean configure plus smoke test on supported Linux environment | complete |
| A0 | Public API, shapes, dtype/device, error and ownership contracts | — | Reviewed specification and compile-only API fixtures | complete |
| R0 | Audit Eshkol tensor/autodiff/runtime capabilities | — | Executable capability probe and gap report with no inferred support | complete |
| Q0 | Test harness and frozen reference-oracle format | — | [Deterministic harness, frozen fixture, and passing compiled parity](Q0_VALIDATION.md) | complete |
| B0 | Benchmark and memory-measurement harness | F0 | [Versioned/checksummed definition, report schema, and smoke benchmark](BENCHMARK_FORMAT.md) | complete |

## Wave 1 — independent foundations

| ID | Workstream | Depends on | Acceptance evidence | Status |
|---|---|---|---|---|
| E1 | [Shared structured-error construction and accessors](ERROR_CONTRACT.md) | A0, Q0 | 112 construction/accessor/ownership/forgery/negative checks, repeated AOT/JIT parity, and declared-public-boundary gates | complete |
| E1B | [Separately compiled fixed-arity raise-only consumer boundary](E1B_CONSUMER_BOUNDARY.md) | E1 | Fresh-cache mixed-facade AOT identity plus closure, arity, localization, exact defined/undefined-symbol allowlists, determinism, and artifact evidence | complete |
| P1 | [Named parameter tree, buffers, modules, state dictionaries](P1_MODULE_STATE.md) | A0, R0, Q0, E1 | 178 structural/ownership/atomicity checks, 183 native identity/failpoint/cross-role checks, 15 test-only registry-atomicity checks, fresh-cache source/object/AOT public-boundary negatives, sanitizers, and deterministic AOT | review |
| T1 | Byte tokenizer, special tokens, fingerprints and format | A0, Q0, E1, X1 | All-byte/UTF-8 round trips and deterministic format tests | planned |
| D1 | Versioned token-shard format and corpus writer | A0, Q0, E1 | Checksums, malformed-input tests and deterministic shard fixtures | planned |
| K1 | Native kernel ABI/capability layer | A0, R0, Q0 | [Versioned ABI, canonical unverified baseline, and 596 conformance/unsupported/malformed-call checks](K1_KERNEL_ABI.md) | complete |
| I1 | Exact signed-i64 dense CPU tensor container and bounded K1 storage-copy provider | A0, R0, Q0, E1, K1 | [ABI 1.0 ownership/layout contract, exact boundary round trips, malformed/failure-atomic checks, sanitizers, and canonical-pin AOT interop](I1_I64_TENSOR.md) | complete |
| C1 | Versioned checkpoint container and atomic I/O | A0, P1, Q0 | Tensor round trip, corruption rejection and atomic-write tests | planned |
| X1 | [Declarative configuration and resolved-run manifests](CONFIG_FORMAT.md) | A0, Q0, E1, E1B | 111 native semantics, 11 reference/isolation checks, source-before-overlay admission, deterministic fresh object/AOT builds, both E1 import orders, exact 12-export/95-undefined artifact admission, and private-source/symbol leakage negatives; supported CI pending | review |

## Wave 2 — model and training primitives

| ID | Workstream | Depends on | Acceptance evidence | Status |
|---|---|---|---|---|
| T2 | Deterministic BPE training and streaming encode/decode | T1, D1 | Merge determinism, byte fallback and corpus round trips | planned |
| D2 | Memory-bounded shard loader, batching, packing and cursor state | D1, T1, Q0 | Shifted targets, masks, deterministic shuffle and exact resume | planned |
| N2 | Embedding, linear, normalization, activations, dropout, residuals | P1, K1, Q0 | Forward and gradient parity for every operation | planned |
| A2 | Causal attention, masks, RoPE and KV-cache primitives | P1, K1, Q0 | Masking, forward/backward and cache parity tests | planned |
| L2 | Fused indexed token cross-entropy | K1, Q0 | Stable per-token loss and direct-backward parity tests | planned |
| O2 | AdamW, parameter groups, clipping, accumulation and schedules | P1, Q0 | Reference update parity and serializable-state tests | planned |
| C2 | Full training-state checkpoint schema | C1, D2, O2, X1 | Model/optimizer/scheduler/RNG/cursor exact-resume test | planned |

## Wave 3 — first complete language model

| ID | Workstream | Depends on | Acceptance evidence | Status |
|---|---|---|---|---|
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
