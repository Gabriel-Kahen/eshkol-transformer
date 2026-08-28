# R0 Eshkol capability audit

## Scope and evidence state

This report uses only the canonical Eshkol repository:

- repository: `https://github.com/tsotchke/eshkol.git`
- revision: `90cbd7130f47b8184bcc77b8d5c1b0026da980de`
- compiler identity: `Eshkol Compiler v1.3.4-evolve`
- compiler SHA-256 in this run:
  `caa295b19a6e9388963aa0def99dade63656d2dcbffccad421bd1daaa1db3750`

The bounded run proves only the `tensor_core` cases listed below. It used F0's
existing canonical build on CachyOS with LLVM 22.1.6. This is an explicitly
unsupported compatibility lane, not the supported Ubuntu 22.04/LLVM 21.1.8 lane.
No GPU was visible. The complete suite, sanitizer lane, supported-host lane, and GPU
lane have not run, so every capability outside the observed subset remains
`untested-with-reason`.

## Observed canonical evidence

The retained run directory is
`/home/gabe/.cache/eshkol-r0-canonical-final.8XyQEq/tensor-results`. It is local
evidence, not a checked-in fixture. The harness validated the supplied F0 provenance
repository, commit, source path, and binary hash, then retained an exact copy as
`f0-build-provenance.tsv`. `manifest.tsv` records six successful commands:

| Phase | Exit | Evidence |
|---|---:|---|
| compiler help | 0 | supplied compiler executed |
| `tensor_core` AOT compile | 0 | executable produced |
| AOT run 1 | 0 | all seven checks passed and completion marker printed |
| AOT run 2 | 0 | stdout byte-identical to run 1 |
| JIT run 1 | 0 | stdout byte-identical to AOT |
| JIT run 2 | 0 | stdout byte-identical to the first JIT run |

All four execution streams have SHA-256
`6c7d571e519fd68a308a1abd504c9279c59e0379dfd3e4fab625ed41363e8a3b`.
`summary.txt` reports `failures=0`; both assertion and parity failure logs are
empty. The output demonstrates rank/shape for 2-D and 3-D tensors, first/last
index reads, transpose values, reshape values, and mutable vector storage for the
specific f64-valued program. It does not prove storage dtype, contiguity, ownership,
view aliasing, arbitrary ranks, or malformed-input behavior.

## Capability matrix

| Capability | Classification | Evidence or reason |
|---|---|---|
| 2-D/3-D construction, rank/shape, indexing | observed-supported (compatibility lane) | `tensor_core.esk`, AOT and JIT, exact repeat parity |
| Reshape and transpose values | observed-supported (compatibility lane) | known values in `tensor_core.esk`; ownership/aliasing untested |
| Mutable vector storage | observed-supported (compatibility lane) | one `vector-set!`/`vector-ref` case; tensor mutation untested |
| AOT/JIT parity for `tensor_core` | observed-supported (compatibility lane) | byte-identical stdout across two AOT and two JIT executions |
| Contiguity, ownership, lifetimes, view aliasing | untested-with-reason | dedicated probes and sanitizers not executed |
| Broadcasting and malformed shape/index handling | untested-with-reason | positive and negative suite not executed |
| Integer/boolean/f32/f16/bf16 storage | untested-with-reason | numeric values do not establish storage dtype |
| f64 storage and precision bounds | untested-with-reason | f64-looking literals do not prove runtime storage representation |
| CPU backend selection/identity | untested-with-reason | execution succeeded, but backend identity was not observed |
| GPU availability, selection, transfer, or execution | untested-with-reason | no GPU visible and no device telemetry |
| Dot, matmul, batched/broadcasted matmul | untested-with-reason | probes not executed |
| Embedding gather/scatter-add gradient | untested-with-reason | probes not executed |
| LayerNorm, RMSNorm, activations, causal mask, RoPE, attention | untested-with-reason | probes not executed |
| Indexed loss prerequisite | untested-with-reason | probes not executed |
| Forward/reverse AD and gradient accumulation | untested-with-reason | probes not executed; no inference from documentation |
| RNG and fresh-process determinism | untested-with-reason | probes not executed |
| File I/O, atomic rename, safe serialization, checksums | untested-with-reason | probes not executed |
| Long-loop memory behavior | untested-with-reason | bounded RSS/sanitizer evidence absent |
| Supported Ubuntu 22.04/LLVM 21.1.8 lane | untested-with-reason | this run used CachyOS/LLVM 22.1.6 |

## Source inspection versus executable proof

No source-inspection statement is promoted to a runtime classification. The checked-in
probe inventory reflects APIs worth testing, but an API name, implementation file,
compiler diagnostic, or documentation claim is not support evidence. A guessed-syntax
failure also remains inconclusive. The supported lane has no R0 execution evidence yet.

## Core-upstream versus K1 decision matrix

No row authorizes a fallback. Until its evidence is complete, the dependent feature
must remain explicitly unsupported.

| Gap | Primary disposition | Trigger and boundary |
|---|---|---|
| Core tensor shape/index/ownership defect | Eshkol-core upstream issue | File a minimal canonical reproducer for a wrong result, crash, or lifetime defect; block P1 |
| Batched/broadcasted matmul absent | Versioned K1 candidate | Use an isolated native op only after proving no reachable core API; wrong advertised core matmul goes upstream |
| Embedding gather/scatter-add absent | Versioned K1 candidate | Extension must specify repeated-index accumulation and deterministic gradient behavior |
| LayerNorm/RMSNorm absent | Versioned K1 candidate | Keep shape/axis/epsilon explicit; incorrect reachable core norm goes upstream |
| Indexed loss primitive absent | Versioned K1 candidate | Isolate indexed gather/reduce and backward; never materialize a hidden scalar fallback |
| Reverse AD wrong, approximate, or silently finite-difference | Eshkol-core upstream issue | This is compiler/runtime semantics; block dependent training until executable gradient checks pass |
| Device identity/selection not observable | Eshkol-core upstream issue | Core must expose truthful device state; K1 may expose only its own explicit kernel/backend identity |
| AMD execution primitive absent | Versioned K1 candidate | HIP/rocBLAS extension must fail explicitly when unavailable and provide external observed-device proof |
| True f32 storage unavailable | Eshkol-core upstream issue | Do not relabel f64 storage; a distinct K1 tensor type would require a separate public contract/version |
| f16/bf16 unavailable | Deferred, then K1 candidate | Only after true storage and accumulation semantics are specified; no precision emulation |
| Checksummed/atomic checkpoint primitives absent | Versioned K1 candidate | Non-executable format, explicit errors, and atomic replacement are extension-scoped |
| Reachable primitive silently falls back to CPU/scalars | Eshkol-core upstream issue | Treat as a defect and block the feature; K1 must never mask it |

## Exact rerun commands

Representative canonical rerun using the existing F0 build:

```bash
R0_RUN_ROOT="$(mktemp -d /home/gabe/.cache/eshkol-r0-canonical.XXXXXX)"
/usr/bin/bash probes/r0/run.sh \
  --eshkol-source /home/gabe/.codex/worktrees/49f7/eshkol-transformer/.deps/eshkol-src \
  --existing-build /home/gabe/.codex/worktrees/49f7/eshkol-transformer/.deps/eshkol-build-minimal \
  --work-dir "$R0_RUN_ROOT/tensor-work" \
  --results-dir "$R0_RUN_ROOT/tensor-results" \
  --probe tensor_core \
  --run-timeout 120 \
  --compile-timeout 360
```

Full reproducible build-and-suite mode on the supported lane omits
`--existing-build` and supplies the pinned LLVM executable:

```bash
R0_RUN_ROOT="$(mktemp -d /home/gabe/.cache/eshkol-r0-full.XXXXXX)"
ESHKOL_LLVM_CONFIG=/usr/lib/llvm-21/bin/llvm-config \
/usr/bin/bash probes/r0/run.sh \
  --eshkol-source /absolute/clean/tsotchke-eshkol-at-90cbd713 \
  --work-dir "$R0_RUN_ROOT/work" \
  --results-dir "$R0_RUN_ROOT/results"
```

R0 remains incomplete until the full canonical suite and an independent review run
establish the remaining rows. This report does not update `docs/ROADMAP.md`.
