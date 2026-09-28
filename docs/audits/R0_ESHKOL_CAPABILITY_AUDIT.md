# R0 Eshkol capability audit

## 2026-09-28 successor pin adoption: review

The successor probe pin is `Gabriel-Kahen/eshkol@fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9`
(tree `66c21f7ec19b1b4a42199fa30ed8e0e9727021bf`). R0 validated a clean
checkout with canonical HTTPS origin, exact commit, compiler version
`Eshkol Compiler v1.3.4-evolve`, supplied F0 provenance, and compiler SHA-256
`7dd254bab761fe41142a0e3777338f41b3b9f03a5ce4c0bba419f1e2b22a99aa`.
Execution used network-disabled Ubuntu 22.04.5 / LLVM 21.1.8 Docker image
`sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6`.
The source was a separate canonical-origin checkout mounted at the F0 build's
recorded `/workspace/.deps/eshkol-src` path; both checkouts resolve to the same
commit. The build provenance and command streams are retained under
`/home/gabe/.codex/evidence/eshkol-transformer/`. The verified umbrella
`r0-successor-adoption-20260928/MANIFEST.tsv` has SHA-256
`b701db9a0b54b1d4f31114b3da1dc8daf701fac06c00903ed577fc7ea5c5f33a`;
each listed directory has its own verified `SHA256SUMS`.

| Evidence directory under that root | Observation |
|---|---|
| `r0-fe9dfd5-initial-90s-20260928` | First exploratory `activations` AOT execution printed PASS and `R0_DONE` but exited 124 at 90 seconds; the run was stopped before a full suite result. |
| `r0-fe9dfd5-activations-cold-20260928` | AOT compile and two runs exited 0, with identical PASS output; both JIT attempts exited 124 at the 180-second ceiling with no probe output. |
| `r0-fe9dfd5-tensor-core-20260928` | AOT compile and two runs exited 0 with identical output; the first JIT attempt exited 124 at 180 seconds. The redundant second JIT was interrupted. |
| `r0-fe9dfd5-activations-jit-600-20260928` | Isolated JIT exited 124 at 600 seconds, zero probe output, 79.46 seconds user CPU, 1093.01 seconds system CPU, and 62,076 KiB peak RSS. |
| `r0-81298-activations-jit-compare-20260928` | Prior pinned `81298b4` compiler also exited 124 on the same JIT program at 180 seconds with no probe output. |
| `r0-fe9dfd5-jit-smoke-20260928`, `r0-81298-jit-smoke-20260928` | The same one-line display program timed out with no output at 30 seconds on both compilers; successor with documented `ESHKOL_JIT_COMPILE_THREADS=1` also timed out at 60 seconds. |
| `r0-fe9dfd5-aot-inventory-20260928` | Interrupted AOT-only inventory: complete manifest rows for `activations` (AOT 124/124), `attention` (124/0), and `autodiff_attention` (124/124) at 30 seconds after all three compiled successfully. Each exit 124 is a failure even when stdout contains PASS and `R0_DONE`. The following compile was interrupted and is not classified. |
| `r0-aot-old-new-compare-20260928` | Controlled repeated AOT execution described below. |

For `activations` and `attention`, the comparison ran four executions of each
old and successor compiled binary in the same image with the same 30-second
limit. Every revision/probe pair exited 0 twice and 124 twice. All 16 stdout
streams matched the expected PASS output byte-for-byte within each probe;
including the timed-out runs, `activations` hashes to
`a9bc8d2a61f47583a94739d20e880138438b7ab1fb6c5017a1eb54c01269a45f`
and `attention` to
`0cf39b710c8dc5d341007047e0ed23678d2ffa78e267b090960af3d95e204f68`.
Live timed-out snapshots show a main thread in
`futex_wait` and one worker using about 98% CPU on both revisions. The old
compiler required an explicit `-l stdc++` link option to build these AOT
programs in this image; the successor compiler did not. Thus compilation inputs
were not identical, while probe source, execution image, executable deadline,
and repeated-run procedure were fixed. The liveness failure predates the
successor pin in this lane; the subsequent diagnosis appears below.

These observations support only the successful individual AOT executions and
the failure/timeout classifications above. They do not establish repeatable AOT
completion, JIT startup, AOT/JIT parity, the remaining positive and negative
probe inventory, GPU execution, or sanitizer behavior. The default harness still
runs both modes; `--aot-only` is an explicit partial-evidence mode. Full R0
successor adoption remained under review at that historical checkpoint.

### Supported-image OpenBLAS diagnosis and bounded rerun

Live thread stacks under the original 2 GiB virtual-memory cap identified
OpenBLAS worker allocation as the inherited liveness cause. In the AOT timeout,
the main thread waited in `exit → gotoblas_quit → blas_shutdown →
blas_thread_shutdown_` while a worker used CPU in `blas_memory_alloc → malloc`.
In the JIT timeout, the main thread entered the same shutdown hook from
`eshkol::pkg::run_subprocess → fork`, while two workers allocated memory.
Without the cap, the same AOT binary exited 0 in 12/12 repetitions. With the
original cap and `OPENBLAS_NUM_THREADS=1`, it exited 0 in 12/12. These are
observations of the supported image's OpenBLAS configuration, not evidence of
an Eshkol compiler defect. The diagnostic stacks and repeated-run records are
retained in `r0-runtime-liveness-20260928/` under the evidence root above.

The R0 harness now explicitly sets and records `OPENBLAS_NUM_THREADS=1` while
keeping the 2 GiB capability-run cap and the default full AOT/JIT gate. A tiny
JIT smoke then completed in 55.03 seconds with 1,056,848 KiB peak RSS under
that cap, instead of stalling at the `fork` hook. A separate 180-second JIT
ceiling accommodates focused cold JIT executions of approximately 64 and 60
seconds (retained command/output timestamps) while leaving the AOT
executable's 30-second exit deadline intact. On the
same exact successor source, compiler, and network-disabled image, focused
`activations` and `attention` each passed AOT compile, two AOT executions, and
two JIT executions. Both had byte-identical AOT/JIT output; another 12 capped
AOT runs per probe exited 0 with the same hashes. Evidence is retained in
`r0-openblas-one-20260928/{activations,attention,repeats}`.

The malformed `reshape_size_mismatch` smoke compiled and returned 0 in both
AOT and JIT, printing `#((1 2) (3 1e-323))` rather than rejecting the
three-element input reshaped to four elements. This is a real negative-probe
failure, not a reason to relax the assertion. The single-thread BLAS setting
also leaves multithreaded BLAS behavior and performance untested.

The one default full AOT/JIT inventory ran for 79 minutes 11 seconds in the
network-disabled supported image with a four-hour outer bound. It completed
183 command rows and exited 1 with `failures=38`; each bounded capability run
retained the 2 GiB cap, a 30-second AOT exit limit, and a separate 180-second
JIT limit. The 38 count includes command failures and assertions, not 38
distinct probes. `r0-openblas-one-20260928/full/results/` retains the exact
manifest, streams, assertion and parity logs. Its failing groups are:

| Group | Observed result | Prior `81298b4` status |
|---|---|---|
| `autodiff_attention` | `FAIL attention gradient` in both AOT and JIT repeats | Same failure in controlled old-pin AOT repeats and JIT |
| `autodiff_gradient_tensor` | FAIL marker in both AOT and JIT repeats | Untested |
| `autodiff_layer_norm` | Exit 1 and `expected tensor or numeric vector, got vector` in both modes | Untested |
| `rms_norm` | AOT compile and JIT report unknown `rms-norm` function | Untested |
| `rng` | AOT/JIT stdout differs despite completion markers | Untested |
| Negative `broadcast_shape_mismatch`, `file_missing` | AOT and JIT terminate with signal 139, rather than actionable rejection | Untested |
| Negative `indexed_cross_entropy_targets`, `negative_dimension` | Malformed input succeeds in AOT and JIT | Untested |
| Negative `reshape_size_mismatch` | Malformed input succeeds in AOT and JIT | Same incorrect success and output in controlled old-pin AOT repeats and JIT |

The other 27 positive groups passed their harness assertions. The three
remaining negative groups (`embedding_index_out_of_bounds`,
`index_out_of_bounds`, `matmul_shape_mismatch`) rejected the input with
diagnostics in both modes. The controlled old-pin comparison used the same
image, 2 GiB cap, and `OPENBLAS_NUM_THREADS=1`; old AOT compilation additionally
needed `-l stdc++`. Only the attention-gradient and reshape failures have
old-pin executable classification. All other failing rows are successor
observations with old-pin status untested.

The verified evidence umbrella is
`/home/gabe/.codex/evidence/eshkol-transformer/r0-openblas-one-20260928/MANIFEST.tsv`
(SHA-256 `926cbaee57976e4ee37710d7b32bfb91ca715507a6e2dd6c081adafa1de2d30c`);
it includes focused runs, 24 additional AOT repetitions, the full inventory,
and the old-pin comparison. The diagnostic stack evidence has its own verified
seal, referenced in that manifest. These failures prevent R0 successor
adoption. GPU behavior, sanitizer behavior, and multithreaded BLAS behavior
remain untested.

## Historical baseline: scope and evidence state

The original R0 report used only the canonical Eshkol repository:

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

At that historical baseline, the full canonical suite and an independent review
run had not established the remaining rows.
