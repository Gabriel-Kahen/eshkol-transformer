# R0 Eshkol capability audit

## Result and evidence boundary

R0 is ready for review. All classifications below come from the compiled canonical
Eshkol toolchain, not README/source presence:

- repository: `https://github.com/tsotchke/eshkol.git`
- revision: `90cbd7130f47b8184bcc77b8d5c1b0026da980de`
- compiler: `Eshkol Compiler v1.3.4-evolve`
- compiler SHA-256:
  `caa295b19a6e9388963aa0def99dade63656d2dcbffccad421bd1daaa1db3750`

The canonical evidence is checked in at `docs/evidence/r0/2026-08-28/`. The full
suite retained 183 command records (184 manifest lines including the header) and
ended with `failures=38`. That nonzero result is expected audit evidence: wrong
values, explicit unsupported operations, crashes, timeouts, malformed successes,
and parity failures are counted. Eight supplemental probes retained another 48
command records. Seven passed; corrupt tensor/model rejection failed in all four
executions. A direct backend inventory and an independent representative rerun are
retained alongside them.

The earlier `Gabriel-Kahen/eshkol@746e518...` evidence was invalid for the canonical
F0 lock and has been removed from the branch tip. None of its results are used here.

## Exact environment and commands

- Host: CachyOS Linux `7.1.2-3-cachyos`, x86-64, AMD Ryzen 7 3700X, 16 logical CPUs.
- Host tools: CMake 4.3.3, Ninja 1.13.2, system LLVM 22.1.6.
- Supplied F0 compiler provenance: Clang/LLVM 22.1.6; feature profile
  `minimal-no-stdlib-no-agent-ffi-no-xla-no-quantum-no-tensorcore`.
- Direct compiler inventory: LLVM backend on; GPU/Metal/CUDA/BLAS/XLA off;
  advertised logical dtypes `f64,f32,f16,bf16,i8`.
- Hardware inventory: no NVIDIA or ROCm device visible.
- This is F0's explicitly unsupported CachyOS/LLVM-22 compatibility artifact. F0's
  supported Ubuntu 22.04/LLVM-21 lane is separate; the full R0 suite did not run in
  that CI lane.
- R0 consumed and hash-verified F0's exact pinned build. It did not run CTest or
  claim a fresh R0 build/test result for the supplied build directory.

Full-suite command, run from the transformer repository root:

```bash
/usr/bin/bash probes/r0/run.sh \
  --eshkol-source /home/gabe/.codex/worktrees/49f7/eshkol-transformer/.deps/eshkol-src \
  --existing-build /home/gabe/.codex/worktrees/49f7/eshkol-transformer/.deps/eshkol-build-minimal \
  --work-dir /home/gabe/.cache/eshkol-r0-canonical-full-work-20260828 \
  --results-dir /home/gabe/.cache/eshkol-r0-canonical-full-results-20260828
```

Outer exit `1`; harness completed; `failures=38`; 183 records. Every runnable
positive probe ran twice in AOT and twice in JIT. RMSNorm was rejected during AOT
compilation, so that candidate has three rather than five records.

Supplemental command shape (eight fresh result/work pairs):

```bash
/usr/bin/bash probes/r0/run.sh \
  --eshkol-source /home/gabe/.codex/worktrees/49f7/eshkol-transformer/.deps/eshkol-src \
  --existing-build /home/gabe/.codex/worktrees/49f7/eshkol-transformer/.deps/eshkol-build-minimal \
  --work-dir /home/gabe/.cache/eshkol-r0-canonical-supplemental-20260828/PROBE-work \
  --results-dir /home/gabe/.cache/eshkol-r0-canonical-supplemental-20260828/PROBE-results \
  --probe PROBE
```

The exact backend-inventory command/output and every phase's command, exit code,
stdout, and stderr are retained. Harness output comparisons cover stdout. JIT
compiler stderr can differ between the first and cached repeat, so this report does
not claim diagnostic-stream determinism.

## Capability matrix

`Observed-supported` is limited to the exact tested contract.
`Observed-broken` means reachable behavior returned a wrong/unsafe result, crashed,
timed out, or violated required parity. `Observed-unsupported` means both public
paths rejected the named candidate. All other rows are `untested-with-reason`.

| Capability | Classification | Direct observation |
|---|---|---|
| Numeric construction and rank/shape | observed-supported | Correct 2-D/3-D shapes and values; rank-9 indexed successfully |
| In-bounds indexing | observed-supported | First/last/rank-3 reads passed |
| Out-of-bounds indexing | observed-supported | AOT/JIT exit 1: `tensor-get: index out of bounds` |
| Reshape values and ownership | observed-supported | Values passed; mutating a reshape changed original tensor storage |
| Reshape size validation | observed-broken | 3 values reshaped to 2x2 and returned `#((1 2) (3 1e-323))` |
| Negative dimensions | observed-broken | AOT accepted; JIT timed out after 90 seconds |
| Transpose values and ownership | observed-supported | Values passed; mutation showed transpose copied storage |
| Physical contiguity, strides, partial views | untested-with-reason | No executed stride/contiguity introspection contract |
| Compatible broadcasting | observed-supported | Exact 2x1 plus length-3 addition produced 2x3 reference |
| Incompatible broadcasting | observed-broken | AOT/JIT both SIGSEGV (139) |
| Integer tensor-value semantics | observed-supported | Tested integer elements remained `integer?` |
| Boolean tensor-value semantics | observed-unsupported | Tested `#t/#f` values were numerically coerced |
| Logical f64/f32/f16/bf16/i8 | observed-supported | Dtype tags, cast quantization, f16+f32 promotion, and f16 matmul propagation passed |
| Physical dtype byte width/compact storage | untested-with-reason | Logical tags and quantization do not measure storage size |
| Native CPU execution | observed-supported | Native AOT binaries and JIT ran with compiler `gpu=off`, BLAS off |
| Explicit CPU device identity/selection | untested-with-reason | No executed tensor-device selector/query contract |
| `gpu-*` numerical aliases | observed-supported on CPU | Five aliases returned correct values while compiler reported GPU off |
| GPU selection, transfer, or execution | untested-with-reason | No device/backend enabled or visible; no dispatch banner/telemetry |
| Rank-1 dot | observed-supported | `tensor-dot` returned 32 |
| 2-D matmul | observed-supported | 2x3 by 3x2 exact reference passed |
| Batched matmul | observed-supported | Two 2x2 identity batches returned exact inputs |
| General/broadcasted batched matmul | untested-with-reason | Only one rank-3 same-batch shape was exercised |
| Matmul dimension error | observed-supported | AOT/JIT exit 1 with inner-dimension mismatch |
| Embedding gather and bounds | observed-supported | `[2,0]` rows correct; OOB index exited 1 with actionable diagnostic |
| Embedding repeated-index scatter gradient | observed-supported | Repeated index accumulated `[0,0,2,2,0,0]` in both engines |
| LayerNorm forward | observed-supported | Four-element reference values passed |
| LayerNorm derivative | observed-unsupported | Dual-bearing vector rejected with a type error in all runs |
| RMSNorm language primitive | observed-unsupported | `rms-norm` unknown/undefined in AOT and JIT |
| ReLU, sigmoid, GELU, SiLU | observed-supported | Forward reference values passed |
| Axis and large-logit softmax | observed-supported | Row sums and shifted logits matched references |
| ReLU and SiLU gradients | observed-supported | Reference tensor gradients passed |
| GELU/softmax gradients | untested-with-reason | No checked-in numerical gradient probe |
| Causal mask | observed-supported | Allowed/blocked 3x3 entries matched |
| Scaled-dot attention forward | observed-supported | 2x2 zero-score reference returned expected averages |
| Attention derivative | observed-broken | All four finite-difference comparisons printed `FAIL` |
| Masked/MHA attention and RoPE | untested-with-reason | Not exercised; no support inference from advertised names |
| Indexed-target cross entropy | observed-broken | Indexed `[0,1]` returned `2.820075...`, not `0.126928...`, exit 0 |
| Scalar forward derivative | observed-supported | Central finite difference at x=3 passed |
| Scalar `gradient` | observed-supported | Same finite-difference reference passed |
| Generic tensor `gradient` | observed-broken | Reference gradient `[4,3]` failed in all four runs |
| Matmul tensor gradient | observed-supported | Identity-matmul sum returned four ones |
| AD through scalar control flow | observed-supported | Positive/negative branches returned 4 and -1 derivatives |
| Repeated independent gradient calls | observed-supported | Two calls produced exact total 16 |
| Universal reverse-mode/captures/loops | untested-with-reason | Mixed tensor results disprove a global claim; broad semantics not established |
| Deterministic stdout within a backend | observed-supported | All runnable positives repeated exactly in AOT and JIT |
| AOT/JIT stdout parity | observed-broken for RNG | Non-RNG supported probes matched; seeded RNG sequences differed by backend |
| Fixed-seed fresh-process RNG | observed-supported within backend | Two AOT and two JIT processes repeated their respective sequences |
| Text and binary byte I/O | observed-supported | Text plus bytes 0/127/255 round-tripped |
| Rename prerequisite | observed-supported | Source disappeared and destination appeared |
| SHA-256 file checksum | observed-supported | `abc` matched the standard digest |
| Tensor/model serialization round trip | observed-supported | Shapes/values and named model tensor round-tripped |
| Corrupt tensor/model rejection | observed-broken | Both three-byte payloads returned truthy values in all four runs |
| Missing-file error | observed-broken | AOT/JIT both SIGSEGV at null address |
| Version/schema/checksum-corruption guarantees | untested-with-reason | Round trip and SHA-256 do not prove format safety |
| fsync/durable atomic replacement | untested-with-reason | Rename alone does not establish durability/atomic protocol |
| Bounded long-loop smoke | observed-supported | 100,000 tensor construction/read iterations returned exact total under 2 GiB cap |
| Flat RSS, ASan, and long-lived ownership | untested-with-reason | `/usr/bin/time` absent; one bounded completion is not a growth proof |
| Other advertised constructors/linear algebra | untested-with-reason | `zeros/ones/eye/arange/linspace/solve/det/inv` were outside the transformer-critical executed set |

## Numerical and determinism evidence

Scalar `derivative` and scalar `gradient` use central finite differences with
epsilon `1e-5` for `f(x)=x*x+x*x` at x=3. Attention derivative uses the same
method with tolerance `1e-4`. Tensor matmul, ReLU, SiLU, generic tensor, and
repeated-index embedding use exact analytic references. The first five pass except
generic tensor gradient and attention derivative; therefore K1/P1 cannot treat the
advertised reverse-AD surface as generally reliable.

All runnable positive AOT outputs repeated exactly, as did all JIT outputs. Supported
non-RNG AOT/JIT stdout matched. Fixed seed repeated within AOT and within JIT but the
two backend sequences differed; compiler warning stderr also differed on cached JIT
repeats and is outside the parity claim.

## Decision matrix

| Capability/gap | Decision | Downstream boundary |
|---|---|---|
| Tested tensor shape/index, logical dtype casts, dot/2-D/batched matmul, embedding, LayerNorm, activations, mask/forward attention | Use existing reachable Eshkol only for the exact observed cases | K1 discovery must remain shape/backend specific |
| Bounds checks that explicitly reject | Use existing Eshkol | Keep negative tests; do not generalize to other malformed cases |
| Reshape validation, negative dimensions, broadcast crash, missing-file crash | Eshkol-core issues [#550](https://github.com/tsotchke/eshkol/issues/550), [#549](https://github.com/tsotchke/eshkol/issues/549) | Reject affected project operations before core dispatch |
| Corrupt serialization acceptance | Eshkol-core issue [#549](https://github.com/tsotchke/eshkol/issues/549); isolated project format | C1 needs versioned, checksummed, non-executable, atomic persistence |
| Generic/attention/LayerNorm AD gaps | Eshkol-core issue [#551](https://github.com/tsotchke/eshkol/issues/551) | P1/N2/A2 training remains blocked on per-op gradient gates |
| Indexed cross entropy | Eshkol-core issue [#552](https://github.com/tsotchke/eshkol/issues/552); versioned K1 kernel if unresolved | L2 must not use the current result |
| Backend-dependent seeded RNG | Eshkol-core issue [#553](https://github.com/tsotchke/eshkol/issues/553) | Q0/checkpoints record backend or provide project RNG |
| RMSNorm, truthful device identity, explicit GPU/AMD kernels | Isolated versioned native-extension candidates | Never infer GPU from a `gpu-*` name; unavailable paths fail explicitly |
| Physical compact f32/f16/bf16 storage | Measure before choosing core or extension | Logical tags alone do not satisfy MP4 memory/precision contracts |
| Scalar-loop or CPU alias hidden in hot paths | Do not use as production proof | B0/K1 require device/kernel telemetry and benchmarks |

The upstream reports name the audited pin and explicitly state that newer upstream
HEAD was not rebuilt.

## Independent audit

An independent agent reran seven representative positives and three malformed cases
with fresh scratch paths and the same canonical compiler. It corroborated tensor
core, precision discriminator, matmul, broadcast, scalar AD, attention-AD failure,
within-backend RNG repeat/cross-backend divergence, explicit index/matmul errors, and
unsafe reshape success. Available outputs matched the main run byte-for-byte. Its
false-positive audit is retained in
`docs/evidence/r0/2026-08-28/independent-review.md` and drives the narrow claim
wording above.

## Known limits and follow-ups

- This host cannot prove GPU, reduced-precision physical storage, device transfer,
  cross-device determinism, or the supported Ubuntu/LLVM-21 lane.
- Rank 9 works; no maximum-rank boundary was established.
- No stride/contiguity API, sanitizer lane, flat-RSS series, masked/MHA/RoPE probe,
  or durable atomic-write test ran.
- The F0 build is exact and provenance-verified but minimal; results do not extend
  to disabled BLAS/XLA/GPU/TensorCore/agent-FFI configurations.
- K1 is unblocked for conservative capability discovery and explicit rejection.
  P1/N2/A2 remain blocked on per-operation gradient reliability; C1 remains blocked
  on safe persistence semantics.
- No transformer subsystem or fallback was implemented by R0.

Historical runner outage retained verbatim for the handoff (resolved before this
evidence run):

```text
exec_command failed for /usr/bin/bash -c pwd: CreateProcess { message: "Rejected(\"Failed to create unified exec process: No such file or directory (os error 2)\")" }
```
