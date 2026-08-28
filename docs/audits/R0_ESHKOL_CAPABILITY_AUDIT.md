# R0 Eshkol capability audit

## Result and evidence boundary

R0 is ready for review. The audit executed pinned Eshkol commit
`746e5185071c90e0e386f3acb9fba2d90329016f` (tree
`a10c6e8b44cd9bc5dfcee19c8f9f82f50b56938b`) from a clean checkout. The
transformer audit base was `bb5a2e358d68a13505d681bc4d09511be6e1b3d6`.

The retained final run is in `docs/evidence/r0/2026-08-28/`. It contains 184
recorded commands plus the manifest, exact command lines, exit codes, stdout,
stderr, environment, assertion failures, and parity failures. Its nonzero suite
result (`failures=71`) is expected: the harness treats every missing completion
marker, crash, unsafe malformed-input success, unsupported candidate, zero-test
CTest result, and parity failure as audit evidence.

The checked-in generated text has trailing whitespace stripped for repository
hygiene; command tokens, substantive stdout/stderr, and exit codes are unchanged
from `/home/gabe/.cache/eshkol-r0/final-results6-20260828`.

Earlier evidence is not used for classification. One fresh build was invalidated
when an early harness revision applied a file-size limit to compiler object files.
That defect was fixed by bounding captured streams instead. A second early run
exposed stale scratch-file collisions; file probes now self-clean task-owned paths.

## Exact environment and commands

- Host: Linux `7.1.2-3-cachyos`, x86-64, AMD Ryzen 7 3700X, 16 logical CPUs.
- Toolchain: GCC 16.1.1, CMake 4.3.3, Ninja 1.13.2, LLVM 21.1.8.
- Configure: XLA disabled; CUDA not found; GPU disabled; BLAS not found. No NVIDIA
  or ROCm device inventory was present.
- Default LLVM 22.1.6 was rejected: `Expected LLVM 21, got 22.1.6 from
  /usr/bin/llvm-config`. LLVM 21.1.8 was staged without a system install.
- The initial empty `work3` configure/build completed with exit 0. CTest exited 0
  but printed `No tests were found!!!`; upstream tests are absent, not passed.

The final full probe command was:

```bash
/usr/bin/bash -c 'TMPDIR=/home/gabe/.cache/eshkol-r0/tmp ESHKOL_LLVM_CONFIG=/home/gabe/.cache/eshkol-r0/llvm21/root/usr/bin/llvm-config-21 /usr/bin/bash probes/r0/run.sh --eshkol-source /home/gabe/.cache/eshkol-r0/source --work-dir /home/gabe/.cache/eshkol-r0/work3 --results-dir /home/gabe/.cache/eshkol-r0/final-results6-20260828'
```

The final run reconfigured and incrementally rebuilt the clean full build, then ran
every probe twice under AOT and twice under JIT. It recorded 185 manifest lines
including the header and ended with `failures=71`. Repeated AOT output was stable
for every runnable capability. The only parity failures were RNG AOT/JIT divergence
and JIT RNG repeat divergence.

## Capability matrix

`Observed-supported` is limited to the tested contract. `Observed-broken` means a
reachable operation returned a wrong/unsafe result, crashed, silently stopped, or
violated parity. `Observed-unsupported` means both public paths rejected the named
operation. Everything else remains `untested-with-reason`.

| Capability | Classification | Direct observation |
|---|---|---|
| Numeric construction, rank and shape | observed-supported | Correct rank-1/2/3 and rank-9 values/shapes in AOT/JIT |
| In-bounds indexing and mutation | observed-supported | First/last/rank-3 reads and `tensor-set!` mutation passed |
| Out-of-bounds indexing | observed-broken | AOT/JIT exited 0 and returned `9.88131e-324` |
| Reshape values and ownership | observed-supported | Values passed; mutating a reshape changed original storage |
| Reshape size validation | observed-broken | 3 values reshaped to 2x2, padded with junk, exit 0 |
| Negative dimensions | observed-broken | Both paths reached `MALFORMED_INPUT_ACCEPTED` |
| Transpose values and ownership | observed-supported | Values passed; transpose copied storage in both engines |
| Physical contiguity, strides, partial-index views | untested-with-reason | No public layout/stride introspection contract |
| Compatible broadcasting | observed-supported | 2x1 plus length-3 produced the correct 2x3 result |
| Incompatible broadcasting | observed-broken | AOT returned `#()`; JIT segfaulted (139) |
| Integer semantic tensor values | observed-supported | Vector elements remained `integer?` in both engines |
| Integer physical storage dtype | untested-with-reason | No public storage-dtype introspection |
| Boolean tensor values/storage | observed-broken | `(tensor 2 #t #f)` exposed numeric values |
| f64 discriminator | observed-supported | `1.0000000000000002 - 1.0 > 0` passed |
| f32 storage/arithmetic | untested-with-reason | No reachable explicit f32 construction/cast contract |
| f16/bf16 storage/arithmetic | untested-with-reason | No reachable public contract and no capable device |
| Native CPU execution | observed-supported | AOT binaries and JIT probes ran on the CPU-only build |
| Explicit CPU device selection/identity | untested-with-reason | No public selector or tensor-device query |
| GPU selection/execution | untested-with-reason | GPU disabled/no device on measured host; no global claim |
| Device transfer/tensor identity | untested-with-reason | No reachable public API |
| Rank-1 dot | observed-supported | `tensor-dot` returned scalar 32 |
| 2-D matmul | observed-supported | 2x3 by 3x2 matched exact references |
| Matmul dimension error | observed-supported | Exit 1 with actionable inner-dimension mismatch |
| Batched matmul | observed-unsupported | AOT unknown `batch-matmul`; JIT undefined function |
| Embedding gather | observed-supported | Indices `[2,0]` returned the correct rows |
| Embedding bounds checking | observed-broken | OOB index returned junk with exit 0 |
| Embedding scatter-add gradient | observed-broken | All runs exited 0 without output/completion |
| LayerNorm forward | observed-supported | Reference values passed in AOT/JIT |
| LayerNorm gradient | observed-broken | All runs segfaulted (139) |
| RMSNorm | observed-unsupported | AOT unknown function; JIT undefined function |
| ReLU, sigmoid, GELU, axis softmax | observed-supported | Reference outputs/row sums passed |
| SiLU/large-logit softmax stability | untested-with-reason | Exact contracts not probed |
| Causal mask | observed-supported | 3x3 allowed/blocked entries matched |
| Scaled-dot attention forward | observed-supported | 2x2 reference returned expected averages |
| Attention gradient | observed-broken | All runs segfaulted (139) |
| Masked attention argument and RoPE | untested-with-reason | Not exercised by forward probe |
| Indexed-target cross entropy | observed-broken | Invalid indexed targets returned `2.82008` |
| Scalar forward derivative | observed-supported | Repeated-expression derivative matched finite difference |
| Scalar `gradient` | observed-supported | Matched finite difference; does not prove reverse mode |
| Repeated scalar gradient calls | observed-supported | Two independent calls summed to exact reference |
| Tensor gradients (generic/ReLU/matmul/embedding) | observed-broken | Exit 0 silently without completion markers |
| Autodiff through control flow | observed-broken | Both engines aborted (134) on dual comparison |
| True reverse mode/captures/loops/cross-step accumulation | untested-with-reason | Scalar success does not establish semantics |
| AOT/JIT parity for supported deterministic probes | observed-supported | Exact stdout parity for supported non-RNG cases |
| Fixed-seed RNG | observed-broken | AOT repeated; JIT differed per process/from AOT and duplicated its first scalar internally |
| Text file round trip | observed-supported | Self-cleaning write/read passed twice per engine |
| Binary byte I/O | observed-supported | Bytes 0, 127, 255 round-tripped twice per engine |
| File rename prerequisite | observed-supported | Source disappeared and destination appeared |
| Missing-file error | observed-broken | AOT/JIT segfaulted (139), no actionable error |
| Safe versioned tensor serialization | untested-with-reason | No named AOT/JIT format probe; no public contract discovered |
| Checksums/fsync/durable atomic-write protocol | untested-with-reason | Rename does not establish these guarantees |
| Bounded long-loop smoke | observed-supported | 100,000 tensor construction/read loop iterations returned exact total |
| Flat RSS/view lifetime/ASan ownership | untested-with-reason | Smoke completion is not a memory-growth proof |
| Malformed input reporting overall | observed-broken | Only matmul mismatch failed explicitly; seven others succeeded/crashed |

## Autodiff and numerical evidence

Scalar probes use central differences with epsilon `1e-5`. For
`f(x) = x*x + x*x` at `x=3`, both `derivative` and scalar `gradient` matched the
reference within `1e-5` in all AOT/JIT runs. This establishes scalar
differentiation only, not reverse mode.

Generic, ReLU, matmul, and repeated-index embedding tensor-gradient probes compiled
then exited 0 without any assertion/completion marker. LayerNorm and attention
derivatives crashed in every run; control-flow AD aborted in every run. K1/P1 must
treat tensor reverse gradients and gradient accumulation as unavailable.

## Source review used only for routing

Independent pinned-source inspection explains but does not upgrade runtime results:
tensor storage candidates use `double*` without public dtype/device identity;
high-level VM `gradient` routes through forward dual propagation; GPU paths can
silently return to CPU; cross entropy expects one-hot targets; and attention contains
recursive scalar loops. Executable observations remain authoritative.

## Decision matrix

| Gap | Decision | Downstream effect |
|---|---|---|
| Proven f64 CPU tensor ops, dot/2-D matmul, embedding gather, LayerNorm, activations, toy attention | Use existing Eshkol only for exact tested contract | A0 models conservative f64 CPU capabilities; K1 rejects other variants |
| Bounds/shape/file errors, broadcast crash, RNG JIT, tensor AD | File upstream core defects; do not wrap/hide behavior | Blocks safe K1/P1 use until fixed or isolated |
| Batched matmul and RMSNorm | Isolated versioned native extension after A0/Q0 ABI | K1 exposes discovery and explicit unsupported errors |
| Embedding scatter-add, tensor reverse gradients, indexed loss | Prefer upstream repair; otherwise reviewed K1 extension | Blocks P1 training/autodiff contract |
| f32/f16/bf16, device identity/transfer, GPU proof | Leave unsupported in project discovery until measured | A0/K1 never silently coerce or fall back |
| Serialization/checksum/fsync/atomic replacement | Versioned non-executable project format/native support | Informs P1/Q0 artifacts |
| Scalar-loop attention | Do not use as production hot path | K1 needs reviewed tensor/native kernels |

## Independent audit

An independent agent rebuilt representative AOT probes and reran AOT/JIT tensor,
matmul, broadcast, activation, embedding, normalization, attention, scalar finite
difference, malformed-input, and RNG cases under explicit `/usr/bin/bash`. It
corroborated supported results and unsafe failures. It also caught four
false-positive risks that were corrected: zero-test CTest, compiler exit 0 with hard
diagnostics, missing completion markers, and unbounded malformed-allocation output.
Scratch evidence remains at `/tmp/eshkol-r0-review.h54c5Q`.

## Known limits and follow-ups

- This host cannot validate GPU, reduced precision, device transfer, or cross-device
  determinism.
- Rank-9 works; no maximum-rank boundary was established.
- No stride/contiguity API was found; flat-memory behavior was not established.
- No upstream unit tests were registered with `ESHKOL_BUILD_TESTS=ON`.
- Confirmed defects have checked-in reproducers, but the upstream
  `Gabriel-Kahen/eshkol` repository has GitHub Issues disabled; issue filing is
  deferred until an upstream tracker is available.
- K1 is unblocked for a conservative f64 CPU capability layer and explicit rejection
  of unproven features. P1 remains blocked on reliable tensor gradients.
