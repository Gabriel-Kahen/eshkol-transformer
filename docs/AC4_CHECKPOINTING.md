# AC4 activation checkpointing

AC4 runs a stack of 1–16 pre-norm residual transformer blocks in serial CPU
binary32. The public C11 and C++17 interface is
`include/eshkol_transformer/ac4_checkpointing_abi.h`. Build
`build/ac4/libeshkol_ac4.a` with `bash scripts/build-ac4.sh` and link it with the
MOD5 archive from `bash scripts/build-mod5.sh`, the K1 ABI implementation, and
`-lm`. AC4 calls the statically linked
`et_mod5_kernel_provider_v1()` from `mod5_primitives_abi.h` and validates each
operation with that provider before invoking it.

## Model and API

The supported shapes are the following MOD5 compositions. RMSNorm and SwiGLU
dispatch over the full batch. Attention dispatches each batch independently
through the matching single-batch self-attention row.

| Attention | Batch | Tokens | Query heads | KV heads | Head width |
|---|---:|---:|---:|---:|---:|
| MQA | 1 | 2 | 2 | 1 | 2 |
| MQA | 2 | 3 | 4 | 1 | 2 |
| GQA | 2 | 3 | 4 | 2 | 2 |
| MHA | 2 | 3 | 2 | 2 | 2 |

For hidden width `D = Hq × Dh` and KV width `V = Hkv × Dh`, the residual
stream is dense `[N,T,D]`. Each layer packs the two RMSNorm scales and seven
bias-free projection matrices in this order.

| Tensor | Shape |
|---|---|
| `gamma1` | `[D]` |
| `Wq` | `[D,D]` |
| `Wk`, `Wv` | `[V,D]` each |
| `Wo` | `[D,D]` |
| `gamma2` | `[D]` |
| `Wgate`, `Wup`, `Wdown` | `[D,D]` each |

Rows of each projection matrix correspond to output channels. The attention
branch projects the first normalized stream to Q, K, and V, runs full causal
attention over positions `0..T-1`, and projects its output to residual width.
The FFN branch applies SwiGLU to the gate and up projections of the second
normalized stream and projects the result back to residual width. Each branch
applies inverted dropout with branch-specific counters before its residual addition.
Both normalizers use the configured positive epsilon.

`et_ac4_plan_v1` returns the tensor counts and minimum workspace for a valid
configuration. `et_ac4_run_v1` returns the final stream and
`loss = sum((output - target)^2) / (2 × N × T × D)`, accumulated in binary64.
In `ET_AC4_FORWARD_BACKWARD` mode it also returns the analytic binary32
gradients of the loss with respect to the input and every packed parameter.
`ET_AC4_FORWARD_ONLY` requires empty gradient spans and uses one rolling
stream regardless of the configured policy.

Supply workspace from an aligned float array or dynamic allocation. Tensor
counts must match the plan exactly. All descriptors and payload spans must be
distinct and nonoverlapping, including the entire declared workspace.
`activation_limit_bytes = 0` disables the activation cap. A smaller nonzero
cap returns `ET_AC4_ACTIVATION_LIMIT`. A buffer shorter than the full workspace
requirement returns `ET_AC4_WORKSPACE_TOO_SMALL`. Both checks run before
scratch writes. Invalid configuration or pointers, invalid tensor shapes,
unsupported shapes or FP controls, and aliasing have distinct statuses.
Nonfinite inputs or evaluation return `ET_AC4_NUMERIC_ERROR`.
`ET_AC4_PROVIDER_ERROR` covers other provider failures.

The caller owns the workspace and may reuse it immediately after a call.
On any failure, result, predictions, and gradients retain their previous
bytes. A run may overwrite workspace while working, including before a
numeric failure. The pinned provider requires x86-64 round-to-nearest with
masked exceptions and FTZ/DAZ disabled. Calls may set sticky FP exception
flags or change `errno`. Each concurrent call needs distinct mutable
storage and stable inputs and floating-point controls.

## Replay and arena accounting

Each layer derives a key from `seed`, `step`, and zero-based layer index
using the SplitMix64 finalizer `mix`. The following expressions use unsigned
64-bit arithmetic.

```text
G       = 0x9e3779b97f4a7c15
key     = mix(seed ^ mix(step + G) ^ mix(layer + 0xd1b54a32d192ed03))
counter = 2 * flat_element_index + branch + 1   (attention 0, FFN 1)
uniform = (mix(key + counter * G) >> 11) * 2^-53
```

A value is dropped when `uniform < dropout_rate`. Kept values are scaled by
`1 / (1 - dropout_rate)` in binary32. The forward, replay, and backward paths
use this same rule. The backward pass starts with the same final residual
stream and visits layers in reverse order for every policy. It accumulates
parameter gradients and residual contributions in the same order.

Let `X = 4 × N × T × D` and `Z = 4 × N × T × V` be the bytes in a residual
stream and one KV projection. A layer tape takes `A = 8X + 2Z` bytes. It holds
both normalized streams, projected Q/K/V, attention output, the first
residual sum, and the SwiGLU gate, up, and output tensors. For a segment
stride `k`, let `C = floor((L-1)/k)` and let `m_i` be the lengths of the
successive segments, including any shorter tail.

| Training policy | Schedule | Peak activation bytes | Extra layer forwards |
|---|---|---|---|
| `NONE` | Retain every tape and interlayer stream | `L×A + (L+6)×X` | `0` |
| `EVERY_LAYER` | Retain only interlayer streams and replay each tape once | `A + (L+6)×X` | `L` |
| `SQRT_SEGMENTS` | Use `k = ceil(sqrt(L))`, keep `C` boundaries, and cache the streams of one segment | `A + (7+C+k-1)×X` | `2L-(C+1)` |
| `EVERY_K` | Keep `C` boundaries and replay each layer's prefix from its nearest boundary | `A + (7+C+[k>1])×X` | `sum(m_i×(m_i+1)/2)` |

The common training term `7X` holds the final stream, the gradient carry,
and five temporary streams. Forward-only peaks at `A + (2+[L>1])×X`.
The run reports the arena's high-water mark from actual reservations.
The accounting covers activation storage controlled by AC4. Caller payloads,
scalar and descriptor stack storage, and the pinned provider's fixed-size
local evaluator scratch are outside the arena.

With a fixed tensor shape, activation retention grows as `O(L)` for
`NONE` and `EVERY_LAYER`, `O(sqrt(L))` for `SQRT_SEGMENTS`, and
`O(ceil(L/k))` for `EVERY_K`. Setting `k = L` gives constant activation
retention and `L(L+1)/2` replay forwards. The other checkpointing schedules
with linear replay retain more intermediate state. One full parameter
gradient buffer is staged in workspace to make publication failure-atomic.
It costs `4 × L × (2D + 5D² + 2VD)` bytes, so total training workspace still
grows linearly with depth. The plan and result report those bytes separately.

## Measured results

The benchmark used Ubuntu 22.04.5 LTS and Clang 21.1.8 in
`eshkol-r0-debug:20260928` on a single logical CPU of an AMD Ryzen 7 3700X.
It timed complete forward/backward calls with dropout 0.2 and GQA
`(N,T,Hq,Hkv,Dh) = (2,3,4,2,2)`. Each reported time is the median of seven
interleaved samples of 120 calls using `CLOCK_MONOTONIC` after warm-up.
Times and ratios describe this run on small verified shapes.

| Layers | Policy | Planned B | Peak B | Activation saved | Workspace B | Replays | µs/call | Time vs. NONE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | NONE | 8,832 | 8,832 | 0.00% | 15,232 | 0 | 66.479 | 1.000× |
| 4 | EVERY_LAYER | 3,648 | 3,648 | 58.70% | 10,048 | 4 | 94.694 | 1.424× |
| 4 | SQRT_SEGMENTS | 3,456 | 3,456 | 60.87% | 9,856 | 6 | 106.509 | 1.602× |
| 4 | EVERY_K, k=4 | 3,264 | 3,264 | 63.04% | 9,664 | 10 | 133.000 | 2.001× |
| 8 | NONE | 16,512 | 16,512 | 0.00% | 29,312 | 0 | 132.061 | 1.000× |
| 8 | EVERY_LAYER | 4,416 | 4,416 | 73.26% | 17,216 | 8 | 186.029 | 1.409× |
| 8 | SQRT_SEGMENTS | 3,840 | 3,840 | 76.74% | 16,640 | 13 | 218.101 | 1.652× |
| 8 | EVERY_K, k=8 | 3,264 | 3,264 | 80.23% | 16,064 | 36 | 367.026 | 2.779× |
| 16 | NONE | 31,872 | 31,872 | 0.00% | 57,472 | 0 | 263.757 | 1.000× |
| 16 | EVERY_LAYER | 5,952 | 5,952 | 81.33% | 31,552 | 16 | 374.150 | 1.419× |
| 16 | SQRT_SEGMENTS | 4,224 | 4,224 | 86.75% | 29,824 | 28 | 448.460 | 1.700× |
| 16 | EVERY_K, k=16 | 3,264 | 3,264 | 89.76% | 28,864 | 136 | 1,153.750 | 4.374× |

The optimized and ASan/UBSan/LSan runs each checked 5,440 central differences
over every parameter and input element of multi-layer fixtures spanning all
four shape rows, with dropout disabled and enabled. Maximum absolute error
was `8.10525438e-06` against the criterion
`7e-6 + 0.003 × abs(analytic_gradient)`. Both runs also passed 136
`EVERY_K` schedules, 16 square-root schedules, 64 forward-only cases,
independent RNG mask goldens, causal and batch isolation checks, and 51
rejected runs plus four planner errors.

At each of 4, 8, and 16 layers, all four policies had byte-identical
predictions, loss, input gradients, parameter gradients, and updated
parameters across eight SGD updates at dropout rates 0 and 0.2. The training
loss with the current step's mask decreased on every update, and the fixed
dropout-free evaluation loss decreased across the full sequence. At 16 layers
with training dropout, evaluation loss decreased from `0.187536549` to
`0.0771545581`.

```bash
CC=clang-21 CXX=clang++-21 AC4_ASAN_DETECT_LEAKS=1 bash scripts/test-ac4.sh
python3 -m unittest tests/q0/test_python_isolation.py
bash scripts/check-ci-topology.sh
```

The supported ABI is limited to the listed CPU binary32 shape rows and the
half mean squared error objective. Arbitrary tensor extents, attention
windows, masks beyond causality, positional biases, GPU execution, mixed
precision, optimizer state, and carrier integration are outside this release.
The callers in the training smoke apply SGD using the gradients returned by
AC4.
