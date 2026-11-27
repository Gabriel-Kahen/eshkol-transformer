# AC4 activation checkpointing

AC4 runs a stack of 1–16 pre-norm residual transformer blocks in serial CPU
binary32. The public C11 and C++17 interface is
`include/eshkol_transformer/ac4_checkpointing_abi.h`. Build
`build/ac4/libeshkol_ac4.a` with `bash scripts/build-ac4.sh` and link it with the
MOD5 archive from `bash scripts/build-mod5.sh`, the K1 ABI implementation, and
`-lm`. AC4 calls the statically linked
`et_mod5_kernel_provider_v1()` from `mod5_primitives_abi.h`. The checked
path validates each operation with that provider before invoking it.

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
| `EVERY_K_REPLAY_GRADIENTS` | Use the `EVERY_K` schedule for two backward passes | `A + (7+C+[k>1])×X` | `2×sum(m_i×(m_i+1)/2)` |

The common training term `7X` holds the final stream, the gradient carry,
and five temporary streams. Forward-only peaks at `A + (2+[L>1])×X`.
The run reports the arena's high-water mark from actual reservations.
The accounting covers activation storage controlled by AC4. Caller payloads,
scalar and descriptor stack storage, and the pinned provider's fixed-size
local evaluator scratch are outside the arena.

With a fixed tensor shape, activation retention grows as `O(L)` for
`NONE` and `EVERY_LAYER`, `O(sqrt(L))` for `SQRT_SEGMENTS`, and
`O(ceil(L/k))` for the `EVERY_K` policies. Setting `k = L` gives constant
activation retention. The first four policies stage one full parameter
gradient buffer in workspace to make publication failure-atomic. It costs
`4 × L × (2D + 5D² + 2VD)` bytes, so their total training workspace still
grows linearly with depth.

`ET_AC4_POLICY_EVERY_K_REPLAY_GRADIENTS` is an opt-in ABI 1.1 policy with the
same struct layouts and `checkpoint_every` range as `EVERY_K`. It stages
one layer of parameter gradients, `4 × (2D + 5D² + 2VD)` bytes. At `k = L`,
its activation arena and gradient staging are both constant in depth for
`L > 1`. Caller-owned parameters and the output gradient span still scale
with depth. `activation_peak_bytes` excludes the separately reported
`gradient_staging_bytes`. `rematerialized_layers` includes replay forwards
from both backward passes.

The first backward pass validates the full reverse traversal while
overwriting the same one-layer staging buffer. Only after it succeeds does
the second pass repeat that traversal and copy each staged gradient into
its packed `d_parameters` slot. Both passes start with the same final
stream, target, immutable boundaries and parameters, and recompute dropout
from the same counters. They share the same reserved scratch and zero the
gradient buffer before each layer. Replays initialize the tapes and scratch
streams before their values are read. The publication pass uses the
non-failing MOD5 invoke path for the already accepted operations. All
allocation and numeric validation precede publication. This relies on the
pinned, deterministic MOD5 provider and on the caller keeping inputs and FP
controls stable for the full call. `d_input`, predictions, and the result
are published after the second pass. Forward-only calls use the same
rolling stream as the other policies.

## Native replay and AdamW witness

The opt-in native gate uses the admitted three-layer GQA fixture with two
batches, three tokens, four query heads, two KV heads, and head width two.
It compares ordinary retained activations with bounded gradient replay at
stride two, using separate workspaces and the same fixed seed and dropout
rate of 0.2 at each of four steps. The packed AC4 roster maps to 27 distinct
native F32 parameter tensors. Each successful AC4 call supplies one
already-normalized gradient per tensor to the atomic F32 gradient plan.
The accepted constant-schedule O2 AdamW step updates the parameters and
both moments. The next AC4 step consumes the resulting native weight bits.

Before the replay optimizer is enrolled, an activation cap one byte below
the plan must reject. The gate checks unchanged AC4 outputs, workspace,
native parameters and gradient state, and O2 enrollment counts. Restoring
the cap retries the same input, seed, step, and workspace. Every successful
step then compares predictions, loss, input and parameter gradients,
updated weights, and both moment tensors byte for byte. Both moments have
nonzero finite data for every destination.

```bash
bash scripts/test-ac4-o2-replay.sh
```

Normal, repeat, and native ASan/UBSan/LSan runs match the full binary trace.
The witness covers this CPU F32 fixture and in-process optimizer trajectory.

## Measured results

The composed AC4/MOD5-B benchmark used Ubuntu 22.04.5 LTS and Clang 21.1.8 in
`eshkol-r0-debug:20260928` on a single logical CPU of an AMD Ryzen 7 3700X.
It timed complete forward/backward calls with dropout 0.2 and GQA
`(N,T,Hq,Hkv,Dh) = (2,3,4,2,2)`. Each reported time is the median of seven
interleaved samples of 120 calls using `CLOCK_MONOTONIC` after warm-up.
Times and ratios describe this run on small verified shapes.

| Layers | Policy | Planned B | Peak B | Activation saved | Workspace B | Replays | µs/call | Time vs. NONE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | NONE | 8,832 | 8,832 | 0.00% | 15,232 | 0 | 67.923 | 1.000× |
| 4 | EVERY_LAYER | 3,648 | 3,648 | 58.70% | 10,048 | 4 | 91.396 | 1.346× |
| 4 | SQRT_SEGMENTS | 3,456 | 3,456 | 60.87% | 9,856 | 6 | 103.137 | 1.518× |
| 4 | EVERY_K, k=4 | 3,264 | 3,264 | 63.04% | 9,664 | 10 | 127.038 | 1.870× |
| 8 | NONE | 16,512 | 16,512 | 0.00% | 29,312 | 0 | 125.994 | 1.000× |
| 8 | EVERY_LAYER | 4,416 | 4,416 | 73.26% | 17,216 | 8 | 177.876 | 1.412× |
| 8 | SQRT_SEGMENTS | 3,840 | 3,840 | 76.74% | 16,640 | 13 | 208.153 | 1.652× |
| 8 | EVERY_K, k=8 | 3,264 | 3,264 | 80.23% | 16,064 | 36 | 347.909 | 2.761× |
| 16 | NONE | 31,872 | 31,872 | 0.00% | 57,472 | 0 | 252.628 | 1.000× |
| 16 | EVERY_LAYER | 5,952 | 5,952 | 81.33% | 31,552 | 16 | 365.811 | 1.448× |
| 16 | SQRT_SEGMENTS | 4,224 | 4,224 | 86.75% | 29,824 | 28 | 429.435 | 1.700× |
| 16 | EVERY_K, k=16 | 3,264 | 3,264 | 89.76% | 28,864 | 136 | 1,091.355 | 4.320× |

The optimized and ASan/UBSan/LSan runs each checked 5,440 central differences
over every parameter and input element of multi-layer fixtures spanning all
four shape rows, with dropout disabled and enabled. Maximum absolute error
was `8.10525438e-06` against the criterion
`7e-6 + 0.003 × abs(analytic_gradient)`. Both runs also passed 136
`EVERY_K` schedules, 16 square-root schedules, 64 forward-only cases,
independent RNG mask goldens, causal and batch isolation checks, and 53
rejected runs plus four planner errors.

At each of 4, 8, and 16 layers, all four policies had byte-identical
predictions, loss, input gradients, parameter gradients, and updated
parameters across eight SGD updates at dropout rates 0 and 0.2. The training
loss with the current step's mask decreased on every update, and the fixed
dropout-free evaluation loss decreased across the full sequence. At 16 layers
with training dropout, evaluation loss decreased from `0.187536549` to
`0.0771545581`.

### Gradient replay candidate

The same pinned image, CPU model, GQA shape and timing protocol measured the
new policy against `EVERY_K` with `k = L`. Both used a 3,264-byte activation
arena. Replay staged 1,600 bytes of parameter gradients at all three depths.

| Layers | EVERY_K workspace B | Replay workspace B | Workspace saved | Replay forwards | Median µs/call | Time vs. EVERY_K |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 9,664 | 4,864 | 49.67% | 10 → 20 | 140.112 → 188.553 | 1.346× |
| 8 | 16,064 | 4,864 | 69.72% | 36 → 72 | 364.379 → 506.040 | 1.389× |
| 16 | 28,864 | 4,864 | 83.15% | 136 → 272 | 1,124.097 → 1,610.385 | 1.433× |

The optimized and ASan/UBSan/LSan suites each passed 5,440 central
differences for replay gradients across all four supported shapes, 136
depth/stride combinations for each `EVERY_K` policy, and byte-identical
predictions, loss, gradients, and parameters after each of eight SGD steps
across all five policies. A 16-layer GQA stack with a short tail at `k = 5`
preserved every output byte when either a MOD5 backward input or a
projection weight gradient overflowed at the last layer visited in
reverse. The suite also checked malformed shapes and spans, unsupported
FP controls, and short workspace.

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
