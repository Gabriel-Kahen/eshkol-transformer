# MOD5 transformer variants

MOD5 supplies RMSNorm, SwiGLU, and causal variant attention through K1 ABI 1.0.
The source-tree archive is `build/mod5/libeshkol_transformer_mod5.a`. Link it with
K1 and `-lm` and pass `et_mod5_kernel_provider_v1()` through an explicit K1
resolver. The archive exports only that accessor. Provider-free K1 discovery
retains its existing behavior. The MOD5 ABI macros version the operation schemas
below. All descriptors and errors use the existing K1 types.

Views borrow caller-owned dense row-major storage synchronously at offset zero on
`cpu`. Input payloads are read-only and pairwise disjoint. Each output payload
is disjoint from all inputs and other outputs. Call and view descriptors and
their metadata remain immutable throughout dispatch. Float operands must be
finite, including elements excluded by an attention mask. Integer views use exact
signed `i64` and bool views use bytes 0 and 1. Scalar views have rank zero.

## Capability rows

The provider and implementation name is `mod5.cpu-f32.serial`, version `1.0`,
evidence `MOD5:cpu-f32-variants-v1`. All three capabilities report
`VERIFIED`, `deterministic = 1`, compute dtype `f32`, and device `cpu`. Each
row is an exact min=max shape and applies to both operations in its capability.

| Capability | Operations | Exact request shapes |
|---|---|---|
| `kernel.rmsnorm` | `rmsnorm.forward`, `rmsnorm.backward` | `[1,1,1]`, `[1,1,4]`, `[1,2,4]`, `[2,3,4]`, `[2,3,8]` |
| `kernel.swiglu` | `swiglu.forward`, `swiglu.backward` | `[1,1,1]`, `[1,1,8]`, `[1,2,4]`, `[2,3,4]`, `[2,3,8]` |
| `kernel.variant-attention` | `variant-attention.forward`, `variant-attention.backward` | `[1,2,1,2,2,2]`, `[1,4,1,3,3,2]`, `[2,4,1,2,3,4]`, `[1,4,2,3,3,2]`, `[2,4,2,2,3,4]`, `[1,2,2,3,3,2]` |

## Tensor schemas

RMSNorm and SwiGLU use request shape `[N,T,D]`. Variant attention uses
`[N,Hq,Hkv,Tq,Tk,Dh]`, with positive extents and `Hq % Hkv == 0`. The attention
rows cover MQA (`Hkv == 1`), GQA (`1 < Hkv < Hq`), and MHA (`Hkv == Hq`),
including two-batch rectangular MQA and GQA. Query head `h` uses KV head
`h / (Hq / Hkv)` by integer division.

| Operation | Inputs in order | Outputs in order |
|---|---|---|
| `rmsnorm.forward` | x `f32[N,T,D]`, gamma `f32[D]`, epsilon `f32[]` | y `f32[N,T,D]` |
| `rmsnorm.backward` | x, gamma, epsilon, upstream `f32[N,T,D]` | dX `f32[N,T,D]`, dGamma `f32[D]` |
| `swiglu.forward` | gate `f32[N,T,D]`, up `f32[N,T,D]` | y `f32[N,T,D]` |
| `swiglu.backward` | gate, up, upstream `f32[N,T,D]` | dGate `f32[N,T,D]`, dUp `f32[N,T,D]` |
| `variant-attention.forward` | Q, K, V, query_pos, key_pos, keep_mask, alibi_slopes, window_size | out `f32[N,Hq,Tq,Dh]` |
| `variant-attention.backward` | The same eight inputs, then upstream `f32[N,Hq,Tq,Dh]` | dQ, dK, dV with the corresponding operand shapes |

| Attention input | Type and shape | Domain |
|---|---|---|
| Q | `f32[N,Hq,Tq,Dh]` | finite |
| K, V | `f32[N,Hkv,Tk,Dh]` each | finite |
| query_pos | `i64[N,Tq]` | strictly increasing per batch in `[0,16777215]` |
| key_pos | `i64[N,Tk]` | strictly increasing per batch in `[0,16777215]` |
| keep_mask | `bool[N,Tq,Tk]` | exact 0 or 1 per key |
| alibi_slopes | `f32[Hq]` | finite, all zeros disables ALiBi |
| window_size | `i64[]` | `1 <= window_size <= INT64_MAX` |

Epsilon must be strictly positive, including rejection of both signed zeros.
Epsilon, positions, mask, slopes, and window size are fixed configuration
operands. Each backward operation computes the analytic vector-Jacobian product
for the listed gradient outputs.

## Binary32 arithmetic

The provider evaluates in serial f32 with contraction and fast math disabled.
Reductions begin at positive zero. A nonfinite intermediate or result rejects
the call. Finite subnormal results and underflow to zero are admitted.

### RMSNorm

For each row in batch-major, token-major order, accumulate the squares in
increasing feature order. The forward order is:

```text
sum      = sum + x[d] * x[d]
mean_sq  = sum / (float)D
rstd     = 1.0f / sqrtf(mean_sq + epsilon)
norm[d]  = x[d] * rstd
y[d]     = norm[d] * gamma[d]
```

Backward recomputes `rstd` and `norm` in the same order. Accumulate `dot` over
features and each `dGamma[d]` over batch, then token:

```text
h[d]       = upstream[d] * gamma[d]
dot        = dot + h[d] * norm[d]
mean       = dot / (float)D
dX[d]      = (h[d] - norm[d] * mean) * rstd
dGamma[d]  = dGamma[d] + upstream[d] * norm[d]
```

### SwiGLU

Each element follows the order below. The direct exponential is part of the ABI.
For example, a sufficiently negative finite gate can cause `expf(-gate)` to
overflow and reject even when an algebraically rearranged expression would
remain finite.

```text
sig    = 1.0f / (1.0f + expf(-gate))
silu   = gate * sig
y      = silu * up
dUp    = upstream * silu
dGate  = (upstream * up) * (sig * (1.0f + gate * (1.0f - sig)))
```

### Variant attention

A key participates when its keep-mask byte is 1, `key_pos <= query_pos`, and
`query_pos - key_pos < window_size`. The position bound
`ET_MOD5_MAX_EXACT_POSITION` guarantees safe signed subtraction and exact
binary32 conversion of every difference, including differences between the
endpoints. A window of one admits only equal positions. A window larger than the
maximum position admits every otherwise eligible causal key.

For each admitted key, accumulate `dot = dot + Q[d] * K[d]` in feature order.
Then compute `scale = 1.0f / sqrtf((float)Dh)`,
`bias = alibi_slopes[h] * (float)(key_pos - query_pos)`, and
`score = (dot * scale) + bias`. Find the maximum over admitted scores.
In increasing key index, calculate `weight = expf(score - max)` and sum the
weights. Divide each weight by that denominator to get `p`. Forward accumulates
`out[d] = out[d] + p * V[d]` in the same key order.

Backward first computes `dp` for each admitted key as a feature-ordered dot
product of upstream and V. It then reduces `mean_dp = mean_dp + p * dp` in key
order and updates the gradients:

```text
ds     = p * (dp - mean_dp)
ddot   = ds * scale
dQ[d]  = dQ[d] + ddot * K[d]
dK[d]  = dK[d] + ddot * Q[d]
dV[d]  = dV[d] + p * upstream[d]
```

Each dQ row sums in key order. Shared dK and dV accumulate contributions in
query-head-major, query-token-major order within each batch. A fully masked query
row emits positive zeros in out and dQ and contributes nothing to dK or dV.
Excluded keys contribute neither values nor gradients.

## Validation and failure atomicity

K1 validates the generic descriptors and capability before calling MOD5.
MOD5 verifies its exact operand schemas, alignment, alias rules, and FP
environment. On x86-64, both x87 and MXCSR must use round-to-nearest and mask
all exceptions. MXCSR must have FTZ and DAZ disabled. Preexisting exception
flags are allowed.

Validation runs the evaluator into fixed-size automatic buffers, restoring the
caller's floating-point environment and `errno` on either numeric outcome.
Invoke repeats the evaluator and publishes the complete output spans. As required
by K1, inputs, descriptors, and floating-point controls stay stable across the
two phases. Under this contract invoke is non-failing. The K1 dispatch boundary
leaves all input and output bytes unchanged on a recoverable failure. Invoke may
set sticky exception flags or update `errno` as a result of finite libm evaluation.

| Condition | Error category | Error code |
|---|---|---|
| Valid request outside the advertised capability, operation, dtype, device, or exact shape rows | `UNSUPPORTED` | K1 `CAPABILITY_NOT_VERIFIED`, direct `PROVIDER_REJECTED` |
| Tensor dtype differs from operation schema | `DTYPE_MISMATCH` | `INVALID_TEXT` |
| Tensor device differs from the request | `DEVICE_MISMATCH` | `INVALID_BUFFER` |
| Layout differs from dense row-major or offset is nonzero | `NONCONTIGUOUS` | `INVALID_BUFFER` |
| Truncated descriptor or invalid view prefix relative to stride | `VERSION_MISMATCH` | `INVALID_STRUCT_SIZE` |
| Tensor table span mismatch or arithmetic overflow | `INVALID_ARGUMENT` | `INTEGER_OVERFLOW` |
| Wrong operation-specific tensor count or misaligned table | `INVALID_ARGUMENT` | `INVALID_BUFFER` |
| Tensor rank or extent differs from operation schema | `SHAPE_MISMATCH` | `INVALID_SHAPE` |
| Tensor byte length differs from dtype and shape | `SHAPE_MISMATCH` | `INVALID_BUFFER` |
| Misaligned shape storage or shape address-span overflow | `INVALID_ARGUMENT` | `INVALID_BUFFER` |
| Data alignment failure | `INVALID_ARGUMENT` | `PROVIDER_REJECTED` |
| Data address-span overflow | `INVALID_ARGUMENT` | K1 `ALIASING_OUTPUT`, direct `PROVIDER_REJECTED` |
| Output overlaps an input or another output | `INVALID_ARGUMENT` | `ALIASING_OUTPUT` |
| Input alias, invalid epsilon, position, mask, or window, or nonfinite arithmetic | `INVALID_ARGUMENT` | `PROVIDER_REJECTED` |
| Incompatible or unreadable FP environment | `UNSUPPORTED` | `PROVIDER_REJECTED` |
| Failure to restore FP environment | `INTERNAL` | `PROVIDER_REJECTED` |

Malformed requests can fail generic K1 validation before MOD5 checks its exact
schema. Error storage remains separate from descriptors and tensor payloads.

## Validation results and scope

Run `bash scripts/build-mod5.sh` to build and audit the deterministic archive.
Run `bash scripts/test-mod5.sh` for the focused suite. The scripts default to
`clang` and `clang++` and honor `CC` and `CXX`. Set
`MOD5_ASAN_DETECT_LEAKS=1` on lanes that support LeakSanitizer.

The focused suite passed with Clang 21.1.8 on Ubuntu 22.04 x86-64, including
C++17 linkage, ASan, UBSan, LSan, Python isolation, and CI topology. It covers all 16 exact
shape rows and both operations per row, unsupported capabilities, malformed views
and tables, float controls and status preservation, aliasing, byte-for-byte
failure atomicity, and deterministic repeatability. Central finite differences
check all 2,324 gradient elements. Attention cases cover each head layout with zero
and nonzero ALiBi, a one-position window, wider windows, selective and fully
masked rows, and the largest exact position difference.

The independent double-precision Python oracle compared 3,478 forward and
backward values in 40 cases with maximum absolute error `1.03e-7`. The native
training smoke composed two RMSNorms, Q/K/V and output projections, ALiBi
sliding-window attention, SwiGLU projections, and two residual paths. It checked
416 MQA and 448 GQA input and parameter derivatives and required all gradients
to remain finite during training. Two runs per variant produced bit-identical
loss sequences and final parameters. Each of 24 gradient-descent updates
decreased the loss.

| Training variant | Initial loss | Final loss |
|---|---:|---:|
| MQA | 0.000956520166 | 0.000770789451 |
| GQA | 0.00113966946 | 0.000730001387 |

Dispatch uses fixed-size stack scratch and has no allocator, foreign runtime,
or device fallback. The training smoke's projections and gradient descent are
test code. Python runs only as a development oracle. Support is bounded to
the listed serial CPU f32 rows and the reviewed x86-64 libm lane. Production
carrier and model wiring, compiler reverse AD, GPU, mixed precision, arbitrary
shapes, and performance measurements are outside this provider scope.
