# N2 transformer primitives

Status: **accepted and complete** within the exact N2 domains. The historical
post-merge oracle failure and superseding supported retest are recorded explicitly
in [Wave 2 primitive acceptance](WAVE2_PRIMITIVES_ACCEPTANCE.md). The ABI and
operation contract below are frozen from the
accepted integration decision after independently approved I2 PR #53 merged as
`309de7262ebe33120e782ffc1c12f8cc10cbe74b`. N2 adds no A0 Eshkol name, tensor
carrier, parameter registry, serialized format, or canonical K1 provider symbol.

## Native surface and composition

The C11 header is
`include/eshkol_transformer/n2_primitives_abi.h`. ABI 1.0 has one accessor:

```c
const et_kernel_provider_v1 *et_n2_kernel_provider_v1(void);
```

The deterministic archive is
`build/n2/libeshkol_transformer_n2.a`, containing only
`n2_primitives_provider.o`. Its provider name/implementation is
`n2.cpu-f32.serial`, version `1.0`, and evidence tag
`N2:cpu-f32-primitives-v1`. It is immutable for process lifetime, retains no
call storage, registers nothing globally, and does not define
`eshkol_transformer_kernel_provider_v1`.

A future Wave-2 aggregate collects provider accessors and owns whole capabilities.
It rejects duplicate capability names before publication and routes each unchanged
request, descriptor, and storage identity to exactly one owner. It does not merge
operation lists or K1 row rectangles from providers sharing a capability name.

## Exact capability rows

Each row is an exact K1 min=max alternative. Every operation/row Cartesian
combination is implemented. A zero extent or absent row is unsupported during K1
matching; inconsistent operand descriptors on an admitted row are shape mismatch.

| Capability | Operations | Exact request shapes |
|---|---|---|
| `kernel.embedding-forward` | `embedding.forward` | `[1,1,1,1]`, `[1,4,3,2]`, `[2,3,5,6]` |
| `kernel.embedding-backward` | `embedding.backward` | `[1,1,1,1]`, `[1,4,3,2]`, `[2,3,5,6]` |
| `kernel.matmul` | `linear.backward`, `linear.backward-no-bias`, `linear.forward`, `linear.forward-no-bias` | `[1,1,1,1]`, `[1,2,3,4]`, `[2,3,4,5]` |
| `kernel.norm` | `layer-norm.backward`, `layer-norm.forward` | `[1,1,1]`, `[1,1,3]`, `[1,2,4]`, `[2,2,4]` |
| `kernel.activation` | `gelu.backward`, `gelu.forward`, `relu.backward`, `relu.forward` | `[1,1,1]`, `[1,1,8]`, `[1,2,5]`, `[2,2,4]` |
| `kernel.dropout` | `dropout.eval.backward`, `dropout.eval.forward`, `dropout.train.backward`, `dropout.train.forward` | `[1,1,1]`, `[1,1,2]`, `[1,1,3]`, `[1,1,4]`, `[1,2,5]` |
| `kernel.residual` | `residual.backward`, `residual.forward` | `[1,1,1]`, `[1,3,4]`, `[2,2,4]` |

All rows advertise only compute dtype `f32`, device `cpu`, and deterministic true.
They do not advertise `tensor.f32`, `tensor.i64`, `tensor.contiguous`, reverse AD,
f16/bf16/f64, accelerators, broader shapes, or performance.

## Ordered tensor tables and VJPs

All views are naturally aligned, dense row-major, zero-offset CPU views with exact
K1-checked spans. Inputs are synchronous read-only borrows. Outputs are
caller-owned, preallocated, mutually disjoint, fully overwritten on success, never
retained, and independent of their initial bytes. Every backward operation is the
unnormalized VJP of `sum(forward_output * upstream)`; no output is accumulated into
a parameter slot.

| Operation and request shape | Ordered inputs | Ordered outputs |
|---|---|---|
| `embedding.forward [N,T,V,D]` | signed exact IDs `i64[N,T]`; weight `f32[V,D]` | y `f32[N,T,D]` |
| `embedding.backward [N,T,V,D]` | IDs; upstream `f32[N,T,D]` | dWeight `f32[V,D]` |
| `linear.forward [N,T,Din,Dout]` | x `f32[N,T,Din]`; weight `f32[Dout,Din]`; bias `f32[Dout]` | y `f32[N,T,Dout]` |
| `linear.forward-no-bias` | x; weight | y |
| `linear.backward` | x; weight; upstream `f32[N,T,Dout]` | dX; dWeight; dBias `f32[Dout]` |
| `linear.backward-no-bias` | x; weight; upstream | dX; dWeight |
| `layer-norm.forward [N,T,D]` | x `f32[N,T,D]`; gamma `f32[D]`; beta `f32[D]`; epsilon `f32[]` | y `f32[N,T,D]` |
| `layer-norm.backward` | x; gamma; epsilon; upstream `f32[N,T,D]` | dX; dGamma `f32[D]`; dBeta `f32[D]` |
| `gelu.forward` / `relu.forward [N,T,D]` | x `f32[N,T,D]` | y, same shape |
| `gelu.backward` / `relu.backward` | original x; upstream, same shape | dX, same shape |
| `residual.forward [N,T,D]` | x; branch, same shape | y, same shape |
| `residual.backward` | upstream | dX; dBranch, same shape |
| `dropout.train.forward [N,T,D]` | x; p `f32[]`; original RNG `i64[4]` | y; successor RNG `i64[4]` |
| `dropout.train.backward` | upstream; p; original pre-forward RNG | dX only |
| `dropout.eval.forward` / `dropout.eval.backward` | x or upstream | byte-identical y or dX |

IDs, epsilon, probability, and RNG state do not differentiate. Bias and beta are
omitted from their backward inputs because their VJPs do not depend on those primal
values; dBias and dBeta remain mandatory outputs.

Input ranges are pairwise disjoint except that the two `residual.forward` inputs
may be disjoint or the exact same descriptor and storage. Partial overlap and every
other input overlap reject as `ET_KERNEL_ERROR_INVALID_ARGUMENT` with
`ET_KERNEL_CODE_PROVIDER_REJECTED`. They never use K1's
`ET_KERNEL_CODE_ALIASING_OUTPUT`, which remains reserved for K1-detected
output/input and output/output overlap.

For an exact residual input alias, backward still emits two per-edge VJPs. Future
composition recognizes the one canonical I2 handle, adds the two contributions in
fixed residual-edge order, and submits one contribution for that unique handle.

## Binary32 evaluation contract

The provider requires x86-64, eight-bit bytes, IEEE-754 C binary32, and
`FLT_EVAL_METHOD==0`. It is compiled without fast math, with contraction disabled
and standard binary32 excess precision. Dispatch requires round-to-nearest-even,
masked x86 FP exceptions, and FTZ/DAZ disabled. Unsupported control state rejects;
N2 does not repair it.

Validation snapshots and restores the complete incoming floating environment around
its dry evaluation, including independent x87 and MXCSR status. Invoke may accrue
ordinary flags from the same successful calculation. Callers serialize dispatch
and may not change storage or FP control between K1's successful validate and
invoke. All floating inputs and every intermediate/result must be finite. Every sum
starts at positive `0.0f`; each written arithmetic statement is one binary32
operation.

- Embedding forward copies selected row bits. dWeight `[v,d]` scans increasing n,
  then t, adding upstream only for matching IDs. Unused rows are exact positive
  zero. Every ID is signed exact i64 and must satisfy numeric `0 <= id < V`; a
  negative or `id >= V` rejects before row access.
- Linear y `[n,t,o]` scans increasing i and performs
  `sum=sum+x[n,t,i]*weight[o,i]`, then the bias variant performs
  `y=sum+bias[o]`. dX `[n,t,i]` scans o and adds `upstream[n,t,o]*weight[o,i]`.
  dWeight `[o,i]` scans n then t and adds `upstream[n,t,o]*x[n,t,i]`. dBias
  `[o]` scans n then t and adds upstream.
- LayerNorm uses a two-pass biased variance. It sums x in increasing d and divides
  once by D, then sums `(x-mean)*(x-mean)` and divides once by D. It computes
  `adjusted=variance+epsilon`, `root=sqrtf(adjusted)`, `rstd=1.0f/root`,
  `normalized=(x-mean)*rstd`, and `y=(normalized*gamma)+beta` in that order.
  Backward sums `h=upstream*gamma` and `h*normalized` independently in increasing
  d, then sets `factor=rstd*(1.0f/(float)D)` and
  `dX=factor*(((float)D*h-sum_h)-normalized*sum_h_normalized)` with the shown
  parentheses. dGamma `[d]` scans n then t and adds `upstream*normalized`; dBeta
  scans n then t and adds upstream. Epsilon must be finite and numerically greater
  than zero; both signed zeros, negative values, infinities, and NaN reject before
  evaluation or mutation.
- GELU uses the exact-erf form with `half=0x1p-1f`, `one=0x1p+0f`,
  `inv_sqrt_2=0x1.6a09e6p-1f`, and
  `inv_sqrt_2pi=0x1.988454p-2f`. Forward computes `scaled=x*inv_sqrt_2`,
  `erf_value=erff(scaled)`, `sum=one+erf_value`, `half_x=half*x`, and
  `y=half_x*sum`. Backward recomputes the first three values, then computes
  `cdf=half*sum`, `square=x*x`, `exponent=-(half*square)`,
  `density=expf(exponent)*inv_sqrt_2pi`, `slope=cdf+x*density`, and
  `dX=upstream*slope`, in that order. Libm and LayerNorm `sqrtf` bit determinism is
  limited to the supported Ubuntu 22.04 / LLVM-Clang 21.1.8 lane.
- ReLU returns x only for `x>0.0f`. Negative values and both signed zeros produce
  positive zero; the VJP is zero at the same kink.
- Residual forward performs one `x+branch`. Backward byte-copies upstream to both
  outputs.

## Dropout RNG v1

State is exact signed `i64[4]`: numeric version `+1`, a 64-bit key, then low and
high halves of a 128-bit counter. The latter three words encode unsigned bit
patterns numerically: values through `INT64_MAX` are unchanged; larger values map
to `u-2^64`. Unknown versions reject as version mismatch.

Philox4x32-10 uses little-word counter order and constants:

```text
M0=d2511f53  M1=cd9e8d57  W0=9e3779b9  W1=bb67ae85
```

For each round, with exact unsigned 32x32 products `(hi0,lo0)=M0*c0` and
`(hi1,lo1)=M1*c2`, the simultaneous update is:

```text
c0 = hi1 xor c1 xor k0
c1 = lo1
c2 = hi0 xor c3 xor k1
c3 = lo0
```

After rounds zero through eight, add W0/W1 to k0/k1 modulo 2^32. Returned lane
order is c0,c1,c2,c3. Element j uses lane `j%4` at counter
`C+floor(j/4)`. The uniform is
`(float)(lane>>8) * 0x1p-24f`, and a value is kept iff `uniform>=p`.

The normative Random123 vectors are:

```text
ctr 00000000 00000000 00000000 00000000
key 00000000 00000000
out 6627e8d5 e169c58d bc57ac4c 9b00dbd8

ctr ffffffff ffffffff ffffffff ffffffff
key ffffffff ffffffff
out 408f276d 41c83b0e a20bc7c6 6d5451fd

ctr 243f6a88 85a308d3 13198a2e 03707344
key a4093822 299f31d0
out d16cfe09 94fdcceb 5001e420 24126ea1
```

For positive p, compute `scale=1.0f/(1.0f-p)` and then `input*scale`. Each call
starts at lane zero and discards tail lanes. Block count is
`E/4 + (E%4 != 0)`. The all-ones 128-bit counter is exhausted: max-1 plus one
block may return max, but positive-p work starting at max rejects before writes.
Forward emits the successor; backward receives the original state, regenerates the
mask, and emits no successor.

Both signed probability zeros byte-copy input and preserve the complete state with
no Philox work. Eval has no probability or RNG argument and byte-copies input.
Other probabilities must be finite and satisfy `0<p<1`.

## Validation and failure atomicity

K1 validates the generic descriptor prefix, exact spans, layout/device, output
disjointness, and capability row first. N2 then validates the ordered tensor table,
alignment, input aliases, IDs, epsilon/probability/RNG, finite values, and FP
environment. It dry-evaluates every prospective result with the same helpers and
order as invoke. Only successful preflight reaches allocation-free, nonfailing
invoke. Every recoverable failure leaves all input and output bytes unchanged.

The exact error mapping is:

| Condition | Category | Code |
|---|---|---|
| absent capability/operation, request dtype/rank, zero or out-of-row request | `UNSUPPORTED` | K1 `CAPABILITY_NOT_VERIFIED`; direct provider `PROVIDER_REJECTED` |
| operand device differs from request device | `DEVICE_MISMATCH` | K1 `INVALID_BUFFER` |
| consistent non-CPU request and views | `UNSUPPORTED` | K1 `CAPABILITY_NOT_VERIFIED`; direct provider `PROVIDER_REJECTED` |
| admitted-row operand rank/extent mismatch | `SHAPE_MISMATCH` | `INVALID_SHAPE` |
| operand dtype mismatch | `DTYPE_MISMATCH` | `INVALID_TEXT` |
| layout or nonzero offset | `NONCONTIGUOUS` | `INVALID_BUFFER` |
| truncated call/request/view struct or table stride | `VERSION_MISMATCH` | `INVALID_STRUCT_SIZE` |
| malformed tensor-table byte span or count overflow | `INVALID_ARGUMENT` | K1 `INTEGER_OVERFLOW` |
| tensor-table pointer/stride alignment or wrong operation-specific count | `INVALID_ARGUMENT` | `INVALID_BUFFER` |
| view byte length differs from dtype/shape | `SHAPE_MISMATCH` | K1 `INVALID_BUFFER` |
| view shape storage alignment/address span | `INVALID_ARGUMENT` | K1 `INVALID_BUFFER` |
| admitted view data alignment | `INVALID_ARGUMENT` | `PROVIDER_REJECTED` |
| view data address-span overflow | `INVALID_ARGUMENT` | public K1 `ALIASING_OUTPUT`; direct provider `PROVIDER_REJECTED` |
| unknown dropout state version | `VERSION_MISMATCH` | `PROVIDER_REJECTED` |
| input overlap, ID, epsilon, p, exhausted counter, nonfinite input/intermediate/result | `INVALID_ARGUMENT` | `PROVIDER_REJECTED` |
| output/input or output/output overlap | `INVALID_ARGUMENT` | K1 `ALIASING_OUTPUT` |
| nondefault rounding, unmasked exceptions, FTZ, or DAZ | `UNSUPPORTED` | `PROVIDER_REJECTED` |

No path allocates, casts, transfers, broadcasts, materializes, falls back to
scalar containers or Python, invokes finite differences, or uses compiler reverse
AD.

## Current limitations

N2 supports only the exact rows above on serial CPU f32. Accepted I1 supplies the
exact-i64 IDs and RNG state used by the carrier-backed gate. Accepted I2 supplies
owned f32 storage, synchronous K1 borrows, P1-bound identity/ties, accumulated
gradient metadata, and atomic mutation plans. N2 remains a carrier-neutral provider:
it retains no borrowed view and does not expose I2's private parameter ABI.
RMSNorm and SwiGLU remain MOD5. There is no GPU, mixed-precision, arbitrary-shape,
performance, cross-libm bitwise, general autodiff, optimizer, checkpoint, or
concurrency claim.
