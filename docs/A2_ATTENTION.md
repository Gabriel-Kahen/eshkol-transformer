# A2 causal attention, RoPE, and KV-cache contract

## Evidence boundary

A2 is a carrier-neutral numerical and cache substrate. It adds no A0 public Eshkol
name or arity and does not provide a general owned floating tensor, a P1 tensor
provider, parameter or gradient slots, compiler reverse autodiff, model composition,
or generator composition. Those production seams depend on the separately
coordinated I2 workstream.

The pinned Eshkol core is not a fallback. Its built-in attention and RoPE paths do
not establish the required f32 storage, device identity, exact mask behavior,
reverse-mode gradients, allocation contract, or hot-path contract. A2 instead uses
the accepted K1 ABI with an explicitly linked provider obtained from
`et_a2_kernel_provider_v1`. It does not define
`eshkol_transformer_kernel_provider_v1`, perform dynamic discovery, or change K1's
provider-free baseline report.

`ET_A2_ATTENTION_ABI_MAJOR/MINOR` version only the operation schemas and semantics
documented here. Tensor, request, call, provider, and error struct layouts remain K1
ABI v1; the macros do not introduce a second descriptor ABI.

ABI-local failures use `et_kernel_error`; they are not E1 values. The private AOT
test transport reduces fixed native checks to a test-only pass/fail result. It is
neither a production error mapping nor a production tensor carrier.

## Numerical provider

All views are borrowed for one synchronous dispatch, dense row-major, zero offset,
and on the exact `cpu` device. Floating views are true `f32`; positions are `i64`;
the attention mask is `bool` encoded as bytes 0 or 1. Inputs are read-only. Outputs
are caller-owned, preallocated, mutually disjoint, and disjoint from every input.
No call allocates, materializes, casts, copies between devices, changes precision,
or selects another implementation.

Every floating operand must be finite. Every position is in
`[0,16777215]` and positions are strictly increasing within each batch row. A
malformed value is rejected during the non-mutating provider validation phase.
Only a completely validated call reaches the non-failing invoke phase.

### Causal attention

The capability is `kernel.causal-attention`. Its request shape is the semantic
six-dimensional list `[N,Hq,Hkv,Tq,Tk,Dh]`, with every extent positive,
`Hkv >= 2`, and `Hq mod Hkv = 0`.

K1 v1 cannot express dependent head constraints or even-only dimensions in one
rectangular range. A2 therefore publishes one capability with multiple disjunctive,
exact min=max shape rows. The verified attention rows are:

```text
[1,2,2,1,1,1] [1,2,2,2,2,1] [1,2,2,1,2,1]
[1,2,2,1,2,2] [1,2,2,2,2,2] [1,4,2,1,1,2]
[1,4,2,3,3,2] [2,4,2,2,3,4] [2,4,2,3,3,2]
[2,4,2,1,3,2]
```

The provider validator retains the general schema checks above as defense in
depth, but a shape absent from this list is capability-unverified and K1
capability resolution rejects it. Direct-provider behavior for a broader
schema-valid shape is implementation-only and unverified. In particular,
`Hq < Hkv`, nondivisible head counts, and otherwise schema-valid rectangles
outside the exact rows are not advertised.

`causal-attention.forward` has these tensor tables:

| Table | Index | Tensor |
|---|---:|---|
| input | 0 | Q `f32[N,Hq,Tq,Dh]` |
| input | 1 | K `f32[N,Hkv,Tk,Dh]` |
| input | 2 | V `f32[N,Hkv,Tk,Dh]` |
| input | 3 | query positions `i64[N,Tq]` |
| input | 4 | key positions `i64[N,Tk]` |
| input | 5 | keep mask `bool[N,Tq,Tk]` |
| output | 0 | result `f32[N,Hq,Tq,Dh]` |

`causal-attention.backward` uses the same first six inputs, followed by upstream
`f32[N,Hq,Tq,Dh]`, and writes dQ, dK, and dV in that order. Positions and masks do
not differentiate. This is an explicit analytic backward operation; it is not a
claim that Eshkol compiler reverse AD can differentiate through a foreign call.

Query head `hq` uses KV head
`floor(hq / (Hq / Hkv))`. This admits proved MHA (`Hq = Hkv`) and GQA
(`Hq > Hkv >= 2`). Hkv=1 MQA is not admitted in A2.

A key participates only when its keep-mask byte is 1 and its absolute key position
is no greater than the absolute query position. No broadcasting or additive mask is
defined. Scores use the fixed serial f32 order: accumulate
`sum = sum + Q[d] * K[d]`, compute `root = sqrtf((float)Dh)`, compute
`scale = 1.0f / root`, and finally compute `score = sum * scale`. Division of the
completed dot product by `root` is not an equivalent contracted implementation
for this contract. A maximum-subtracted f32 softmax follows over admitted keys.
A query row with no admitted key returns positive f32 zero in every output element.
Its dQ is zero and it contributes zero to dK and dV. The implementation uses
contraction-disabled compilation and makes no cross-platform bitwise libm claim.
The frozen PyTorch fixture remains a tolerance-based mathematical oracle; a
separate supported-platform bit regression distinguishes reciprocal-then-multiply
from divide-after-sum, so changing the frozen arithmetic order cannot hide inside
the oracle tolerance.

### Rotary positions

The capability is `kernel.rope`; the request shape is `[N,H,T,Dh]`. `Dh` is even
and at least two.

Its verified exact rows are `[1,1,1,2]`, `[1,1,2,2]`, `[1,1,2,4]`,
`[2,2,3,4]`, `[2,4,3,2]`, and `[2,2,3,2]`. An odd `Dh`, or an otherwise
schema-valid even shape outside these rows, is capability-unverified. This exact-row
encoding is how A2 truthfully represents the even-dimension contract without a K1
ABI change.

`rope.forward` consumes x `f32[N,H,T,Dh]`, positions `i64[N,T]`, and positive
finite inverse frequencies `f32[Dh/2]`, then writes y with x's exact shape.
`rope.backward` consumes upstream, positions, and the same inverse frequencies,
then writes dX. For pair `i`:

```text
angle = position * inv_freq[i]
y[2i]   = x[2i] cos(angle) - x[2i+1] sin(angle)
y[2i+1] = x[2i] sin(angle) + x[2i+1] cos(angle)
```

Backward applies the transpose/inverse rotation. Positions and inverse frequencies
are buffers and have no A2 gradient. Passing inverse frequencies keeps model-level
base, scaling, and context policy outside this ABI.

## KV cache ABI 1.0

One opaque, process-local, exclusively mutable cache owns separate finite-zero-
initialized keys and values shaped `[L,N,Hkv,C,Dh]`, shared exact i64 logical
lengths `[N]`, and a canonical bool mask `[N,C]`. Every extent is positive. Capacity
and all four storage identities
remain fixed until explicit native destruction. Growth means logical-length growth;
there is no reallocation, eviction, ring buffer, dtype conversion, transfer, or
cache fallback.

Native cache and transaction handles are admitted to private live registries before
any dereference. Forged, wrong-kind, and currently non-live addresses reject. Copies
of a live opaque pointer are aliases to the same handle; after end/destroy, every
copy is caller-invalid and must not be reused. A2 does not promise immunity if an
allocator later assigns the same address to a different live handle. Concurrency and
reentrancy are unsupported.

An append transaction has a positive width `A` and exact append counts `[N]`.
Every count satisfies `0 <= count[i] <= A`, at least one count is positive, and
`length[i] + count[i] <= C` is proved without overflow. Each layer is staged exactly
once from post-RoPE K and V views `f32[N,Hkv,A,Dh]`. Staged bytes occupy the tail
beginning at the old logical length and remain outside committed lengths.

After a layer is staged, one nested read-only view may expose that layer's full-
capacity dense K and V storage as `[N,Hkv,C,Dh]`, immutable effective lengths
`old_length + count`, and an immutable bool keep mask `[N,C]` whose bytes are exactly
one below each effective length and zero otherwise. A2 never presents a shorter
`Tk < C` tensor as canonically dense over capacity-strided storage. While a nested view is live,
stage, commit, abort, and cache destruction reject. The view must be explicitly
ended.

These descriptors and pointees are read-only by contract; K1's ABI-level `void *`
data field does not grant mutation permission. Full-capacity tail bytes remain
outside the logical domain and must be ignored wherever the mask is false.

The cache mask is key-validity metadata, not a broadcastable attention operand.
Attention requires exact `bool[N,Tq,C]`; a consumer must explicitly materialize
that shape from `[N,C]` and may combine additional per-query keep policy. A2 performs
no implicit broadcast or allocation.

Commit requires all `L` layers staged and no live view. Its only observable work is
the non-failing publication of shared lengths and mask. Abort, or any error before commit,
leaves the committed prefixes, logical lengths, storage identities, and all sources
unchanged. Cache creation deterministically initializes all K/V and mask bytes to
zero, and abort scrubs every staged K/V slot back to positive zero. Full-capacity
tail bytes are observable only under a false mask and never participate in admitted
attention. Beginning a transaction allocates only control and immutable snapshot
metadata; cache storage allocation occurs only during cache creation.
Transaction begin, transaction-view begin, and committed-read-borrow begin allocate
bounded CPU control/snapshot descriptors and report allocation failure before
publishing a lease. Numerical provider dispatch remains allocation-free.

The cache is no-grad and is not serializable. Cache-owned storage is not I2/P1
parameter or gradient storage. Cached keys are post-RoPE. The current I2 aggregate
does not import or route the A2 provider; M3/G3 must wait for separately reviewed
multi-provider composition and production ownership rather than consume A2's private
test transport.

## Verification boundary

The focused gate must prove:

- exact K1 capability metadata, tensor schemas, failure atomicity, and unchanged
  provider-free baseline discovery;
- known-value, frozen-reference, and analytic-gradient parity for MHA and GQA;
- central finite differences for Q, K, V, and RoPE input values;
- pinned independent PyTorch `N=2` vectors with distinguishable batch values,
  positions, and masks for rectangular attention forward and dQ/dK/dV;
  distinguishable values and positions for RoPE forward/backward and every cached
  GQA/RoPE incremental output;
- rejection of exact attention Q/K/V batch-zero indexing and RoPE batch-zero
  position mutations, in addition to finite-difference and full/incremental
  self-consistency checks;
- exact capability-require/report positives for every published row and negatives
  for dependent-head, odd-RoPE, and broader validator-admitted shapes;
- causal future-influence and future-gradient exclusion, selective masks, fully
  masked rows, T=1, short rectangular sequences, and repeated values;
- RoPE position-zero, maximum-position, norm, malformed dimension/frequency, and
  forward/backward cases;
- cache exact-capacity and exact-width appends, per-row counts, every incremental
  prefix against uncached full attention, deterministic repeatability, transaction
  nesting, abort, capacity-one-over, zero-width/all-zero, and overflow rejection;
- allocation failure, span/alias/disjointness, forged/wrong-kind/non-live handle, sentinel-output,
  C/C++ header, ASan, UBSan, exact-symbol, and production Python-isolation gates;
- two fresh pinned-compiler AOT builds and executions of the private Eshkol transport;
  and
- the repository-wide gates and supported Ubuntu 22.04 / LLVM-Clang 21.1.8 CI at
  the exact review head.

No benchmark or acceleration claim is part of A2.

## Source-tree packaging

A2 has no install, plugin, dynamic-loading, or environment-discovery contract. The
source-tree build creates `build/a2/libeshkol_transformer_a2.a`; native consumers
link that archive before `build/k1/libeshkol_transformer_k1.a` and the platform math
library (`-lm`). `make clean` removes the build artifact. The private AOT bridge and
PyTorch oracle remain under `tests/a2` and are never archive members.
