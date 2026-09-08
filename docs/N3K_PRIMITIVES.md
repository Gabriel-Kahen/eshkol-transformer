# N3K bounded diagnostic primitives

Status: **review; contract accepted with corrections**.
Acceptance: https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5584383610. Tracking #66,
parent #1, prerequisite audit #65 at `a97c1095b542f1f86f3273139d39af2d8ae47e93`.
Base is merged main `231f9354f14389db15faac7820ef23dc038ae941`.
The profile direction is accepted at
https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5584217290;
The specific boundary below incorporates the subsequent acceptance corrections.

## Provider identity and immutable predecessors

C11/C++ header `include/eshkol_transformer/n3k_primitives_abi.h`, macros
`ET_N3K_PRIMITIVES_ABI_MAJOR=1u`, `ET_N3K_PRIMITIVES_ABI_MINOR=0u`, sole accessor
`const et_kernel_provider_v1 *et_n3k_kernel_provider_v1(void);`.
Archive `build/n3k/libeshkol_transformer_n3k.a`, one object
`n3k_primitives_provider.o`; provider name/implementation `n3k.cpu-f32.serial`,
version `1.0`, evidence `N3K:cpu-f32-diagnostic-v1`. Descriptor is immutable for
process lifetime; discovery is explicit. No canonical K1 resolver, owned carrier,
allocation, retained view, registry, public Eshkol name or serialized format.

All capability and operation names below are N3K-specific, avoiding whole-capability
ownership collisions with N2. Existing N2/A2/K1 source, headers, schema, metadata,
versions and artifacts remain unchanged. The implementation copies only necessary
reviewed N2 numerical helpers with explicit provenance into private N3K code;
there is no N2 refactor or private accessor call. This avoids changing frozen source closure/object bytes.
No provider composition is implemented or inferred.

## Exact K1 rows and ordered operands

Every row has min=max for every dimension. Only request compute dtype `f32`, device
`cpu`, deterministic true are advertised. A request with deterministic=0 merely
does not require determinism and remains admitted; execution remains deterministic. Every listed operation/row Cartesian
combination is implemented. Zero, neighboring extents, missing rank/row, bias,
ReLU, normalization, dropout, and unlisted operations are unsupported.

| Capability | Operations | Exact request shapes |
|---|---|---|
| `n3k.embedding-forward` | `n3k.embedding.forward` | `[1,2,256,4]`, `[1,2,2,4]` |
| `n3k.embedding-backward` | `n3k.embedding.backward` | same |
| `n3k.linear` | `n3k.linear.backward-no-bias`, `n3k.linear.forward-no-bias` | `[1,2,4,4]`, `[1,2,4,8]`, `[1,2,8,4]`, `[1,2,4,256]` |
| `n3k.gelu` | `n3k.gelu.backward`, `n3k.gelu.forward` | `[1,2,8]` |
| `n3k.residual` | `n3k.residual.backward`, `n3k.residual.forward` | `[1,2,4]` |
| `n3k.head-layout` | `n3k.heads.merge.backward`, `n3k.heads.merge.forward`, `n3k.heads.split.backward`, `n3k.heads.split.forward` | `[1,2,2,2]` interpreted `[N,T,H,Dh]` |
| `n3k.activation-sum` | `n3k.sum2.backward`, `n3k.sum2.forward`, `n3k.sum3.backward`, `n3k.sum3.forward` | `[1,2,4]` |
| `n3k.tied-sum` | `n3k.sum2.backward`, `n3k.sum2.forward` | `[256,4]` |
| `n3k.matrix-init` | `n3k.matrix-init.uniform` | `[256,4]`, `[2,4]`, `[4,4]`, `[8,4]`, `[4,8]` |

All tensor tables are ordered and fixed count, no optional operands:

| Operation | Inputs | Outputs |
|---|---|---|
| embedding forward `[N,T,V,D]` | exact signed `i64[N,T]` IDs; `f32[V,D]` weight | `f32[N,T,D]` y |
| embedding backward | IDs; `f32[N,T,D]` upstream | `f32[V,D]` dWeight |
| linear forward `[N,T,Din,Dout]` | `f32[N,T,Din]` x; `f32[Dout,Din]` weight | `f32[N,T,Dout]` y |
| linear backward | x; weight; `f32[N,T,Dout]` upstream | dX same as x; dWeight same as weight |
| GELU forward | `f32[1,2,8]` x | y same shape |
| GELU backward | x; upstream same shape | dX same shape |
| residual forward | `f32[1,2,4]` skip; branch same shape | y same shape |
| residual backward | upstream | dSkip; dBranch same shape |
| split forward / merge backward | `f32[1,2,4]` token-major input/upstream | `f32[1,2,2,2]` head-major output/VJP |
| merge forward / split backward | `f32[1,2,2,2]` head-major input/upstream | `f32[1,2,4]` token-major output/VJP |
| sum2 / sum3 forward | a; b; c for sum3 only, each exact request shape | y same shape |
| sum2 / sum3 backward | upstream exact request shape | da; db; dc for sum3 only, same shape |
| matrix initializer `[R,C]` | exact `i64[4]` original initializer state | `f32[R,C]` matrix; exact `i64[4]` successor state |

Every analytic backward is the unnormalized VJP of `sum(y*upstream)`, overwrites
per-edge outputs, and accumulates into no parameter slot. IDs and initializer state
do not differentiate; initialization has no backward operation.

## Numerical and ordering semantics

Embedding and bias-free linear use the exact N2 binary32 statement order and weight
layout documented in `docs/N2_PRIMITIVES.md`. Embedding copies row bits; backward
starts every destination at +0 and scans n then t, including repeated IDs. Every
ID must satisfy numeric `0<=id<V`, checked before access. All declared floating
inputs, including unselected embedding rows, must be finite. Linear products and
sums are separate f32 operations, with +0 initial sums and increasing contracted
index; no FMA. GELU is N2's exact-erf formula/constants and analytic derivative,
with every intermediate finite check preserved (including x*x in backward).
No approximation or finite-difference runtime gradient.

For layout, token offset `((n*T+t)*H+h)*Dh+d` maps to head offset
`((n*H+h)*T+t)*Dh+d`. Materialize into disjoint dense storage by bit-preserving
copy; backward uses the mathematical inverse. Exact T=H=2 makes the flat
permutation self-inverse: tests also distinguish operand ranks/direction schemas.
No reshape/relabel, hidden copy, generalized layout or new carrier is advertised.

Residual and sum2 perform exactly one `a+b`; sum3 performs `(a+b)+c` with the
intermediate checked before adding c. Backward bit-copies upstream separately to
each edge. Ordered sum slots mean skip then branch, Q then K then V, and head then
embedding for the corresponding downstream schedules. These are numerical edge
orders, not public model names. Shared primal edges still receive separate VJPs;
composition must combine all edges before one canonical I2 contribution.

## Initializer algorithm and caller draw schedule

Algorithm identity is `n3k.philox4x32-10.uniform-f32.v1`; state belongs to this
algorithm, distinct from N2 dropout despite the compatible integer encoding. This numeric
encoding neither authenticates algorithm identity nor separates streams: equal
key/counter inputs share Philox words. Future typed transport/composition must
carry and validate algorithm identity; no such transport or checkpoint is added here.
State `i64[4]` is numeric version +1, full 64-bit seed/key, low and high words of a
128-bit block counter. Unsigned words above INT64_MAX encode as numeric `u-2^64`;
decode modulo 2^64 without implementation-defined out-of-range casts. Diagnostic
seed is 1729, initial counter 0; all seed bit patterns and representable counters
are supported subject to exhaustion.

Philox counter uses little-word order c0,c1,c2,c3, key k0,k1; constants
M0=d2511f53, M1=cd9e8d57, W0=9e3779b9, W1=bb67ae85. In each of ten rounds,
unsigned 32x32 products `(hi0,lo0)=M0*c0`, `(hi1,lo1)=M1*c2` yield simultaneous
`(hi1 xor c1 xor k0,lo1,hi0 xor c3 xor k1,lo0)`; after rounds 0 through 8,
add W0/W1 to the key modulo 2^32. Return lanes c0,c1,c2,c3.
Element j uses lane j%4 at original counter + floor(j/4), row-major order.
Compute distinct binary32 statements
`u=(float)(lane>>8)*0x1p-24f`, `centered=u-0.5f`, `w=centered*0x1p-4f`.
Equivalently w is the exactly representable dyadic
`((lane>>8)-2^23)*2^-28`, including +0 at the center. Range [-1/32,1/32).
Uniform avoids introducing a platform-dependent normal-distribution transform.

Each matrix begins at lane zero, consumes `E/4+(E%4!=0)` blocks, and discards tail
lanes. All advertised E are divisible by 4; no native partial-block evidence is
claimed. Successor preserves version/key and advances the complete 128-bit counter
by that block count. Maximum counter is an exhausted sentinel: a call may return
max as successor, but may not consume max, wrap, or partially publish matrix/state.
Unknown state version is VERSION_MISMATCH/PROVIDER_REJECTED; exhaustion is
INVALID_ARGUMENT/PROVIDER_REJECTED. No global RNG or seed dependence.

The kernel initializes one matrix and owns no parameter identities or paths.
A caller initializing a parameter collection must first validate the complete
collection, group aliases by canonical unique-storage identity with consistent
shape/dtype, choose each group's least unsigned UTF-8 byte path as representative,
sort representatives lexicographically by unsigned bytes (shorter prefix first),
and call once per group in that order, threading successor state. Constants gamma=1
and beta=+0 consume no draws; existing I2 exact-bit ingress can supply them. A tied
head/embedding group consumes once regardless of path/registration enumeration;
its representative need not be the embedding path. Eight unique diagnostic random
matrices contain 1,160 values and consume 290 blocks. Six affine constant tensors
add 24 values without draws. No public M3 path vocabulary, registry traversal API,
batch publication, rollback, or checkpoint/RNG transport schema is introduced.
Native ordered-sequence tests use test-only abstract paths/identities; they prove
kernel continuation and a reference draw schedule, not production enforcement of
ties/path ordering. Production schedule enforcement remains a downstream obligation.

## Ownership, aliases, errors, floating environment

Views are synchronous read-only inputs and caller-owned preallocated outputs,
naturally aligned, exact dtype, dense row-major, zero-offset CPU with K1-checked
spans. Outputs are fully overwritten on success, mutually disjoint and disjoint
from all inputs; K1 alias failures use INVALID_ARGUMENT/ALIASING_OUTPUT. Input
ranges must be pairwise disjoint except residual/sum forward allows equal complete
semantic views/storage for per-edge reuse (equal data range, dtype, device,
layout, offset, rank and extents; descriptor/shape-array pointer identity is not required); partial overlaps always reject with
INVALID_ARGUMENT/PROVIDER_REJECTED. Other input overlaps reject identically.
Caller keeps request/view metadata/storage stable through validate/invoke and
serializes FP controls; no storage is retained. Direct invoke requires successful full K1 generic checks plus provider
validation and stable inputs, as K1 specifies; the provider callback alone does not
replace K1 output-alias, reserved-field, or generic descriptor admission.

Requires x86-64 IEEE binary32, eight-bit bytes, FLT_EVAL_METHOD=0; compile without
fast-math/contraction/excess precision. Validate both x87 and MXCSR round-to-nearest
even and all exception masks, MXCSR FTZ/DAZ off; reject unsupported control state
without repairing it. Save/restore complete incoming fenv, including independent
x87/MXCSR flags, around the whole validation (including signaling-NaN detection),
on success and failure. Invoke may accrue ordinary flags from successful arithmetic.
Libm GELU bit determinism is scoped to supported Ubuntu22/LLVM21, not cross-libm.

K1 validates descriptor prefixes, spans, device/layout, capability rows and output
aliases. Provider validates ordered operand schemas, data alignment, input aliases,
IDs/state and every finite input, then dry-evaluates every intermediate/result with
the same helpers/order as invoke. Only successful preflight reaches allocation-free
nonfailing invoke. Recoverable errors preserve all input/output/state bytes. Direct
validate checks call/request struct prefixes before reading later fields.

Error mapping follows frozen N2/K1 categories: absent operation/row/request dtype
or well-formed non-CPU request is UNSUPPORTED/CAPABILITY_NOT_VERIFIED through K1
(PROVIDER_REJECTED direct); admitted operand rank/extent mismatch is
SHAPE_MISMATCH/INVALID_SHAPE; dtype mismatch DTYPE_MISMATCH/INVALID_TEXT;
operand/request device mismatch DEVICE_MISMATCH/INVALID_BUFFER through K1;
noncontiguity/offset NONCONTIGUOUS/INVALID_BUFFER; truncated prefix/stride
VERSION_MISMATCH/INVALID_STRUCT_SIZE; malformed count/alignment/span
INVALID_ARGUMENT/INVALID_BUFFER (K1 INTEGER_OVERFLOW for tensor-table count/span arithmetic);
tensor shape-product overflow SHAPE_MISMATCH/INTEGER_OVERFLOW; tensor-data address
overflow INVALID_ARGUMENT/ALIASING_OUTPUT through K1 and
INVALID_ARGUMENT/PROVIDER_REJECTED directly;
view byte length mismatch SHAPE_MISMATCH/INVALID_BUFFER through K1; admitted data
alignment INVALID_ARGUMENT/PROVIDER_REJECTED; nonfinite input/intermediate/result,
invalid ID and forbidden input overlap INVALID_ARGUMENT/PROVIDER_REJECTED;
unsupported FP control UNSUPPORTED/PROVIDER_REJECTED. Direct provider view-device
failures use DEVICE_MISMATCH/INVALID_TEXT and view-byte/data-span failures use
INVALID_ARGUMENT/PROVIDER_REJECTED, matching N2; public K1 categories above remain
authoritative for dispatch. No allocations, casting,
broadcasting, fallback, device transfer, generic fill, general bool/i64 ingress,
loss reduction, compiler AD, optimizer/model lifecycle or public Eshkol transport.

## Acceptance coverage

Every exact operation/row: independent forward and analytic-VJP reference plus
scaled central differences, repeated/boundary IDs, marked layout/inverse tests,
fixed-order sums with wrong/omitted-edge sensitivity, initializer literal bits and
whole-tensor hashes, seed/continuation/carry/exhaustion and ordered unique-storage
reference sequence; malformed shape/dtype/device/alias/nonfinite/fenv and byte
preservation; exact-row neighbors. Native production tests, I1/I2 owned borrows,
ASan/UBSan/LSan, immutable K1 discovery baseline, C/C++ ABI, exact symbol/source/
archive isolation, deterministic builds, two fresh private pinned-Eshkol AOT runs,
meaningful affected gates and supported Ubuntu22/LLVM21 CI. Private AOT does not
claim a public model. Independent Astra-high reviews cover numerical operations,
composition/initializer tests, and ownership/fenv/packaging. No performance, GPU,
training, model quality, full M3, JIT or resume claim follows from N3K.


## Executed evidence and limits

The focused gate is `make test-n3k`. The implementation was locally validated on
explicitly unsupported CachyOS x86-64 / Clang-LLVM 22.1.6 with a freshly built,
fully revalidated pinned Eshkol `90cbd7130f47b8184bcc77b8d5c1b0026da980de`,
version `1.3.4-evolve`. The supported Ubuntu 22.04 / LLVM-Clang 21.1.8 run and
exact PR head are recorded in tracking issue #66 and the integration handoff;
local compatibility evidence is not a substitute for that required lane.

All shell execution uses `/usr/bin/bash` without login initialization. Reproduce
from a clean supported checkout with `make toolchain`, `make configure`,
`make build`, then `N3K_ASAN_DETECT_LEAKS=1 make test-n3k`.
The executed local compatibility commands were:

```sh
ESHKOL_ALLOW_UNSUPPORTED_HOST=1 LLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config \
  CC=/usr/bin/clang CXX=/usr/bin/clang++ ESHKOL_JOBS=4 \
  /usr/bin/bash scripts/bootstrap-eshkol.sh
ESHKOL_ALLOW_UNSUPPORTED_HOST=1 LLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config \
  /usr/bin/bash scripts/configure.sh
# Build each required canonical archive once, including I2's existing aggregate.
ESHKOL_ALLOW_UNSUPPORTED_HOST=1 LLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config \
  /usr/bin/bash scripts/build-k1.sh
ESHKOL_ALLOW_UNSUPPORTED_HOST=1 LLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config \
  /usr/bin/bash scripts/build-i1.sh
ESHKOL_ALLOW_UNSUPPORTED_HOST=1 LLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config \
  /usr/bin/bash scripts/build-i2.sh
ESHKOL_ALLOW_UNSUPPORTED_HOST=1 LLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config \
  /usr/bin/bash scripts/build-n3k.sh
ESHKOL_ALLOW_UNSUPPORTED_HOST=1 LLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config \
  N3K_ASAN_DETECT_LEAKS=1 /usr/bin/bash scripts/test-n3k.sh
```

| Gate | Result |
|---|---|
| All advertised rows | 31 operation/row pairs across nine exact capabilities; metadata, dtype/device/rank/neighbor rejections pass |
| Independent mathematical oracles | 23 Python tests; 20 primitive reference cases and 31 initializer vector cases, generated deterministically without tracked large fixtures |
| Primitive native suite | 15,227 checks and 3,248 actual-native central derivatives; maximum reference absolute error `7.62939453e-06` |
| Composition/initializer native suite | 12,944 checks; materialized marked layouts and inverse VJPs, all sum edges, exact matrix bits/endpoints, signed seeds, continuation/carry/exhaustion, eight-unique reference schedule |
| Negative native suite | 8,328 checks; every operation/row, malformed prefixes/descriptors, both boundary-invalid ID positions, forbidden exact/partial aliases, nonfinite/intermediate overflow, complete output preservation, independent x87/MXCSR flags/controls |
| I1/I2 native integration | 31,458 checks; all 31 rows through owned borrows, active-borrow mutation/destruction blocking, release and unchanged I2 resource counts, one summed tied contribution with duplicate destination rejection |
| Sanitizers | All four suites (67,957 checks) and private bridge pass ASan/UBSan with LSan `detect_leaks=1`; no diagnostics |
| Packaging/AOT | Exact one-member/one-accessor archive, compiler-derived three-file input closure, reviewed undefined symbols, no FMA/binary64/allocator/loader/test/global-resolver leaks, C11/C++17, frozen predecessor hashes and K1 baseline, identical fresh objects/archives and two fresh private AOT binaries/stdout |

Three independent Astra-high reviews cover numerical operation/schema/oracle
correctness, permutation/sum/initializer bits/errors, and native ownership/fenv/
packaging. Numerical and packaging review found no implementation defect. The
negative-suite review identified missing invalid-ID and ordinary-input alias cases;
these were added with repeated-ID scatter overflow and independently rerun under
ASan/UBSan/LSan. No unresolved review blocker remains in this task; integration
still owns the separate integration review and merge decision.

Native carrier comparison uses the same provider with raw and owned views;
independent mathematical parity is established by the separate primitive and
composition suites. Eight-matrix sequence fixtures prove numerical continuation
and a reference schedule, not production model tie/path enforcement. Equal T/H
makes the layout permutation self-inverse, so direction rank contracts are tested
separately. Sum3 reassociation and omitted/duplicated edges are observable; swapping
the two operands of a finite binary sum is commutative and is not a proven caller
wiring distinction. No native row exercises discarded partial Philox blocks.
There is no general arbitrary-shape, accelerated, mixed-precision, cross-libm bit,
public-model, model-quality, whole-model atomicity, flat-RSS training, JIT, optimizer,
checkpoint, or resume claim.
