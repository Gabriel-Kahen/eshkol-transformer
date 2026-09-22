# L3S masked objective: prefreeze proposal

Status: contract accepted for bounded implementation; implementation remains active.
[Binding decision](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5747250069),
[issue mirror](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/87#issuecomment-5747250113).
The original prefreeze proposal below is retained for provenance. Proposal base: `21b1df13940e94adce998ab88cb45c329c956b58`.

## Boundary and discovery

Separate ABI 1.0, header `include/eshkol_transformer/l3s_masked_objective_abi.h`,
source/member `native/l3s_masked_objective_provider.c` /
`l3s_masked_objective_provider.o`, archive `build/l3s/libeshkol_transformer_l3s.a`.
Sole export: `const et_kernel_provider_v1 *et_l3s_kernel_provider_v1(void)`.
Immutable provider/implementation `l3s.cpu-f32.serial`, version `1.0`, evidence
`L3S:cpu-f32-masked-objective-v1`. One verified K1 capability
`l3s.masked-objective`, compute dtype `f32`, device `cpu`, exact request `[1,2]`,
deterministic true (requests with either deterministic boolean are eligible).

Explicit resolver only. No canonical resolver, K1 baseline change, K2 integration,
L2 ABI change, bool/f32 owner, installed scalar, Eshkol module, graph, aggregate,
model schedule, M3 dependency, trainer, checkpoint, accelerator or fallback.

## Exact operations and views

Every name below starts with `l3s.masked-objective.`:

| Suffix | Ordered borrowed inputs | Ordered caller-owned outputs |
|---|---|---|
| `reduce.bool` | CE `f32[1,2]`, mask `bool[1,2]` | numerator `f32[]`, W `f32[]`, mean `f32[]` |
| `reduce.f32` | CE `f32[1,2]`, mask `f32[1,2]` | numerator `f32[]`, W `f32[]`, mean `f32[]` |
| `numerator-seed.bool` | mask `bool[1,2]` | upstream `f32[1,2]` |
| `numerator-seed.f32` | mask `f32[1,2]` | upstream `f32[1,2]` |
| `mean-seed.bool` | mask `bool[1,2]` | upstream `f32[1,2]` |
| `mean-seed.f32` | mask `f32[1,2]` | upstream `f32[1,2]` |

Dense row-major CPU, zero offset, natural alignment, exact spans: f32 pair eight
bytes, bool pair two bytes, scalar four bytes and null rank-zero shape pointer.
No broadcasting or additional shapes. Masks have no gradient. Capability reports
advertise f32 compute only, never tensor storage or a generic CE implementation.

All data inputs pairwise disjoint; outputs mutually disjoint and disjoint from all
inputs and the call/request/view tables, shape arrays and referenced metadata.
Adjacent nonoverlapping spans are valid. Borrowed metadata/data and FP controls
remain stable through validation/invocation. No pointers are retained, dispatch
allocates nothing, success fully overwrites each output. Independent I2 scalar
borrows can supply the outputs; this is no new scalar owner.

K1 error storage is disjoint caller-owned control memory by its existing trusted
caller precondition. K1 writes it before provider validation, so error/data alias
cannot be advertised as a recoverable L3S rejection. Native readable-pointer
preconditions remain unchanged.

## Numerical domain and order

Bool storage is read as bytes and admits only 0/1. F32 masks and CE are finite,
nonnegative; signed zero is accepted and locally canonicalized to positive zero.
Both CE entries are validated even when masked off. Positive subnormals are valid.
All six operations require positive finite W, including numerator seeds: an
all-zero mask rejects consistently as outside the valid masked-objective domain.

Every operation is separate IEEE binary32 RN-even arithmetic, no FMA, reciprocal
substitution, reassociation, higher-precision path or hidden loss scaling:

```text
w = +0
w = f32(w + m[0])
w = f32(w + m[1])
p0 = f32(m[0] * ce[0])
p1 = f32(m[1] * ce[1])
n = +0
n = f32(n + p0)
n = f32(n + p1)
mean = f32(n / w)
numerator_seed[i] = m[i]
mean_seed[i] = f32(m[i] / w)  # direct division for each index 0 then 1
```

Seed operations compute W from their mask and do not take an unauthenticated
denominator. Reduction checks every intermediate and all three final values;
seed operations check W and both output values. Reject nonfinite values even if
a later real-number quotient would be finite. Permit gradual underflow, subnormal
results and correctly rounded positive computations becoming +0. No rejection
based solely on sticky underflow/inexact flags. Canonicalize zero outputs to +0.
All-zero CE with positive W succeeds; seeds retain the mask/normalization semantics.

The numerator seed is passed to L2 backward for unnormalized weighted gradients;
the mean seed is passed for the mathematical mean-loss VJP. A later accumulator
using numerator/W must not receive already mean-normalized gradients and divide
again. These are mathematical derivatives evaluated in f32, not derivatives of
the discontinuous rounded machine program.

Environment: x86-64, eight-bit bytes, IEEE binary32, `FLT_EVAL_METHOD=0`, no
fast-math/contraction/excess precision. Require x87 and MXCSR nearest-even, all
exception masks set, FTZ/DAZ off. Reject unsupported control state without repair.
Save/restore complete validation fenv including independent sticky flags on both
success and failure, before any floating classification. Successful invoke may
accrue normal arithmetic flags. L2 composition tests establish this environment
externally; L3S does not retroactively change L2's environment contract.

## Errors and atomicity

K1 generic validation/capability matching retains precedence and categories.
Provider validates schemas/counts/spans/alignment and metadata exclusions, then
environment, all operand values, W, and every prospective result in fixed stack
storage. Only full success reaches the allocation-free nonfailing commit. Any
recoverable failure preserves all tensor, descriptor and guard bytes; the separate
error record reports the failure.

Request/operation/dtype/device/profile mismatch: K1 UNSUPPORTED /
CAPABILITY_NOT_VERIFIED. Operand shape mismatch: SHAPE_MISMATCH / INVALID_SHAPE;
dtype: DTYPE_MISMATCH / INVALID_TEXT; layout/offset: NONCONTIGUOUS /
INVALID_BUFFER. K1 handles truncated prefixes, table arithmetic, byte-length and
output-alias errors unchanged. Provider alignment/range errors: INVALID_ARGUMENT /
INVALID_BUFFER; input or metadata alias: INVALID_ARGUMENT / ALIASING_OUTPUT;
bad bool/negative/nonfinite/zero-W/unrepresentable intermediate: INVALID_ARGUMENT /
PROVIDER_REJECTED. Unsupported FP controls: UNSUPPORTED / PROVIDER_REJECTED.
Direct callbacks require the existing complete K1 validation preconditions.

## Independent dispositions and gates

Three independent Astra HIGH reviews: numerical/reference conditionally approves
the stated arithmetic; API/packaging approves the isolated boundary; adversarial
review approves proceeding with explicit trusted-error-memory and fenv boundaries.
All recommend the common positive-W domain and no predecessor changes.

Issue #87's reversed-accumulation mutation needs a bounded-profile clarification:
two nonnegative finite terms with +0 initialization have the same RN-even sum in
either order. No honest output-sensitive reversal test exists here. Proposed
acceptance is fixed traversal source/order checks plus observable reversed pairing
and seed-order mutations, documenting the equivalence instead of widening shape.
Independent exact-rational probes already pass 10 tests, including direct division
versus reciprocal and separate product/add versus FMA witnesses. They are
non-freezing development evidence only.

After acceptance: frozen pinned-PyTorch and independent analytic/FD composition
with real L2 `[1,2,256]`, bool/f32/weighted/zero/subnormal/extreme cases; all schema,
environment, alias and late-failure byte checks; genuine I1/I2 scoped borrowed
views; two private canonical pinned-Eshkol AOT builds/runs; strict C/C++, native
ASan/UBSan/LSan, exact source/member/export/undefined manifests, provider-free
baseline, deterministic artifacts and production language isolation. No Python
runtime. Affected A0/K1/L2/I1/I2/Q0 gates remain required.

CI proposal: one L3S producer in canonical build and native-numerics /
acceptance-predecessors plans; one test-l3s command in full/local/core paths;
L3S_ASAN_DETECT_LEAKS=1 with topology/plan tests. Reuse current K1/L2/I1/I2
prerequisites, 17 suites unchanged, top-level command count 24 to 25, no duplicate
job/build or M3-owned edits. Supported exact-head full CI, separate review, root
integration and postmerge checks remain mandatory. ROADMAP stays planned until
contract acceptance.
