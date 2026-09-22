# L3S bounded masked objective

Status: active implementation of the [accepted ABI 1.0 contract](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5747250069).
Supported exact-head CI, independent integration review, merge, and postmerge checks
remain pending. This is carrier-neutral numerical evidence, not public model loss
or training integration.

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
Metadata exclusions cover declared call/request sizes, full table byte spans
(including compatible-minor extensions), exact shape-array spans, and each
referenced string through its terminating NUL.
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

## Verification and limitations

Run `make test-l3s` with the canonical pinned Eshkol toolchain and `Q0_PYTHON`
pointing to Python 3.14.6 / PyTorch 2.13.0+cpu. The focused gate builds only
K1/L2/I1/I2/L3S prerequisites; full CI runs it once in the existing native-numerics
suite and reuses those prerequisite builds. The strict union with accepted M3 is
18 suites and 26 top-level commands.

Local CachyOS/LLVM 22 compatibility evidence includes 9,849 native assertions,
all six schemas, independent analytic and finite-difference real-L2 composition,
frozen pinned-PyTorch parity, exact-rational binary32 witnesses, and 11 compiled
mutants. Maximum analytic-gradient absolute error is `3.74637521e-08`; native f32
finite-difference maxima are `5.99622726e-05` (numerator) and `1.19954348e-05` (mean).
The development fixture SHA-256 is
`3cec8eec7574bcc78a0ccc369b631d45e5172e70c06122649201557b794009fa`; two regenerations
match exactly. Fixture and Python execution are development-only.

Two nonnegative summands with a +0 start have an observationally identical RN-even
sum under pure addition reversal. The [accepted traversal disposition](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5747145256)
requires executable equivalence controls and structural traversal/rounding checks,
not a fabricated numerical mutation kill. Observable compiled pairing, seed-position,
missing/double/count normalization, numerator/mean seed swap, reciprocal and FMA
mutants fail independent numerical assertions. Optimized IR/object inspection
excludes binary64 arithmetic and FMA, and source anchors preserve explicit rounding
points. See `tests/l3s/reference_design.md` for independent reference methodology.

Genuine I1 targets and I2 logits/loss/mask/seed/independent rank-zero scalar views
are borrowed synchronously and released. Separate L2 and L3S runtimes are explicitly
resolved; no global provider composition is added. Bool test storage is private
canonical byte storage, not a bool owner. Two fresh private pinned-Eshkol AOT
binaries and stdout match byte-for-byte; the bridge is absent from production and
a negative link proves its absence.

The archive has one member and one public accessor, checked exact source/export/
undefined manifests, unchanged predecessor source hashes and provider-free K1
baseline, warning-clean C11/C++17, and deterministic fresh object/archive bytes.
Native, numerical, carrier, and private bridge ASan/UBSan/LSan runs pass with
`detect_leaks=1` outside the traced local sandbox. The initial sandbox LSan run
failed because leak detection cannot run under ptrace; this was not a leak result.
Supported Ubuntu 22.04 / LLVM 21.1.8 evidence is still required.

There is no arbitrary shape, accelerator, mixed precision, ownership graph,
installed mask/scalar facade, public Eshkol model-loss API, compiler autodiff,
trainer, checkpoint, generation or performance claim. Future model/trainer
composition must preserve numerator-versus-mean seeds and the single normalization
point. Pure addition-order sensitivity needs more than two terms and is outside
this accepted profile.
