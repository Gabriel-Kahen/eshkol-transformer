# G3-C4-N exact forward provider decision

**Proposed for independent numerical implementation; no runtime evidence.**
Source baseline `ba0e37d06076d0a16c473ff742ab95723cb2cb89`. This standalone
prerequisite uses merged K1/I1/I2/G3-N/N2/N3K/A2 APIs and does not depend on
C4 model construction, shared C4 pins, G3 transport, persistence, P1 mode changes
or the restricted sampler review. Root may accept/dispatch it independently.

## Exact identity and metadata

Header `include/eshkol_transformer/g3c4_primitives_abi.h` includes only the
unchanged K1 ABI header, supports C11/C++17 and `extern "C"`, defines
`ET_G3C4_PRIMITIVES_ABI_MAJOR 1u`, `ET_G3C4_PRIMITIVES_ABI_MINOR 0u`, and declares:

```c
const et_kernel_provider_v1 *et_g3c4_kernel_provider_v1(void);
```

Return one immutable process-lifetime K1 ABI1.0 descriptor, required_features0,
provider name and every capability implementation `g3c4.cpu-f32.serial`,
provider/capability version `1.0`, evidence `G3-C4:cpu-f32-c4-forward-v1`.
Verified metadata is published only by the accepted implemented provider.
Every capability has deterministic1, dtype list exactly f32, device list cpu.
Deterministic0 requests use the same implementation; other flag encodings retain
K1 admission. Provider metadata reserved bytes are zero; do not add rejection
of caller reserved fields or interpret unknown tails beyond the K1 contract.

Exactly seven capabilities, eight distinct operation names, 28 operation/row
pairs. Capability and operation tables are lexicographically ordered; row order
is exactly as expanded below. All extents min=max; unbounded/reserved flags0.

| Capability | Operation(s) | Ordered exact rows |
|---|---|---|
| `g3c4.causal-attention` | `g3c4.causal-attention.forward` | [1,2,2,1,4,2], [1,2,2,2,4,2], [1,2,2,3,4,2], [1,2,2,4,4,2], [1,2,2,3,3,2] |
| `g3c4.embedding-forward` | `g3c4.embedding.forward` | [1,1,4,4], [1,2,4,4], [1,3,4,4], [1,4,4,4], [1,3,256,4], [1,4,256,4] |
| `g3c4.gelu` | `g3c4.gelu.forward` | [1,3,8], [1,4,8] |
| `g3c4.head-layout` | `g3c4.heads.merge.forward`, `g3c4.heads.split.forward` | [1,3,2,2], [1,4,2,2], for both operations |
| `g3c4.layer-norm` | `g3c4.layer-norm.forward` | [1,3,4], [1,4,4] |
| `g3c4.linear` | `g3c4.linear.forward-no-bias` | [1,3,4,4], [1,3,4,8], [1,3,8,4], [1,3,4,256], [1,4,4,4], [1,4,4,8], [1,4,8,4], [1,4,4,256] |
| `g3c4.residual` | `g3c4.residual.forward` | [1,4,4] |

Counts:6 embedding +8 linear +2 norm +2 GELU +1 residual +4 layout +5 attention.
T4 cached/full-prefix attention shares one row. T3 residual [1,3,4] already
belongs to N2; no duplicate is added. N3K already has initializer [4,4]. Existing
T1/T2 routes remain with G3-N/N3K/N2/A2 except C4 position embeddings and full-
capacity attention above. No backward, sampler, initializer, RoPE, cache owner,
activation-sum or generic provider resolver belongs to this ABI.

## Ordered operands and layout

Use K1 request/call/view layouts and native descriptor arities below. Views are
naturally aligned, dense row-major, offset0, on cpu, exact byte lengths (f32×4,
i64×8,bool×1). Request compute dtype is f32 even for integer/bool operands.
No optional operand or auxiliary scalar field exists. Admit at least the v1.0
prefixes: call88, request56, view72, capability120, provider96 bytes; tensor table
stride>=72, each72<=struct_size<=stride and checked count×stride coverage.
Future tails are accepted without interpretation as in G3-N/K1.

| Operation / request | Ordered input descriptors | Sole output |
|---|---|---|
| embedding [N,T,V,D],2→1 | IDs i64[N,T], weight f32[V,D] | f32[N,T,D] |
| linear [N,T,Din,Dout],2→1 | x f32[N,T,Din], weight f32[Dout,Din] | f32[N,T,Dout] |
| norm [N,T,D],4→1 | x f32[N,T,D], gamma f32[D], beta f32[D], epsilon f32[] | f32[N,T,D] |
| GELU [N,T,D],1→1 | x f32[N,T,D] | f32[N,T,D] |
| residual [N,T,D],2→1 | skip f32[N,T,D], branch f32[N,T,D] | f32[N,T,D] |
| split [N,T,H,Dh],1→1 | f32[N,T,H*Dh] | f32[N,H,T,Dh] |
| merge [N,T,H,Dh],1→1 | f32[N,H,T,Dh] | f32[N,T,H*Dh] |
| attention [N,Hq,Hkv,Tq,Tk,Dh],6→1 | Q f32[N,Hq,Tq,Dh], K f32[N,Hkv,Tk,Dh], V same as K, query positions i64[N,Tq], key positions i64[N,Tk], keep bool[N,Tq,Tk] | f32[N,Hq,Tq,Dh] |

Every floating input is finite, including unselected weight rows and masked K/V.
Embedding IDs satisfy0<=id<V before access: V4 for position rows, V256 for token
rows. Epsilon is any finite positive f32; both zeros reject. Positions are i64 in
[0,16777215], strictly increasing within each query and key sequence; no query/
key relationship beyond the causal predicate is imposed. Mask bytes are exactly0/1.
The C4 consumer separately fixes epsilon bits0x3727c5ac and positions0..3. This
primitive's position domain is not a larger-context model claim.

Inputs are pairwise disjoint except residual skip/branch may be identical complete
semantic views and attention query/key positions may be identical complete views
when Tq=Tk. Complete equality means same data span/dtype/device/layout/offset/rank/
extents, not descriptor address. No partial overlaps; Q/K/V remain disjoint.
Output data is disjoint from every input and other output under complete K1
generic validation; caller-owned descriptor metadata must remain stable. GELU is not in-place. No view survives synchronous invoke.

## Exact forward arithmetic

All operations use binary32 statement rounding, increasing index order, no FMA,
fast-math, precision/device cast, clamping or scalar Eshkol fallback. Inputs and
all computed intermediates/results must be finite, except expf underflow to zero
is admitted. A nonwriting dry run and invoke use the same evaluation routine.

- Embedding copies selected row bits. Layout maps token offset
  `((n*T+t)*H+h)*Dh+d` to head offset `((n*H+h)*T+t)*Dh+d`; merge reverses the
  mapping into distinct storage. T3/T4 require materialization, not relabeling.
- Linear loops n,t,o,i, sum starts+0; compute product then addition as separate
  f32 statements with finiteness checks. Weight layout is[Dout,Din]. Residual
  computes exactly skip+branch once per element.
- Norm uses increasing-d sum/D, increasing-d squared deviations/D (biased
  variance), variance+epsilon, sqrtf, reciprocal, (x-mean)*rstd, then
  normalized*gamma followed by +beta. Check every intermediate separately.
- GELU copies N2 `gelu_one`: half=0x1p-1f, one=0x1p+0f,
  inv_sqrt_2=0x1.6a09e6p-1f; scaled=x*inv_sqrt_2, erf_value=erff(scaled),
  sum=one+erf_value, half_x=half*x, y=half_x*sum. Every named value is checked.
  No tanh approximation or backward-only x*x operation.
- Attention uses hkv=hq/(Hq/Hkv); keys participate iff mask1 and key_position<=
  query_position. Dot products scan increasing d from+0, checking each product
  and partial sum. scale=1/sqrtf((float)Dh), score=dot*scale (not dot/root).
  Scan admitted keys in increasing order for maximum, then compute
  shifted=score-maximum, weight=expf(shifted), denominator as increasing-key sum.
  Recompute score/shifted and probability=expf(shifted)/denominator for each
  output d, then increasing-key sum of separately checked probability*V.
  No participating key returns exact+0 for each output with no score evaluation;
  declared masked values still undergo finite-input checks.

**Explicit multikey validation refinement:** reject a nonfinite `shifted` before
expf in both the denominator and probability passes. Two finite extreme scores
can subtract to negative infinity; the existing G3-N one-key row cannot expose
this case, while inherited A2/G3-N helper source does not explicitly reject it.
This new C4 provider rejects such subtraction overflow as
INVALID_ARGUMENT/PROVIDER_REJECTED, preserving output. Finite shifted values may
underflow through expf to0. This is a named adaptation, not a claim of unchanged
helper bytes or an amendment of existing A2/G3-N rows. For admitted finite
arithmetic, all existing statement order remains unchanged.

There are no gradients, RNG, graphs, parameters, model mutation or cache leases.
Forward-only capabilities make no compiler AD or VJP claim. Primitive acceptance
cannot establish the eventual C4 no-grad model schedule or generation behavior.

## Errors, floating environment and call lifetime

Normatively inherit the complete descriptor admission/error table, prefix/table/
alignment/alias/value precedence, future-tail rules and x87/MXCSR contract from
[accepted G3-N ABI](G3_N_ABI_PROPOSAL.md#aliases-errors-and-validateinvoke-lifetime),
with attempted operation renamed to `g3c4.*` and fallback `g3c4.provider`.
All new GELU rows use the same envelope, not N2's different public name mapper.
This incorporation is fixed to the source baseline above; future predecessor
changes do not automatically amend C4-N. In particular:

- Absent row/operation/request dtype/device: K1 UNSUPPORTED/CAPABILITY_NOT_VERIFIED;
  direct provider UNSUPPORTED/PROVIDER_REJECTED.
- Malformed prefix/stride: VERSION_MISMATCH/INVALID_STRUCT_SIZE; operand shape:
  SHAPE_MISMATCH/INVALID_SHAPE; dtype: DTYPE_MISMATCH/INVALID_TEXT; layout/offset:
  NONCONTIGUOUS/INVALID_BUFFER. Other buffer/null/overflow/device distinctions
  remain exactly the incorporated table.
- Invalid ID/position/mask/epsilon, forbidden input alias, nonfinite input or
  intermediate (including shifted overflow): INVALID_ARGUMENT/PROVIDER_REJECTED.
- Unsupported FP controls/capture failure: UNSUPPORTED/PROVIDER_REJECTED;
  captured-fenv restoration failure: INTERNAL/PROVIDER_REJECTED.

Require x86-64 IEEE binary32, eight-bit bytes and FLT_EVAL_METHOD0; round-to-nearest-
even independently in x87/MXCSR, masked exceptions, FTZ/DAZ off. Check FP controls
before any floating reads, including signaling NaNs. Validation saves/restores
complete fenv, including independent exception flags, on success and failure;
never repair unsupported controls. Invoke may accrue arithmetic flags.

Caller must complete K1 generic and provider validation, then keep descriptors,
storage and FP controls stable through invoke. Direct provider validation alone
does not replace generic output-alias/descriptor checks; direct invoke without the
complete precondition is unsupported. Validation changes no tensor bytes; invoke
fully overwrites its output without allocation or recoverable branch. Concurrent
mutation, retained views and reentrant mutable call buffers are unsupported.

Bitwise evidence is scoped to the reviewed supported Ubuntu22/Clang-LLVM21.1.8/
libm lane and the accepted pinned Eshkol build. The pending upstream repin must
be accepted before final compiler/manifest evidence freezes; compatibility hosts
remain separately labelled. There is no cross-libm/CPU-lane bitwise assertion.

## Immutable source provenance and package

Copy reviewed helpers into one new C source; never modify frozen predecessor
sources or call their private accessors. Baseline source SHA-256 values:

| Source | SHA-256 | Intended reuse |
|---|---|---|
| `native/g3n_primitives_provider.c` | `cdc7a0416ea8f519c135aeb79b4c74f4660b2c116684dc24ffb7a975c4b7d195` | checked envelope, embedding/linear/norm/attention, residual/layout branches |
| `native/n2_primitives_provider.c` | `81a213545f2e9256eedf57455b0ea33f56b05d908f1165813da9a2f5077137c1` | `gelu_one` |
| `native/n3k_primitives_provider.c` | `3461963341a15f50cff53889d1b42ce95dd83d2a13ebaa6dfdcf333381567bed` | underlying embedding/linear/layout arithmetic lineage |
| `native/a2_attention_provider.c` | `fdb4ca172ca7aa282e787ff42b2998592c8b77a30f67d35680f7e38b2f550b02` | underlying multikey attention lineage |
| `include/eshkol_transformer/kernel_abi.h` | `09fb4338e9f9bfaaf157d7d3f948914b43c0e5fd44cf289242283b0ed29e70e0` | unchanged K1 layouts |

Review helper-body correspondence mechanically. Explicit adaptations are exact
C4 metadata/names/operand schemas; T/Tq/Tk from admitted requests instead of
hardcoded T1 norm/attention shapes; residual element count16 instead of4;
validation of every embedding ID before any gather, every mask byte and each
strictly increasing query/key-position sequence; GELU dispatch; and the two
shifted-finiteness checks. Preserve the G3-N weighted-
product finiteness check and empty-attention+0 behavior. Do not claim a copied
helper is unchanged when any arithmetic, loop order or validation changes.

Production project-input closure is exactly new
`native/g3c4_primitives_provider.c`, new
`include/eshkol_transformer/g3c4_primitives_abi.h` and unchanged K1 header.
Archive `build/g3c4/libeshkol_transformer_g3c4.a`, sole member
`g3c4_primitives_provider.o`, sole defined global `et_g3c4_kernel_provider_v1`.
No Eshkol facade, aggregate registry, allocator, resolver or private test export.

Use verified-toolchain C11/O2/PIC, strict warnings/stack protection,
`-ffp-contract=off -fexcess-precision=standard -fno-fast-math`, deterministic
archive creation and atomic replacement. Allowed undefined ceiling:
`__stack_chk_fail`, `et_kernel_error_clear`, `erff`, `expf`, `fegetenv`, `fesetenv`,
`memcpy`, `memset`, `snprintf`, `sqrtf`, `strcmp`. Freeze the actual supported
compiler subset after implementation; it is not measured here. Sanitizer objects
have separate nonshipping inventories. No malloc/free, loader, RNG, predecessor
accessor, binary64 numerical helper or Python runtime dependency.

## Independent acceptance gate and root disposition

After contract acceptance, implement only this standalone provider and its focused
build/tests. No G3-S inspection/review reproduction, C4 owner, pins, transport,
full-model loop or persistence work is part of that dispatch.

Required proof: all28 pairs through actual K1 require/dispatch; independent
mathematical/literal f32 references, non-self-inverse marked layouts, all mask/
position arrangements and empty rows, extreme finite opposite-sign scores that
force shifted overflow, expf underflow, weighted-product/reduction overflow,
GELU signs/subnormals/zeros, norm epsilons/overflow and embedding boundaries.
Assert exact metadata and rejected neighbours for every row; mutated arithmetic/
layout/mask/domain checks must fail the intended oracle.

Cover every descriptor/error/alias/fenv path, both complete input-alias exceptions
and partial-alias rejection, unchanged inputs/outputs on failure, identical dry/
invoke arithmetic, genuine I1/I2 carriers and paired scoped borrows, allocation-
disabled invoke, sanitizer/leak gates, immutable pointer/metadata and no retention.
Two fresh private Eshkol AOT callers and deterministic builds must prove source/
member/symbol/header/public-link negatives with no runtime Python. Public model
transport is not inferred from this development bridge. Supported exact-head CI
and proportional merged proof are later implementation acceptance requirements;
this proposal runs neither.

Root decision requested: exact names/rows/schema, inherited error/fenv/alias
contract, multikey shifted-overflow rejection, copied-source provenance and
one-source/one-export package boundary. Public C4 model/continuation, C4 pins,
G3-R policy/wire/replay and shared storage#114 remain separate contracts.
