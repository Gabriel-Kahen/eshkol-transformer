# G3-N exact forward primitive ABI proposal

Status: **ABI proposal only; no headers, implementation or new capability evidence**.
This specializes the settled C2 design in [binding decision 5748532295](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5748532295).
It preserves the accepted profile, no-grad boundary, prerequisite ordering and
unchanged predecessor APIs/formats. Source inspection only was performed for this
document; no builds/tests or supported-platform results are claimed.

## Identity, names and exact rows

Propose header `include/eshkol_transformer/g3n_primitives_abi.h`, C11/C++17 compatible,
including only `eshkol_transformer/kernel_abi.h`, with C++ `extern "C"` guards,
`ET_G3N_PRIMITIVES_ABI_MAJOR 1u`, `ET_G3N_PRIMITIVES_ABI_MINOR 0u`, and sole accessor:

```c
const et_kernel_provider_v1 *et_g3n_kernel_provider_v1(void);
```

It returns one immutable process-lifetime descriptor, with K1 ABI 1.0 and zero
required features. Provider name and every capability's implementation string are
`g3n.cpu-f32.serial`; provider/capability version is `1.0`; evidence is
`G3-N:cpu-f32-c2-forward-v1`. All capabilities are VERIFIED only in the implemented,
reviewed provider, deterministic=1, dtype list exactly `f32`, device list exactly
`cpu`. A request's deterministic=0 does not require determinism and remains admitted;
execution remains the same deterministic implementation. Generic malformed flags
retain K1 admission rules. Provider metadata emits zero reserved bytes; unchanged
K1 does not inspect caller reserved bytes, and this provider adds no reserved-byte
rejection or future-tail interpretation.

Six capabilities, seven operation names, **eleven operation/row pairs**:

| Capability | Sole operation, except where listed | Exact request shapes |
|---|---|---|
| `g3n.embedding-forward` | `g3n.embedding.forward` | [1,1,256,4], [1,1,2,4] |
| `g3n.linear-forward` | `g3n.linear.forward-no-bias` | [1,1,4,4], [1,1,4,8], [1,1,8,4], [1,1,4,256] |
| `g3n.layer-norm-forward` | `g3n.layer-norm.forward` | [1,1,4] |
| `g3n.residual-forward` | `g3n.residual.forward` | [1,1,4] |
| `g3n.head-layout-forward` | `g3n.heads.merge.forward`, `g3n.heads.split.forward` | [1,1,2,2] for both |
| `g3n.causal-attention-forward` | `g3n.causal-attention.forward` | [1,2,2,1,1,2] |

Every extent is min=max; maximum_unbounded and reserved bytes are zero. Capability
and operation tables use lexicographic byte order; rows use the displayed order.
Metadata is the single exact-row authority, including direct provider validation.
No other operation/row/dtype/device is admitted. In particular no backward, GELU,
initializer, sampling, RoPE, normalization variant, cache operation or T2 extension
belongs to G3-N. N2 already provides `gelu.forward` [1,1,8]; G3-T must explicitly
route to that provider. Existing N3K/N2/A2 capabilities and artifacts remain intact.

## Ordered descriptors and parameter admission

Use unchanged K1 request/call/tensor layouts. All views are dense row-major,
zero-offset CPU, naturally aligned, with byte_length exactly element count times
4 for f32, 8 for i64 or 1 for bool. No optional operands or outputs exist.
Request compute dtype stays f32 even when some inputs are i64/bool. Tensor counts
below are native descriptor arities, not new public Eshkol names.
Require at least the v1.0 prefixes (call 88, request 56, tensor view 72,
capability 120, provider 96 bytes); accept valid future tails without interpreting
them. Every tensor table must cover its declared count/stride span without
overflow, with stride>=72 and each view's 72<=struct_size<=stride. There is no
K1 auxiliary parameter field: epsilon/positions/mask are ordinary ordered inputs.

| Operation/request interpretation | Ordered inputs | Output (always exactly one) |
|---|---|---|
| embedding [N,T,V,D], 2→1 | IDs i64[N,T]; weight f32[V,D] | y f32[N,T,D] |
| linear [N,T,Din,Dout], 2→1 | x f32[N,T,Din]; weight f32[Dout,Din] | y f32[N,T,Dout] |
| LayerNorm [N,T,D], 4→1 | x f32[N,T,D]; gamma f32[D]; beta f32[D]; epsilon f32[] | y f32[N,T,D] |
| residual [N,T,D], 2→1 | skip f32[N,T,D]; branch f32[N,T,D] | y f32[N,T,D] |
| split [N,T,H,Dh], 1→1 | token-major x f32[N,T,H*Dh] = [1,1,4] | head-major y f32[N,H,T,Dh] = [1,2,1,2] |
| merge [N,T,H,Dh], 1→1 | head-major x f32[N,H,T,Dh] = [1,2,1,2] | token-major y f32[N,T,H*Dh] = [1,1,4] |
| attention [N,Hq,Hkv,Tq,Tk,Dh], 6→1 | Q f32[1,2,1,2]; K f32[1,2,1,2]; V f32[1,2,1,2]; query positions i64[1,1]; key positions i64[1,1]; keep bool[1,1,1] | y f32[1,2,1,2] |

Every floating input must be finite, including unselected embedding rows and masked
attention values. IDs satisfy 0<=id<V before row access. Epsilon is any positive
finite f32, matching N2's numeric parameter contract; its fixed model value
`0x3727c5ac` is a G3-T/M admission condition, not a G3-N scalar constant.
Attention positions are exact i64 in [0,16777215], matching A2; strict increase
within a one-element sequence is vacuous. Keep is exactly byte 0 or 1. G3-T/M
restrict actual C2 execution to absolute positions 0 and 1. Broader primitive
position-value tests do not advertise larger model context.

## Numerical and no-grad contract

Use reviewed native forward helpers copied with exact source/commit provenance,
not linked predecessor private accessors or modified frozen sources. Preserve
N3K's embedding/linear/residual/layout arithmetic and N2 LayerNorm/A2 attention
arithmetic; the new evidence is for the exact previously missing rows.

- Embedding and layout are bit-preserving copies. Layout maps
  `((n*T+t)*H+h)*Dh+d` to `((n*H+h)*T+t)*Dh+d`; merge reverses it. T=1 makes
  flat element order identical while ranks differ, so shape/direction admission
  remains essential. This row cannot prove nontrivial permutation behavior.
- Linear scans n,t,o then i, starts each sum at +0, and computes separate f32
  product and addition; weights are [Dout,Din]. Residual computes one f32
  `skip+branch` per element. Check every product and partial sum for finiteness.
- LayerNorm is N2's two-pass biased variance: increasing-d sum divided by D;
  increasing-d squared deviations divided by D; variance+epsilon; sqrtf;
  reciprocal; `(x-mean)*rstd`; then `(normalized*gamma)+beta`, as separate f32
  statements. Check every intermediate/result. Do not silently fix overflow.
- Attention preserves A2's increasing-d dot product, `sqrtf((float)Dh)`,
  reciprocal, then dot*scale. A key participates iff keep=1 and key_position is
  <=query_position. Use maximum-subtracted expf softmax and increasing-key f32
  weighted sum. With this single-key row an admitted key has probability one;
  nevertheless score evaluation and its overflow rejection remain mandatory.
  No admitted key produces exact +0 in all output elements, with no score
  evaluation; all declared input values are still checked finite.

There are no gradients, RNG draws, graph/parameter identities, model mutation or
cache leases. Forward kernels are used in a no-grad schedule; this ABI advertises
no analytic VJP or compiler autodiff support. Native loops are reviewed numerical
kernels, with no recursive Eshkol numerical path, cast/device fallback or Python
runtime. Primitive evidence alone does not prove the eventual G3-M schedule.

## Aliases, errors and validate/invoke lifetime

Inputs are read-only synchronous borrows. Output data is disjoint from every input
and every other output under unchanged K1 generic checks. Input ranges are pairwise
disjoint, with exactly two exceptions: residual skip/branch may be equal complete
semantic views, and attention query/key positions may be equal complete semantic
views. Equality means identical data range, dtype/device/layout/offset/rank/extents,
not descriptor-pointer identity. Every partial input overlap rejects, including
partial position overlap. Q/K/V remain mutually disjoint. No allocations, ownership
transfers, retained pointers or registries occur in accessor/validate/invoke.

The caller first performs complete K1 generic admission and provider validation,
then keeps metadata/storage/FP controls unchanged through invoke. The provider
checks call/request prefixes before later fields, ordered table counts/strides/
spans before operands, alignment and exact operand schemas before numerical reads,
then permitted aliases/data values and a non-writing dry evaluation identical to
invoke. Direct provider validation checks exact advertised rows but is not a
replacement for K1 generic output-alias/descriptor checks. Direct invoke
without this complete successful precondition is outside the contract.

Every recoverable validation failure preserves all tensor bytes and input metadata.
Invoke fully overwrites output using a nonfailing allocation-free calculation;
it returns void and publishes no owner or logical generator state. Output bytes
may be observed only after synchronous invoke returns. Concurrent mutation,
reentrancy across mutable call buffers and changes between validate/invoke are
unsupported. There is no finalize, release or shutdown operation in G3-N.

Use frozen `et_kernel_error`, never E1 objects or new error enums. K1 errors keep
their existing precedence/categories/codes; provider-specific mappings are:

| Failure | Category/code |
|---|---|
| Absent operation/row/request dtype or well-formed other request device | K1 UNSUPPORTED/CAPABILITY_NOT_VERIFIED; direct provider UNSUPPORTED/PROVIDER_REJECTED |
| Truncated call/request/view prefix or tensor stride | VERSION_MISMATCH/INVALID_STRUCT_SIZE |
| Admitted operand rank/extent mismatch | SHAPE_MISMATCH/INVALID_SHAPE |
| Operand dtype mismatch | DTYPE_MISMATCH/INVALID_TEXT |
| Operand non-CPU/mismatched device | K1 DEVICE_MISMATCH/INVALID_BUFFER; direct provider DEVICE_MISMATCH/INVALID_TEXT |
| Non-dense layout or nonzero offset | NONCONTIGUOUS/INVALID_BUFFER |
| Wrong count/table span/alignment or invalid data alignment, excluding data-address wrap | INVALID_ARGUMENT/INVALID_BUFFER |
| Generic shape-product overflow | K1 SHAPE_MISMATCH/INTEGER_OVERFLOW |
| Exact view-byte-length mismatch | K1 SHAPE_MISMATCH/INVALID_BUFFER; direct provider INVALID_ARGUMENT/INVALID_BUFFER |
| Generic output overlap/address wrap | K1 INVALID_ARGUMENT/ALIASING_OUTPUT |
| Forbidden input overlap or direct-provider data-address wrap | INVALID_ARGUMENT/PROVIDER_REJECTED |
| Invalid ID/position/bool/epsilon, nonfinite input/intermediate/result | INVALID_ARGUMENT/PROVIDER_REJECTED |
| Unsupported FP controls or failure to capture fenv | UNSUPPORTED/PROVIDER_REJECTED |
| Failure to restore captured fenv | INTERNAL/PROVIDER_REJECTED |

NULL required arguments and generic tensor-table arithmetic overflow retain K1
INVALID_ARGUMENT/NULL_ARGUMENT and INVALID_ARGUMENT/INTEGER_OVERFLOW. Error
operation identifies the attempted `g3n.*` operation, or `g3n.provider` before a
valid operation is available; message text is explanatory, not an ABI discriminator.

Require x86-64 IEEE binary32, eight-bit bytes and FLT_EVAL_METHOD=0. Check x87
and MXCSR round-to-nearest-even, masked FP exceptions, FTZ/DAZ off before reading
floating inputs, including signaling NaNs. Save/restore complete incoming fenv
(independent x87/MXCSR flags included) around all validation, success or failure.
Do not repair unsupported controls. Invoke may accrue ordinary arithmetic flags.
Compile without contraction, fast-math or excess precision. Bit determinism is
scoped to the reviewed supported Ubuntu22/Clang-LLVM21/libm lane; compatibility
hosts are separately qualified and no cross-libm guarantee follows.

## Source/package inventory and acceptance

Proposed production project-input closure is exactly:
`native/g3n_primitives_provider.c`, `include/eshkol_transformer/g3n_primitives_abi.h`,
and unchanged `include/eshkol_transformer/kernel_abi.h`. Numerical helpers are
static inside the one C source. Proposed archive is
`build/g3n/libeshkol_transformer_g3n.a`, containing only
`g3n_primitives_provider.o`, with exactly one defined global symbol,
`et_g3n_kernel_provider_v1`. No installed Eshkol facade, canonical K1 resolver,
private test export, carrier implementation or owning aggregate is introduced.

Use the existing verified-toolchain build pattern: C11, -O2, PIC,
`-ffp-contract=off -fexcess-precision=standard -fno-fast-math`, strict warnings and
stack protection; deterministic archive creation and atomic artifact replacement.
The allowed undefined-symbol ceiling is exactly `__stack_chk_fail`,
`et_kernel_error_clear`, `expf`, `fegetenv`, `fesetenv`, `memcpy`, `memset`,
`snprintf`, `sqrtf`, `strcmp`. Implementation must freeze and independently review
the actual exact subset on the supported compiler; this list is not a measured
object inventory. No allocator, loader, RNG, predecessor provider accessor,
binary64 math or test dependency is allowed. ASan/UBSan objects have separate
instrumented inventories; instrumentation is never shipped as the canonical object.

Required gates, executed only after the ABI is accepted and implemented:

1. All eleven operation/row pairs through K1: independent mathematical forward
   references, literal order-sensitive f32 vectors, every row's ±1/zero/wrong-rank
   neighbors and old-provider unchanged-capability baselines. Single-key attention
   must test both mask values and future/equal/past positions, maximum exact
   position, all Q/K/V heads, and QK overflow despite probability-one output.
2. Embedding boundaries/unselected NaN; weight-layout and contraction mutations;
   LayerNorm nonconstant/constant/subnormal/large values, epsilon ±0/negative/NaN/
   smallest-positive, overflow at each arithmetic stage; signed-zero copy/add
   cases and rank-sensitive split/merge operands. No gradient claim: adding a
   backward operation would require a separate ABI decision and central differences.
3. Every operation's malformed prefixes/counts/strides/shapes/bytes/dtypes/devices/
   alignment/spans; output overlaps; every forbidden input pair plus the two exact
   alias exceptions; finite/intermediate failures; independent x87/MXCSR flags and
   bad controls. Assert all output/input bytes and fenv unchanged on rejection.
4. Raw-view versus I1/I2 scoped-owned dispatch parity and active-borrow exclusions;
   ASan/UBSan/LSan native tests, allocation-fail instrumentation proving no provider
   allocation, immutable provider pointer/metadata and no retained caller storage.
5. Exact source/depfile/member/defined/undefined inventories, predecessor hashes,
   C11/C++17 header clients, private/canonical-symbol link negatives, no runtime
   Python dependency, two fresh deterministic objects/archives, and two fresh
   pinned-Eshkol private AOT callers exercising all rows and negatives. AOT is a
   private development bridge, not G3-T/G3-M public transport evidence. Supported
   exact-head CI and proportional postmerge gates remain implementation obligations.

Remaining root ABI dispositions are narrowly: ratify the proposed names/metadata,
the eleven forward-only pairs and exact schemas, inherited numeric epsilon/position
admission with tighter G3-T/M policy, the two input-alias exceptions, normalized
provider-specific errors/fenv contract, and the one-source/one-export inventory
with supported-build exact undefined-subset freeze. No settled high-level G3
profile, sampler, lifecycle or persistence choice is reopened here.
