# G3 independent numerical and sampling review

Status: **proposed only; no numerical provider, public API, RNG format or G3
acceptance is established by this review**. Scope is issue #89's contract phase.
This source audit ran no builds and makes no new measured execution claim.

## Existing evidence and exact gaps

The recommended first diagnostic profile retains M3's CPU f32, N=1, V=256,
D=4, Hq=Hkv=2, Dh=2, L=1, F=8, learned-position table f32[2,4], bias-free
projections, exact-erf GELU, affine LayerNorms and tied head. Its context is exactly
C=2. Prompt length P is 1 or 2, and admitted requested maximum-new-token count
is an exact integer in [0,2-P], checked before replacing the cache. There is no
truncation, eviction, sliding window, implicit position extension or assumed early
EOS allowance. Positions are exactly 0 through P+G-1. P=1 permits at most one
emitted token; P=2 permits zero. This establishes only a small diagnostic slice,
not useful long generation or completion of G3.

Source witnesses are `native/n3k_primitives_provider.c`'s `dims_*`/`rows_*`,
`native/n2_primitives_provider.c`'s `DEFINE_ROW*` and capability arrays,
`native/a2_attention_provider.c`'s exact dimension arrays, and the fixed `roles[]`
table in `src/eshkol_transformer/m3t_transport.c`. K1 evidence is the actual
operation/row Cartesian product; a general C validator does not expand it.

| Forward role | Existing exact row relevant to M3 | Additional exact row needed for T=1 |
|---|---|---|
| Token embedding, [N,T,V,D] | N3K [1,2,256,4] | [1,1,256,4] |
| Learned-position embedding, [N,T,C,D] | N3K [1,2,2,4] | [1,1,2,4] |
| Q/K/V/attention-output linear, [N,T,Din,Dout] | N3K [1,2,4,4] | [1,1,4,4] |
| FFN up linear | N3K [1,2,4,8] | [1,1,4,8] |
| FFN down linear | N3K [1,2,8,4] | [1,1,8,4] |
| Tied head linear | N3K [1,2,4,256] | [1,1,4,256] |
| LayerNorm, [N,T,D] | N2 [1,2,4] | [1,1,4] |
| GELU, [N,T,F] | N3K [1,2,8] | None numerically: N2 already advertises `gelu.forward` [1,1,8]; explicit transport routing still needed |
| Residual add, [N,T,D] | N3K [1,2,4] | [1,1,4] |
| Split/merge head layout, [N,T,H,Dh] | N3K [1,2,2,2] | [1,1,2,2] |
| Full no-cache attention, [N,Hq,Hkv,Tq,Tk,Dh] | A2 [1,2,2,2,2,2] | [1,2,2,1,1,2] |
| C2 cached attention | A2 [1,2,2,1,2,2] and [1,2,2,2,2,2] | None numerically |
| Last-position logits extraction | M3 owns [1,2,256] logits, with no rank-two accessor | Explicit bit-preserving materialization [1,T,256] to owned [1,256], T=1 or 2 |
| Greedy/categorical sampler | None | Reviewed native no-grad operation over f32[1,256], exact i64 token result and immutable successor RNG |

N2's [1,1,1] and [1,1,3] norm rows do not establish [1,1,4]. Its unrelated
embedding and matmul rows do not fill the missing T1 operations. N2 residual
[1,1,1] likewise does not fill [1,1,4]. A2 [1,4,2,1,1,2] has a different query
head count; its MHA [1,2,2,1,1,1] has different depth. Neither establishes the
missing true T1 no-cache row. Padding a T1 input to T2 would introduce a new
composition and cannot be reported as T1 support.

The complete A2 attention row list currently is [1,2,2,1,1,1],
[1,2,2,2,2,1], [1,2,2,1,2,1], [1,2,2,1,2,2], [1,2,2,2,2,2],
[1,4,2,1,1,2], [1,4,2,3,3,2], [2,4,2,2,3,4], [2,4,2,3,3,2],
and [2,4,2,1,3,2]. A2 RoPE rows do not make M3 rotary: the recommended
profile remains learned-position, with unrotated K/V. Using A2 cache storage
for learned-position K/V therefore needs an explicit downstream interpretation
of A2's documented "post-RoPE K" stage seam: positional processing is complete
with identity rotation, not a new rotary-model claim.

M3T's `discover()` already routes N3K, N2 and A2 through separate K1 runtimes and
checks capability-name collisions. It is a fixed-role production seam, not a
caller-selectable general resolver. Its buffer dimensions and role requests remain
T2 constants. M3 `model-forward` always saves an independently owned VJP graph;
train/eval modes do not select a no-grad path. The generator cannot advertise
no-grad by calling that API and merely discarding its graph afterward.

## Proposed bounded dependency contracts

These are planning IDs, not new public operation names or accepted ABIs:

1. **G3-N**: establish the missing exact T1 forward rows and the true T1 no-cache
   attention row above. Prefer a separate provider with distinct capability names
   and reviewed numerical helpers; preserve frozen predecessor identities and
   capability ownership. Reusing N2's existing T1 GELU row is explicit dispatch,
   not fallback. If integration instead extends a predecessor, it must accept
   that provider's version/evidence/source-closure change first. Advertise only
   forward/no-grad operations needed here; a backward advertised on a new row
   requires its independent analytic/numerical gradient gate too.
2. **G3-T**: transport exact owned i64[1,1]/[1,2] prompts, i64[1,1] decode tokens,
   f32 activations at T1/T2, and owned f32[1,256] final logits; explicitly
   materialize `bool[1,Tq,2]` masks and `i64[1,Tq]`/`i64[1,2]` positions.
   No descriptor relabel over capacity-strided cache memory is admissible.
3. **G3-M**: a genuine no-grad Eshkol schedule over the reviewed kernels, sharing
   current parameter identities but creating no VJP graph, contribution, or
   gradient mutation. Learned position lookup uses absolute position 0 or 1;
   decode must not reuse position 0. Cached and full-prefix schedules must agree
   before cache execution is advertised. G3-N and G3-T precede G3-M.
4. **G3-S**: the sampler contract below, with a separate registered typed RNG.
   It can be reviewed independently of G3-M. It consumes neither N3K initializer
   authority nor N2 dropout state. A private kernel schema and capability identity
   must be accepted before code; this proposal does not freeze either.
5. **G3-G**: combine G3-M/G3-S with independently reviewed cache transactions,
   output owners, T1 decoding, error translation and one source-composed registry
   owner. No G3 public execution before those prerequisite contracts are accepted.

C4 or a larger context is a separate later decision: it changes the learned
position tensor to [C,4], initializer row/draw schedule, model profile identity,
full-prefix T3/T4 primitive rows and full-capacity attention rows. It is not
admitted by this C2 proposal or by A2's ability to allocate a larger cache.

## Proposed exact sampling contract

The following is one decision-ready candidate for G3-S; it is not an existing
feature. All inputs are CPU, dense, zero-offset f32[1,256] logits. All 256 values
must be finite, even values later removed by filtering. There is no logit cast,
CPU transfer, numerical Eshkol list loop, approximate gradient or alternate
implementation. Native bounded loops are acceptable only as a reviewed kernel.

Greedy selects the largest numeric logit, breaking exact numeric ties by lowest
token ID. +0 and -0 tie. Greedy does not divide, exponentiate or consume RNG.
Require canonical inactive sampling settings (temperature=1, k=256, p=1) in greedy
mode; inconsistent policy fields are invalid rather than silently ignored.

Categorical policy admits finite positive f32 temperature, exact integer k in
[1,256], and finite f32 p in (0,1]. Temperature and p must enter through an exact
f32 representation contract; no implicit narrowing from an Eshkol generic real.
The config/carrier proposal must resolve that representation before freezing its
constructor. "Temperature", "top-k" and "top-p" are one ordered pipeline, with
k=256 and p=1 as the default settings:

1. Find maximum logit m (ties have no numerical consequence). In token-ID order,
   evaluate separate binary32 statements d[j]=logit[j]-m,
   z[j]=d[j]/temperature, e[j]=expf(z[j]); require finite d,z,e. Underflow of
   e to +0 is permitted and explicit. Subtraction/division overflow is a
   categorized invalid numeric input; it is not repaired by clipping.
2. Sum e in token-ID order from +0 using separate binary32 additions, yielding
   S>0. Compute a[j]=e[j]/S with separate binary32 division. Sort all tokens
   by a descending, then token ID ascending. This order is based on the
   rounded probabilities, including ties introduced by rounding/underflow.
3. Retain the first k. Sum their a values in that sorted order from +0 to K>0,
   then compute b[j]=a[j]/K in that order. Compute cumulative b with binary32
   additions in the same order. Keep the shortest nonempty prefix whose
   cumulative value is >=p. If rounding leaves the full cumulative value <p,
   keep all k. There is no epsilon comparison. Thus p=1 can retain fewer than
   k if the rounded cumulative value reaches 1 earlier; this is intentional and
   must have literal regression vectors, not an undocumented special case.
4. For the retained prefix, use its b as weights. Sum b in sorted order from +0
   to W>0. Obtain one u as below and compute r=u*W as one binary32 multiply.
   Sum b in the same order; select the first token whose cumulative weight is
   strictly greater than r. If multiplication rounds r to W, or the accumulation
   otherwise reaches no strict crossing, select the last token with b>0.
   This explicit upper-endpoint rule ensures a token is selected without selecting
   a zero-weight tail. u=0 selects the first positive interval, not a zero-weight
   entry. No second random draw or rejection loop exists.

Arithmetic requires the same reviewed binary32 platform and floating controls as
the numerical providers: round-to-nearest ties-to-even, FTZ/DAZ off, contraction
disabled, no excess precision/fast-math. Validation preserves incoming fenv and
all outputs; invocation may accrue ordinary successful arithmetic flags. Exact
libm bits and deterministic continuation are scoped to the supported toolchain/
libm lane, not every CPU/libm. Unsupported deterministic environment rejects
explicitly. Independent high-precision references test mathematical behavior with
declared tolerance; separate literal bit vectors test this exact ordering.

## Proposed RNG and token publication

Algorithm candidate: `g3.philox4x32-10.categorical-f32.v1`. Words are exact i64
[1,seed,counter-low,counter-high], with unsigned 64-bit counter halves represented
numerically as u or u-2^64 above INT64_MAX. Seed constructor range is nonnegative
signed i64, following the bounded M3 constructor policy. The state is an immutable
same-aggregate registered kind; neither a numeric list nor an initializer/dropout
value grants this authority. Exposed words are detached data. The numeric layout
is proposed logical state only, not a serialized byte format.

Use Philox4x32-10's documented N3K round function and lane/counter ordering, with
key `uint64(seed) XOR 0x4733434154454731` split low/high into k0/k1. This explicit
domain tag distinguishes same-seed sampler use from the untagged initializer;
it is not a claim that every differently seeded stream is globally disjoint.
For each categorical emitted token consume one whole block at the original
counter, use lane 0 only, and discard lanes 1 through 3. Define the exact uniform
as `(float)(lane0 >> 8) * 0x1p-24f`, giving [0,1). Advance the complete 128-bit
counter by one block. The maximum counter is an exhausted sentinel: it may be
returned as successor but never consumed or wrapped.

Every categorical emitted token consumes that one block, including a singleton
candidate set and EOS. Greedy consumes zero and preserves the exact state.
Prefill, caller-supplied decode, empty generated output, policy validation and
failed token attempts consume zero. A future admitted batch profile must separately
freeze its row schedule; N=1 here implies no claimed multi-row scheduling.

The generator takes a logical copy of initial RNG words; immutable input and output
RNG objects never mutate when the generator advances. A token attempt computes its
candidate and successor speculatively, appends that candidate's K/V and validates
all return/output resources, then atomically publishes cache length, generated ID
and successor RNG. A failure before that publication leaves the previous RNG and
cache prefix unchanged. Counter exhaustion rejects before token publication.
Fallible final output decoding/allocation must not silently move an already emitted
token back before its A0 commit point; output preallocation/decoding lifetime is a
separate G3-G obligation.

## EOS and output interpretation

V256 T1 is the byte vocabulary and supplies no named EOS special token. The proposed
config admits EOS `#f` or one exact integer in [0,255], explicitly a caller-selected
byte terminator. It neither changes tokenizer identity nor adds a vocabulary item.
Only an emitted matching byte stops generation; a matching byte already in the
prompt does not. An emitted terminator is included in IDs, length, decoded input
and cache, with the same RNG draw as any other categorical token. No synthetic
terminator is appended at maximum-new-tokens or context exhaustion. Output text is
decoded using the actual T1 byte-decoding policy; arbitrary bytes do not imply
valid Unicode. Zero-budget generation returns an empty generated sequence after
successful prefill, unchanged RNG and cache length P.

## Required acceptance evidence

- Every added exact forward row: independent mathematical reference and adjacent
  shape/dtype/device/nonfinite/fenv rejections; output-byte and input preservation
  on failure. New differentiable/backward claims, if any, require central
  differences, not inference from an old row.
- T1 head permutation has T!=H and must use distinguishable marked values; it
  catches assumptions hidden by M3's T=H=2 layout.
- All C2 routes: P1 cached prefill against true uncached T1, P2 cached prefill
  against current M3 full T2, and P1 then decode at absolute position 1 against
  full-prefix T2. Compare all logits (supported-lane bits where arithmetic order
  guarantees them, independently justified tolerance otherwise), both K/V and
  committed lengths/masks. No-cache oracle must actually use Tk=prefix length.
- Greedy signed-zero/equal-logit ties; probability ties created by quantization;
  k=1/256; p boundary equality, p=1 rounded sums and underflowed tails; u=0 and
  maximum uniform endpoint; temperature overflow/underflow rejection; NaN/Inf in
  discarded entries; highest/lowest token ID and EOS at the only emitted token.
- Literal Philox vectors with this domain tag, lane discard, multi-call raw sampler
  continuation, low-word carry, exhaustion, same-seed identity-independent equality,
  greedy/singleton/EOS/zero-budget draw counts, failure retry and byte-identical
  continuation after an accepted data-only RNG save/reload contract exists.
- Native and compiled public API sanitizer/failpoint tests must cover every
  allocation before token publication, cache rollback, detached RNG survival,
  no gradient/graph creation, and one registry owner. No full CI or repeated full
  builds were requested or performed by this review.

Deterministic save/reload remains a required future gate, not an assertion that
C2 trainer checkpoints currently serialize generators, A2 caches, or this RNG.
