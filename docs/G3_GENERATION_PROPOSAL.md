# G3 generation: contract and compiled reachability proposal

Status: **bounded design accepted; exact primitive ABI deltas proposed separately**.
Tracking [#89](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/89).
The [binding decision 5748532295](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5748532295)
accepts proposal commit `c354cb2b96c521bb7c8115e20da21c1aa7030f64` with the
draw-dependent exhaustion clarification incorporated below. This is design
acceptance, not runtime acceptance or G3 completion. Exact [G3-N](g3/G3_N_ABI_PROPOSAL.md)
and [G3-S](g3/G3_S_ABI_PROPOSAL.md) ABI proposals still require disposition;
[G3-T seams](g3/G3_T_SEAM_PLAN.md) remain separately gated.
No production API, ABI, capability, configuration schema or file format is changed.
No implementation PR, full build, CI dispatch or generation-completion claim is
part of this phase. The implementation base is merged M3
`aa9e78f96b5545b52b77c238b92b26c281b41af0`, tree
`7d81855be5ad35a44e74a708642668f8d789ebff`; ancestry was checked locally.
M3 acceptance-document follow-up belongs to integration and is not edited here.

The recommendation is a **C=2 diagnostic prerequisite slice**, followed by a
separately approved larger profile and persistence work before G3 completion.
The current model has only two learned positions. One-token prefill plus one
generated token is the largest new-token trajectory this proposal admits.
Treating arbitrary A2 cache capacity as additional model context would be wrong.

## What actually works

The executable [probe](../scripts/probe-g3.sh) builds development callers against
an existing verified-toolchain M3 package. It does not build prerequisites or add
a generator. [Package evidence](g3/PACKAGE_REVIEW.md) records exact commands,
hashes and limitations. Independent reviews cover [numerics/sampling](g3/NUMERICAL_REVIEW.md),
[cache/lifetime](g3/LIFETIME_REVIEW.md) and [compiled packaging](g3/PACKAGE_REVIEW.md).
The consolidated decisions below govern where review alternatives differ.

| Compiled observation | Consequence |
|---|---|
| Fixed M3 i64[1,2] ingress produces 2,048 exact logits bytes, repeated identically in two processes | Existing f32[1,2,256] diagnostic reachability only |
| One-token and three-token ingress reject with `shape-mismatch` | Neither T1 nor T3 execution exists |
| `:rng #f` rejects as `unsupported`; `:cache` and `:no-grad?` reject as `invalid-argument` | No RNG/cache/no-grad option contract exists |
| Eval logits survive output release and accept a zero-seed analytic VJP | Eval still owns a backward-capable graph |
| Failed ingress/options leave a usable model and unchanged next logits | Evidence for these existing admission failures, not generator transactions |
| All nine A0 declarations compile and zero-argument calls fail with exact arities | Declaration evidence only |
| Production-only `transformer.generation` import fails | No installed generation runtime |

This run reused an M3 package whose seven inventories, actual archive/member/global
symbols and 61 available project source/facade files were checked. Fresh caller
compilation and exact eight-file public closure were verified. Source correspondence
is not a fresh package-build attestation. The host is explicitly unsupported
CachyOS/Clang-LLVM22, using the pinned Eshkol revision. This is neither supported
Ubuntu22/LLVM21 evidence nor a new sanitizer, numerical-parity, cache or sampling run.

## Proposed profile, admission and configuration

Retain N1/V256/D4/Hq=Hkv2/Dh2/L1/F8, affine LayerNorms, exact-erf GELU,
bias-free projections, tied head, 14 unique M3 parameters, learned positions
f32[2,4], CPU f32 and no dropout. Require the genuine same-aggregate M3 model
in eval mode; never change mode implicitly. Require the baseline T1 V256 raw-byte
tokenizer with no specials/prefix/suffix and its actual fingerprint. Retain the
model/tokenizer; do not destroy them when the generator closes.

Prompt P is 1 or 2; IDs are exact CPU i64[1,P] in [0,256). Decode input is
i64[1,1]. Capacity is exactly 2, positions start at zero, and an entire generation
request must satisfy `0 <= max-new-tokens <= 2-P` before replacing any cache.
No empty prompt, padding, truncation, sliding window, eviction, implicit BOS/EOS,
batch widening or expectation of early EOS relaxes that bound. A manual decode
requires an existing cache with length 1 and appends at position 1.
Only authentic G3 input owners from the two proposed input constructors enter
these operations: either constructor's [1,1] value may prefill or decode, and
[1,2] may only prefill. Output-ID/length clones, T1 shells and M3T inputs do not
gain input authority merely by matching a shape.

Candidate `generator-create` config is a detached, bounded flat option list,
**not** an X1 resolved-run extension. Exactly eight key/value pairs are required:
`:profile 'diagnostic-c2`, `:sampling 'greedy|'categorical`,
`:temperature-bits`, `:top-k`, `:top-p-bits`, `:max-new-tokens`, `:eos`, and
exactly one of `:seed`/`:rng`. Unknown, duplicate, missing, malformed, cyclic or
overlong lists reject. The two bit fields are exact integers in [0,2^32-1]
representing binary32 by bit copy, with positive finite temperature and
0<p<=1; ordinary real narrowing is not permitted. Top-k is an exact integer
1..256; max-new-tokens is 0..1 and is further checked against each prompt.
Seed is nonnegative signed i64; `:rng` requires the proposed authenticated G3 RNG
kind, never M3T initializer/N2 dropout state or a list of words. A seed starts
counter zero; a state is logically copied. Greedy requires temperature=1,
k=256, p=1 rather than silently ignoring policy values. Determinism is mandatory
in this slice; environment failure is explicit `determinism-unavailable`.

EOS is `#f` or an explicit integer byte ID 0..255. It is a caller-selected byte
terminator, not a newly invented T1 special token. Only an emitted match stops;
a prompt match does not. Include emitted EOS in IDs, text bytes, lengths and
cache. No synthetic token is appended at budget exhaustion. Named-special EOS,
strict Unicode, V>256 and larger context require separate contracts.

## Public surface and closed carrier proposal

The nine A0 names/arities remain unchanged:

| Name | Arity | Proposed bounded result |
|---|---:|---|
| `generator-create` | 3 | idle generator owning RNG/workspace, initially no cache |
| `generator-prefill!` | 2 | new immutable CPU f32[1,256] last-position logits |
| `generator-decode-step!` | 2 | new immutable CPU f32[1,256] appended-position logits |
| `generator-generate!` | 2 | opaque immutable generation output |
| `generation-output-ids` | 1 | new one-element list of owned i64[G], G=0 or 1 |
| `generation-output-lengths` | 1 | new owned CPU i64[1] |
| `generation-output-text` | 1 | new one-element list of detached raw bytevectors |
| `generation-output-rng` | 1 | independent immutable, explicitly releasable G3 RNG snapshot |
| `generation-output-cache-lengths` | 1 | new owned CPU i64[1], value P+G |

Six additive names are accepted design commitments for the later reviewed aggregate:

| Proposed name | Arity | Authority and lifetime |
|---|---:|---|
| `generation-input-create` | 1 | Copy genuine same-aggregate T1 i64[P] into new owned i64[1,P]; P=1 or 2 |
| `generation-token-input-create` | 1 | Validate one exact integer 0..255 and create owned i64[1,1] for manual decode |
| `generation-tensor-release!` | 1 | Release only the closed G3 tensor union: these inputs, last logits, output-ID and length clones |
| `generation-output-release!` | 1 | Release output's private snapshots, preserving independent accessor results |
| `generator-close!` | 1 | Release cache, scratch, binding snapshots and owned RNG; never borrowed model/tokenizer |
| `generation-rng-release!` | 1 | Release an independent G3 RNG snapshot; generators already seeded from it remain valid |

These are proposed narrow adapters, not a generic tensor constructor, native-pointer
release or existing API. All non-release use of an authentic dead owner is
`invalid-state`; release of the exact dead owner is idempotent. Wrong-kind,
forged/copied/unregistered/cross-aggregate shells are `invalid-argument`; busy
or active-borrow owners reject before mutation. Authentication precedes payload
access. A live shell allocation remains necessary for stale/idempotent guarantees.
Dropping a reference is not evidence of finalization. Tensor clones, RNG and raw
bytes survive generator close, output release and subsequent model mutation.

Proposed validation order is receiver authentication/liveness, argument-kind
authentication, bounded config/option syntax and scalar ranges, carrier metadata
and token/context bounds, profile/tokenizer/provider admission, current model/cache
state and RNG exhaustion, then numerical preflight/allocation. Within a stage,
inspect arguments left to right and tensor metadata rank/extents, dtype, device,
layout, then values. No mutation precedes these checks. Release first authenticates
the exact identity, then returns on an exact dead owner before busy/borrow checks.

| Failure | E1 category |
|---|---|
| Wrong/forged/copied/cross-owner kind; malformed config or scalar policy | `invalid-argument` |
| Authentic dead/busy owner, missing cache, train mode, stale cache binding; exhausted authenticated RNG only when a categorical draw is required | `invalid-state` |
| Admitted carrier rank/extent, token range, context overflow or model/tokenizer vocabulary mismatch | `shape-mismatch` |
| Admitted carrier dtype, device or layout mismatch | `dtype-mismatch`, `device-mismatch`, `noncontiguous` respectively |
| Well-formed unavailable model profile, tokenizer policy or exact provider row | `unsupported` |
| Required deterministic FP/toolchain execution unavailable | `determinism-unavailable` |
| Nonfinite numeric input/intermediate or arithmetic domain overflow | `invalid-argument` |
| Allocation failure or violated private invariant | `internal`, with bounded source code distinguishing the cause |

Output accessors preserve A0's `device-mismatch` for inconsistent retained storage.
Errors carry the invoked public operation, bounded data-only details and `cause #f`;
no raw pointer, hidden receiver or executable diagnostic. Required arguments not yet
authenticated are never inspected after an earlier failure. These are proposed
public rules, not changes to predecessor native error schemas.

Repeated T1 encode still retains process-lifetime identities. Reuse encoded
prompts and use the scalar token adapter for manual decode. The reviewed
source-private T1 decoding seam must borrow authenticated G3 i64[G] without
permanent T1 shell enrollment; range-check, copy raw bytes, end the borrow in the
same call. Public `tokenizer-decode` stays unchanged and rejects G3 tensors.
`generation-output-text` is the supported output decoding route.

## Numerical and transport dependencies

I1/I2 own higher-rank carriers, but their K1 storage-copy capability still verifies
only ranks zero/one. Do not label their rank-two ownership as a verified generic
tensor operation. M3T's fixed roles and separate N3K/N2/A2 discovery runtimes
provide a pattern for closed dispatch, not a general new resolver.

The numerical review enumerates every missing row. In compact form, the new T1
forward-only scope needs embeddings [1,1,256,4]/[1,1,2,4]; bias-free linears
[1,1,4,4], [1,1,4,8], [1,1,8,4], [1,1,4,256]; LayerNorm/residual [1,1,4];
split/merge layout [1,1,2,2]; true uncached attention [1,2,2,1,1,2].
N2 already verifies GELU [1,1,8], and A2 already verifies cached full-capacity
attention [1,2,2,1,2,2] and [1,2,2,2,2,2]. A separate bounded numerical
provider with distinct capability names is preferred over silently expanding
frozen predecessors. New forward-only rows advertise no backward capability;
any added backward claim requires analytic and numerical-gradient tests.

The private Eshkol no-grad schedule must read the same canonical model parameters
without creating saved VJP graphs, gradient contributions or gradient plans.
Its full-sequence path owns f32[1,S,256] scratch, S=1/2; public last logits are
an explicit bit-preserving copy to f32[1,256]. Never relabel an existing tensor.
Native code handles reviewed kernels, ownership/copying and metadata; no recursive
scalar numerical Eshkol path, Python runtime, CPU/cast fallback or built-in AD.

Cache K/V are separate A2 allocations f32[1,1,2,2,2], lengths i64[1], validity
bool[1,2]. Stage f32[1,2,A,2], A=P for prefill or 1 for decode. Nested borrowed
views retain full C2 shape f32[1,2,2,2]. Materialize bool[1,A,2] explicitly;
query positions are i64[1,A], key positions i64[1,2]={0,1}. False mask protects
unused initialized tails; it does not authorize a shorter dense view over
capacity-strided storage. Learned-position K is already position-conditioned
before projection; the binding decision admits these unrotated keys for this
exact profile. G3-T must document and directly test that consumer contract;
existing A2 ABI, RoPE semantics and capability rows stay unchanged.

## Mutation, failure and resource boundaries

Use serialized, non-reentrant calls and the model's existing active-frame guard.
Bind each cache to canonical parameter identities, exact value bits, topology,
profile and eval mode. Decode rejects a mismatch before staging; exact restoration
may revalidate it. Fresh prefill may establish a new binding after between-call
model changes. Creation does not permanently bind old value bytes. Model gradients,
including already-present gradients/counts, remain unchanged on all paths.

Prefill computes into a new candidate cache and new logits. Close views and
preflight all cleanup before a no-failure swap/publication; any failure preserves
the old cache and RNG. Decode stages one transaction and owned logits, closes
views, validates commit eligibility, then publishes cache/result without fallible
work. Failure aborts and scrubs the tail, preserving committed prefix and RNG.
Manual prefill/decode consume no random draws, including manual EOS append.

Generate validates the whole request before replacing the old cache, including
exhaustion only when categorical sampling with nonzero budget requires a draw.
Such exhaustion rejects before the initial prefill commit. Its successful
prefill is an explicit initial commit. Zero budget still prefills and returns an
empty output, cache length P and unchanged RNG. After prefill, token failure leaves
the prompt-only cache. Each sampled token, including EOS, must run through the
model and append K/V before it is emitted. Prepare token, successor RNG, output
storage, raw-text capacity and cleanup before publishing cache/ID/RNG together.
No recoverable error may occur inside that commit tail. Later-token failure in a
future larger profile preserves earlier committed tokens and raises without a
partial output, as A0 requires. C2 cannot measure that later-token behavior.
Output shell/list, empty-ID storage, final RNG snapshot and raw-byte result
storage are preallocated before the initial prefill commit, including zero budget.
The final return only seals already-prepared data and cannot introduce a fallible
packaging step after a successful zero-budget prefill or token commit.

Generator, cache, output, accessor clone and RNG have distinct owners. Persistent
model/tokenizer retention is separate from released G3 payloads. Count both peak
live storage and cumulative identity/arena storage: existing I2/M3 tombstones
prevent assuming flat memory. G3 must declare and measure any cumulative shell
cost; fixed cache capacity alone proves neither bounded process memory nor a
production long-running loop.

## Deterministic sampling and RNG

The exact proposed arithmetic is in the numerical review's sampling section:
greedy max-logit/lowest-ID ties, or temperature -> binary32 softmax -> descending
rounded probability/ascending-ID order -> top-k -> renormalization -> shortest
top-p prefix -> weighted CDF. All 256 logits and every intermediate must be finite;
explicit exponential underflow to zero is admitted, overflow is rejected without
clipping. No unspecified host sort, epsilon cutoff or alternative arithmetic.
Top-p compares rounded cumulative probabilities: retain all k if no prefix reaches
p; p=1 may reach 1 early. CDF uses strict greater-than and last-positive-weight
upper-rounding protection. These boundary decisions need literal golden vectors.

Candidate algorithm is `g3.philox4x32-10.categorical-f32.v1`, logically exact
i64[4] words [1,seed,counter-low,counter-high], signed encoding of unsigned counter
halves as in N3K. Typed G3 authority is separate from numeric words and other RNG
kinds. Key is seed XOR `0x4733434154454731`; this distinguishes same-seed use
from the initializer, not every possible differently seeded stream. Consume one
Philox block per categorical emitted token, use lane0's high24 bits times 2^-24,
discard other lanes, and advance the full 128-bit counter. Maximum counter is an
unconsumable exhausted sentinel. An authentic exhausted snapshot remains admitted
at construction and for greedy, prefill, manual decode and zero-budget generation;
each preserves it. The final consumable block may publish that sentinel successor.
Singleton categorical selection and EOS still
consume one block; greedy, prefill, manual decode, zero budget and failed token
attempts consume zero. Immutable input/output snapshots never advance in place.

The native sampler is a separately reviewed no-grad kernel over f32[1,256].
Binary32 rounding, no contraction, FP controls and supported libm identity are
part of determinism. Exact bits are scoped to the supported lane. This phase
measures no sampler, RNG or cross-platform equivalence.

## Proposed dependency graph and package owner

Planning IDs below are proposed; they are not existing accepted workstreams.

| ID | Scope | Depends on |
|---|---|---|
| G3-N | Missing exact T1 forward/no-cache rows; standalone provider identity and evidence | merged N2/N3K/A2/K1/I1/I2 |
| G3-T | Closed input/logits/bool/position transport; learned-position cache admission; private T1 borrow; explicit release/authentication contracts | merged foundations; G3-N row contracts |
| G3-S | Exact sampler and separately authenticated RNG, including prepare/commit and release | merged K1/I1/I2; accepted sampling contract |
| G3-M | True no-grad cached/full-prefix Eshkol schedules and value-bound model/cache lifetimes | G3-N, G3-T |
| G3-G | Nine A0 operations plus accepted additions, joint cache/RNG/output transaction, one completed package | G3-M, G3-S, G3-T |
| G3-R | Reviewed rank-two model restore and typed sampler-state reconstruction for generation equivalence | separate C2/K2/model composition decisions; G3-G |
| G3-C4 | Separately proposed larger learned-position profile and exact numerical rows for repeated decode | profile/initializer decision and preceding bounded work |

G3-N and G3-S may proceed independently after contract acceptance. G3-T contracts
can be reviewed alongside them, but implementation must wait for merged APIs.
There is no dependency on unmerged L3S. No current missing contract is replaced by
an invented signature in code.

Build one source-composed successor of M3, retaining its seven facades and adding
`transformer.generation`. Preserve existing public arities and the one owner of
P1/T1/E1/I2 registries. Fixed private routes choose each provider; no caller-supplied
resolver or canonical generic K1 symbol. Do not link M3/M3T/T1/C2 or other owning
archives together. Retain the inherited 93-global/87-export baseline; enumerate
the exact successor manifests once additions and private seams are dispositioned.
No predicted count is a frozen package contract. Private symbols, source roots,
native helpers and development observers must remain unavailable to public callers.

## Serialization and acceptance still required

This proposal changes no T1/T2/X1/C1/C2 bytes or model parameter shape. G3 RNG is
new logical state; C2 dropout words do not serialize or authenticate it. Do not
serialize caches, pointers, live registries, leases, graph objects or executable
code. G3-R must choose versioned/checksummed data-only RNG/config continuation,
admit the model's rank-two tensors truthfully through the persistence capability
route, and restore model/tokenizer identities. Reconstruct cache by no-grad
prefill of token history without RNG consumption. Persistence is not proved by
rerunning initialization or merely saving identical seeds.
Reconstruction uses manual prefill/decode replay, not `generator-generate!`:
a full length-2 history remains replayable when the saved policy requests one new
token, even though a new generation request at that length must reject.

Before bounded C2 implementation acceptance require:

- Independent reference parity for every new row and every vocabulary logit;
  true T1 no-cache versus C2 prefill, T2 no-cache versus P2 prefill, and P1+decode
  at position1 versus T2 full-prefix, including both K/V and masks/lengths.
  Use exact bits where the arithmetic proves them; justify any tolerance.
- Numeric ties, signed zero, f32 sort/cutoff/CDF boundaries, all filtering limits,
  nonfinite/overflow/subnormal cases, literal domain-tagged Philox vectors,
  lane discard, carry/exhaustion, EOS and all zero/one-draw cases.
- Every public arity/option/type/rank/extent/ID/dtype/device/alias and capability
  negative; no-grad observed through unchanged gradient values/counts and absence
  of graph allocation. Test stale/mutated/exact-restored models and busy frames.
- Cache replacement, every staged-layer/view/allocation failure, unchanged prefix
  and RNG on rejection, cleanup retry, old outputs across replace/close, detached
  clone/RNG survival, forged/cross-owner/released shells and idempotent release.
- ASan/UBSan and supported LSan; exact payload/control/arena counts at 1,024/8,192
  reuse and create/release cycles, with RSS separately reported. No hidden lower
  limit or finalizer argument substitutes for evidence.
- Fresh independent package/AOT builds, exact source/native/defined/undefined/
  strings/closure manifests, duplicate-owner and private-link negatives, actual
  execution with no Python runtime, and supported-lane exact-head CI after the
  implementation phase is authorized.

Before full G3 completion additionally require data-only save/reload equivalence
for logits, IDs, raw bytes, lengths and RNG; repeated decode and later-token
failure/EOS gates on a separately admitted larger profile. The C2 slice leaves
these obligations open. This does not prevent separate acceptance of a bounded
prerequisite whose own required gates pass.

## Historical consolidated request (dispositioned)

The binding decision linked above settles these high-level choices. They are
retained as the decision record, not reopened questions. The next disposition
concerns only the exact primitive ABI deltas linked at the top.

1. Accept the minimal C2 slice as a prerequisite only, keeping larger-context
   generation and G3 completion separate; or direct a separately contracted C4
   profile now (new position/initializer identity and T3/T4 rows required).
2. Approve the six additive names, closed releasable carriers, exact-bit config
   fields, explicit raw-byte EOS and mandatory deterministic scope above.
3. Approve genuine private no-grad schedules, learned-position A2 cache admission,
   full-capacity mask materialization and the no-enrollment private T1 decoder.
4. Ratify successful prefill as generate's initial commit, zero-budget prefill,
   per-token cache/RNG/output publication, and value-based cache staleness with
   exact-restoration acceptance.
5. Accept the exact f32 cutoff/CDF rules, domain-tagged Philox algorithm, one-block
   categorical draw policy, separate typed RNG owner and unsupported-environment
   rejection; or identify the specific rule to revise before kernel implementation.
6. Disposition G3-N/T/S/M/G/R/C4 ordering and one successor registry owner; assign
   persistence/profile follow-ups without changing current public formats or
   making a dependency on L3S.

All six questions are consolidated from the independent reviews and executable
evidence. Implementation starts only after the applicable contracts are accepted
and their upstream implementations merged.
