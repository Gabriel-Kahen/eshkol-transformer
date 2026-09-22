# G3-T exact private-seam contract

**Accepted, implementation dependency-gated**, by
[decision 5752756205](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5752756205),
mirrored in [G3 issue #89](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/89#issuecomment-5752756256).
Reviewed commit: `a7a45908d9b5224ad6d2d6cb08926ecc3bb75ccf`; proposal SHA-256:
`78a770237a000bc0d3a7b22c678fc1a6692a9d8052456bbed350c02e4366b9fe`.
Base: merged contract PR #92, `4353bd2f14ae4fb50c877c27fc73b8edef8e2f30`,
tree `c5c7d2879cefe2d331b2f5167627c947409d7af0`. This is one accepted
source-private contract, not installed headers, a public ABI or implementation.
Production integration remains blocked until **both** G3-N #93 and G3-S #94
implementations are independently approved and merged.

The [binding C2 decision](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5748532295)
and [accepted N/S decision](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5751933660)
remain unchanged. [N](G3_N_ABI_PROPOSAL.md) and [S](G3_S_ABI_PROPOSAL.md)
specify numerical operands/arithmetic; [G3](../G3_GENERATION_PROPOSAL.md)
specifies configuration, nine A0 names plus six additions, EOS, ownership and
transactions. G3-T installs **none** of those generation names. C2 admits P=1/2,
G=0/1, and P+G<=2; this cannot prove repeated generation or save/reload.

The subsequent [E3-CG exact shared-source contract](../E3_SHARED_CALL_CONTRACT.md)
consolidates the three pin helpers and 38 checked adapters for G3 and E3. Its
source ownership and lifecycle refinements are accepted by
[root decision 5771662657](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/102#issuecomment-5771662657);
Shared runtime was subsequently accepted and merged in PR #105; see its
[integration provenance](../M3_SHARED_CALL.md#integration-provenance). The exact
E3 consumer design merged separately in PR #104. The accepted G3 calls, slots,
kinds, 29/95 inventory and public counts below do
not change. E3-CG owns that shared contract; this G3-T owner retains affected-contract
review. Contract merge and explicit root dispatch precede shared implementation;
the G3-T N/S implementation hold is unchanged.

The [readiness and future witness checklist](G3_T_READINESS.md) records G3-N's
completed prerequisite and the remaining shared/G3-S gates at main `ba0e37d`.
It is planning only and changes none of the accepted contract below.

## Accepted decisions

1. One shared invocation guard and fixed 14-pin frame, including temporary use
   of I2's parameter control-pin count to exclude reset/contribution preparation.
   Gradient preservation also relies on the closed no-gradient/no-callback path.
2. The exact closed owner kinds, registry slots, calls and publication protocol
   below, including preallocated T1 raw-byte decoding without shell enrollment.
3. Fixed provider routing and the private-only successor tuple: seven facades,
   87 boxed exports, 93 globals, six selected provider/cache objects.
4. Separate normal-package and instrumented private witness gates. No test hook
   or new public transport escape is admitted to make private code callable.

Root accepted these four bounded deltas in the decision above. No broader
context, generic parameter/tensor access, resolver, gradient plan, checkpoint
format or P1 slot is proposed.

## Authentication, slots and shared admission

One source-composed root owns M3/M3T/I2/P1/T1 and G3 identities. All public-facing
shells are authenticated by exact `eq?` lookup in the root-owned registry before
reading native payloads. A one-word shell tag is inert, not authority. Published
shells and native control records become unreused tombstones on release; copied,
forged, wrong-kind and cross-aggregate identities reject. Native entry points
independently admit pointer values through the closed native registry before
any dereference. Only complete native owners are enrolled; partial constructors
free unpublished allocations and preserve the first error.

New Eshkol root cell `g3t-registry` is `(vector '())`. Each entry has exactly:

| Slot | Content |
|---:|---|
| 0 / 1 / 2 | exact shell / kind / pending, live or dead |
| 3 | native owner, or #f for the private call entry |
| 4 | parent generator for a pending result/call; otherwise #f |
| 5 / 6 | authentic M3T model entry / baseline T1 tokenizer for generator/call |
| 7 | immutable normalized policy vector for generator; otherwise #f |
| 8 | output auxiliary vector `#(raw-bytes[G] i64le-staging[8G])`; otherwise #f |
| 9 | active private call entry for generator; otherwise #f |
| 10 | rooted pending-child ledger for generator/call; otherwise #f |
| 11 | `diagnostic-c2` |

Kinds are `generator`, `input`, `logits`, `output`, `ids`, `lengths`,
`cache-lengths`, `rng`, and private `call`. Dead entries retain only slots0..2.
Output publication clears parent/model/tokenizer roots and staging; raw bytes
remain output-owned until release. Detached accessor copies retain no parent.
Slot7 is `[diagnostic-c2,mode,temperature-bits,k,p-bits,max-new,eos]`;
it retains neither the original option list nor the source RNG shell after
copying. Slot6 fixes tokenizer identity/fingerprint for the generator lifetime.
Root promotion and canonical readback must follow M3T's existing region-safe
staging discipline; merely storing a transient pointer is insufficient.

Native kind IDs1..8 respectively are GENERATOR, INPUT_I64_R2,
LAST_LOGITS_F32_R2, OUTPUT, IDS_I64_R1, LENGTH_I64_R1,
CACHE_LENGTH_I64_R1, RNG. Each begins with registry next/kind/state/busy;
state0/1/2 means pending/live/dead. No native CALL kind exists: the generator
context is address-stable through close. Inputs own I1[1,P]; logits own
I2[1,256]; ID/length clones own exact I1[G]/I1[1]. RNG owns aligned inline
i64[4], not an I1 shell. OUTPUT owns I1[G], generated/cache lengths and immutable
RNG words; text has its separate rooted Eshkol storage. OUTPUT payload fields
are exactly parent_ctx, P, G, ids, length, cache_length, rng[4], numeric_ready,
ids_copied and text_ready. The three flags start false; prepare, copy and accept
advance them in order, once each. The context separately owns candidate_ready.
No inspected numeric
word list, initializer or dropout owner grants G3 RNG authority.

Existing P1 private slots0..63 stay unchanged: slot26 supplies mode,27/28
parameter tensor/identity. Existing M3T model entry slots0/2/3/8/10/11 supply
shell/liveness/native owner/canonical14/active frame/profile. Require sealed
N1/V256/D4/H2/Dh2/L1/F8/C2 topology, 14 unique canonical handles with the tied
role using one handle, profile and eval, and the authentic raw V256 tokenizer
with empty specials/prefix/suffix. No model-release API is introduced.

Use existing transient `m3-call-state` once at the outer invocation. During G3,
generator slot9 and M3T model slot10 point to the same private call entry;
native `owner.active` points to that generator context. Admission requires all
three idle, then acquires pins before publishing busy tokens. Rollback clears
only tokens acquired by this call. A manual M3T frame between invocations blocks
G3 on its model, not an otherwise idle different model. While an invocation is
running, the aggregate-wide guard excludes all other M3/G3/M3T outer entries.

Gate **all 38** `m3t-public-*` rows in the existing 46-line M3 rename map by
replacing only that Eshkol prefix with `g3t-checked-m3t-*`, retaining exact
suffixes, arities, order and C ABI target names. The other eight M3 rows are
unchanged. Each wrapper enters the shared guard once and calls the original
trusted function directly. M3's internal schedule keeps using the originals;
wrapping those originals would incorrectly reject M3's legitimate nesting.

## Fixed14 parameter pins and cache binding

Canonical pin order is K, attention-output, Q, V, FFN-down, FFN-up,
norm1-beta/gamma, norm2-beta/gamma, tied-token/head, final-beta/gamma, position.
Their shapes are four[4,4], [4,8], [8,4], four[4], [256,4], two[4], [2,4]:
1,184 f32 values, exactly **4,736 bytes**. No role may choose a parameter index.

The accepted common source contract preserves C-only type `et_g3t_model_pins_internal`.
Its [exact declaration and begin/check/end lifecycle](../E3_SHARED_CALL_CONTRACT.md#3-exact-private-pin-declaration-and-geometry)
are consolidated there. It contains `self`,
`parameters[14]`, `identities[14]`, `values[14]`, read-only `views[14]` and
`uint16_t held_mask`. It is embedded in the stable authenticated generator context for G3, or in E3's
separately authenticated stable frame, never copied, moved or reallocated while
held. `self` is copy detection, not context authentication. Only closed consumer
adapters derive its address and canonical arrays after registry admission. This is a **new** frame-pin contract;
`et_f32_tensor_scoped_*_internal` retains its synchronous stack-only semantics.

```c
int32_t et_g3t_model_pins_begin_internal(
    et_f32_parameter *const parameters[14], const void *const identities[14],
    et_g3t_model_pins_internal *pins, et_f32_tensor_error *error);
int32_t et_g3t_model_pins_check_internal(
    const et_g3t_model_pins_internal *pins, et_f32_tensor_error *error);
void et_g3t_model_pins_end_internal(et_g3t_model_pins_internal *pins);
```

Only authenticated M3T owner arrays enter begin. Independently re-admit all
parameters/values, exact identities/shapes/uniqueness, and unborrowed/unpinned
value, gradient and parameter controls before acquiring. Each acquisition sets
parameter `plan_pins` from0 to1 and value `active_borrow` to
`(et_f32_tensor_borrow *)(void *)&pins->views[i]` as the exact stable sentinel. This reserves a control count; it creates **no gradient plan**, gradient
view, contribution or numerical mutation. A value borrow alone would not exclude
existing gradient reset/plan preparation, which checks the parameter counter.
The binding decision clarifies that parameter `plan_pins` is not a universal
gradient-access lock: current privileged `gradient_borrow_begin` admission does
not test that count. Preservation also requires the closed role inventory, no
gradient-view/plan calls, serialized execution and no callback/reentrancy. This
clarification does not strengthen or change I2's existing locking contract.
Partial failure reverses only held bits without allocation or error clobbering.
Check validates self, registries, identities, sentinels and exact count1. End,
after successful preflight, clears sentinels and decrements1 to0 in reverse order;
no recoverable branch occurs in that tail. Clear views/owners/self only after
all held sentinels and control counts have been released. No ordinary retained borrow shell is
allocated. No view escapes the context or survives the outer invocation.

Hold all14 across the closed Eshkol role transcript, including prefill plus the
one candidate-token frame during generate. Roles reuse those views rather than
reborrow. Snapshot identities and all4,736 value bytes into preallocated storage;
committed binding also covers topology/profile/eval and exact tokenizer identity.
Decode compares committed bits while pins are held. Exact restoration is valid;
a fresh prefill can replace the binding. A mode change is not stopped by I2 pins:
serialized no-callback execution plus explicit Eshkol mode/profile recheck before
each prepare is required. Saved graphs, gradients and contribution counts remain
unchanged on success and failure.

## Closed native and Eshkol call inventory

All native names below have prefix `et_g3t_private_` and suffix `_v1`; `ptr`
means `void *`, `i64` means `int64_t`. Status returns0 on success and a nonzero
category on failure; pointer constructors returnNULL on failure. No symbol is
installed/global in the finished archive. Bytevector headers below are synchronous
trusted copies, never exported or retained. Every function re-authenticates its
closed owner arguments; there is no shape, parameter-index, provider or operation
string argument.

| Return and stem(arguments) | Exact authority/effect |
|---|---|
| `ptr generator_seed(ptr model_owner, i64 seed, i64 mode, i64 temperature_bits, i64 k, i64 p_bits, i64 max_new, i64 eos)` | authenticated native M3T owner; new idle context with seeded RNG |
| `ptr generator_rng(ptr model_owner, ptr rng, i64 mode, i64 temperature_bits, i64 k, i64 p_bits, i64 max_new, i64 eos)` | identical construction; deep-copy kind8 RNG, retain no source owner |
| `ptr input_from_t1(ptr sealed_t1)` | Eshkol T1 registry authentication first; native repeats sealed/rank1/P1or2/byte-ID checks; copies to new I1[1,P] |
| `ptr input_from_token(i64 id)` | exact byte-ID to new I1[1,1] |
| `i64 tensor_release(ptr owner)` | only kinds2/3/5/6/7 |
| `ptr logits_reserve(ptr ctx)` | exact pending manual result attached to this call |
| `ptr output_reserve(ptr ctx, i64 P)` | exact pending generate result G=budget; all I1/RNG capacity before prefill |
| `ptr output_ids_clone(ptr out)` | detached kind5 I1[G] |
| `ptr output_lengths_clone(ptr out)` | detached kind6 I1[1] |
| `ptr output_cache_lengths_clone(ptr out)` | detached kind7 I1[1] |
| `ptr output_rng_clone(ptr out)` | detached kind8 immutable logical i64[4] |
| `i64 output_release(ptr out)` / `i64 rng_release(ptr rng)` / `i64 generator_close(ptr ctx)` | corresponding typed release only |
| `i64 call_acquire(ptr ctx, i64 call_kind)` | 0manual-prefill,1manual-decode,2generate; shared guard and14pins |
| `i64 frame_begin(ptr ctx, ptr input, i64 frame_kind)` | 1prefill or2decode, subject to exact call state; own copy of IDs before persistence |
| `i64 role_step(ptr ctx, i64 ordinal)` | exactly next ordinal0..20, advance only on success |
| `i64 sample(ptr ctx)` | only generate/G1 after committed prefill; stored policy selects S; no RNG/cache publication |
| `i64 frame_prepare(ptr ctx, ptr staged_result)` | completed transcript; copy last logits into attached pending manual result before marking prepared; otherwise text-ready output; mode/binding/cleanup preflight |
| `i64 frame_commit(ptr ctx)` | sole prepared cache/result/RNG publication tail |
| `i64 call_prepare_end(ptr ctx)` | validate pin/sentinel/view/owner cleanup before a final commit |
| `i64 call_finish(ptr ctx)` / `i64 call_abort(ptr ctx)` | exactly-once prepared postcommit drain / precommit failure rollback preserving committed prefix/RNG |
| `i64 output_prepare(ptr ctx, ptr out)` | fill reserved G0 output from current RNG or G1 output from candidate/successor; no publication |
| `i64 output_copy_decode_ids(ptr ctx, ptr out, ptr staging_header)` | pending linked output; copy prepared IDs to exact8G little-endian bytes |
| `i64 output_accept_text(ptr ctx, ptr out, ptr raw_header)` | pending linked output; exactG bytes equal preparedIDs, mark text-ready |
| `i64 last_error_domain()` / `i64 last_error_category()` / `i64 last_error_code()` | scalar-only frozen error snapshot source |

For generate decode, `frame_begin` requires inputNULL and consumes only its
private candidate scratch; manual decode requires authentic kind2 input and
never acceptsNULL. Prefill always requires kind2. The sampler copies its token
I1[1] into separate owned I1[1,1] decode scratch; it does not relabel metadata.
All frame scratch has its actual T=1 orT=2 shape. Generated and manual logits
use explicit last-position bit copies into owned f32[1,256].

There are **29** native boundary functions in this table plus the three C-only
pin functions. Constructors repeat numeric checks: mode0greedy/1categorical,
bit integers0..2^32-1, k1..256, budget0/1, eos-1(noEOS)or0..255, and accepted
seed/RNG rules. C2 profile is implicit; Eshkol first authenticates model/tokenizer
and normalizes the exact eight-pair config. No raw tokenizer address enters C.

Each native boundary is declared only under the corresponding private Eshkol
extern name `g3t-native-<stem>` (underscores become hyphens), with the table's
fixed arity. Exactly12 shell-authenticated owner operations have prefix `g3t-`:
generator-create(3), generation-input-create(1), generation-token-input-create(1),
generation-tensor-release!(1), generation-output-release!(1), generator-close!(1),
generation-rng-release!(1), generation-output-ids(1),
generation-output-lengths(1), generation-output-text(1), generation-output-rng(1),
generation-output-cache-lengths(1). They authenticate Eshkol identities before
selecting only their closed constructor/accessor/release calls.

Exactly12 additional transport wrappers are `g3t-call-acquire`(2),
`g3t-frame-begin`(3), `g3t-role-step`(2), `g3t-sample`(1),
`g3t-frame-prepare`(2), `g3t-frame-commit`(1), `g3t-call-prepare-end`(1),
`g3t-call-finish`(1), `g3t-call-abort`(1), `g3t-logits-reserve`(1),
`g3t-output-reserve`(2), `g3t-output-prepare`(2). `g3t-call-acquire` accepts the generator entry and returns the rooted private
call entry. Subsequent ctx positions require that exact call entry, whose
generator/model links and both active cells must match; translation to native
ctx is internal. Result operands require exact linked pending entries. Only
generate's private candidate input uses #f→NULL.
Reservations root pending envelopes before native allocation; wrappers maintain
the shared guard/ledger and exact state transitions, not a numerical schedule.
A private lexical `g3t-with-call` macro enters the existing shared `m3-call`
boundary around trusted step forms; it accepts no runtime callback. Every
failure after acquire is unwound through the admitted context before reraising.

Additional named bindings are `g3t-registry`, the38guard wrappers, and the two
decoder helpers below; other Eshkol helpers remain lexically local. There is no
private prefill/decode/generate operation wrapper, full21-role production loop,
or sampling-plus-token loop in G3-T. G3-M/G3-G own that later orchestration; the
development witness calls explicit steps. The macro provides lexical cleanup
only and supplies neither the21-role transcript nor a generation loop. No new P1 slot or boxed ABI wrapper
exposes these definitions. Additional named seams require review and an explicit
local-symbol manifest update. This inventory has95 named Eshkol bindings:
29 extern aliases,12 owner wrappers,12 step wrappers,38 guarded adapters,
two decoder helpers, one registry cell and one macro; it does not predict ELF
symbol counts.

## Fixed role and provider routing

Eshkol owns the numerical schedule: exactly21 ordered calls, never a C full-model
loop or recursive scalar numerical path. Scratch slots0..20 are respectively
ET, EP, X, N1, QT, KT, VT, QH, KH, VH, AH, AT, AO, R, N2, FU, FG, FD, Y, NF, Z.
ET/EP/X/N1/QT/KT/VT/AT/AO/R/N2/FD/Y/NF are[1,T,4]; FU/FG are[1,T,8];
QH/KH/VH/AH are[1,2,T,2]; Z is[1,T,256]. Epsilon bits are0x3727c5ac.

| Ordinal | Fixed operation | T1 route | T2 route |
|---|---|---|---|
| 0,1 | token, position embedding | G3-N embedding | N3K embedding-forward |
| 2,13,18 | embedding, attention, FFN residual | G3-N residual | N3K residual |
| 3,14,19 | block-input, FFN-input, final norm | G3-N layer-norm | N2 kernel.norm |
| 4,5,6,12,15,17,20 | Q,K,V,attention-out,FFN-up,FFN-down,tied-head linear | G3-N linear | N3K linear |
| 7,8,9 / 11 | Q,K,V split / attention merge | G3-N head-layout | N3K head-layout |
| 10 | cached causal attention | A2 [1,2,2,1,2,2] | A2 [1,2,2,2,2,2] |
| 16 | exact-erf GELU | N2 kernel.activation/gelu.forward [1,1,8] | N3K gelu [1,2,8] |

Exact G3-N capability/operation strings and11rows are inherited from its accepted
proposal; exact T2 strings/operand ordering are the forward rows of M3T's fixed
table, with cached attention's explicit mask/positions below. Use separate fixed
runtimes for N2,N3K,A2,G3-N,G3-S; no canonical resolver or provider fallback.
Sampling routes only to accepted `g3s.greedy`/`g3s.greedy.forward` or
`g3s.categorical`/`g3s.categorical.forward`, request[1,256], explicit scalar
views and distinct token/successor buffers. Fail exact-row/provider admission;
do not cast, silently substitute, or infer new backward capabilities.

At attention ordinal10, stage final learned-position projected K/V into the
pending A2 transaction, open full-capacity views, materialize bool[1,T,2] using
validity and causal positions, invoke A2, then end the nested view before return.
A2 owns distinct K/V[1,1,2,2,2], lengths[1], keep[1,2]; stage[1,2,T,2] and
consume full[1,2,2,2]. Query positions are[1,T], keys[1,2]={0,1}. Unused tails
remain finite zero; no short capacity-strided view or implicit mask broadcast.
No RoPE is applied in this accepted learned-position consumer.

Uncached parity uses an **instrumented-only** frame kind0/call kind3; normal
admission rejects them. Its T1 attention route is the accepted G3-N
[1,2,2,1,1,2], T2 is A2[1,2,2,2,2,2]. Test-only entry points/failpoints are
absent from normal inventories. This is a reference execution path for later
G3-M parity proof, never fallback or an installed generation API.

## Publication, text and release protocol

Before initial prefill: validate full request/context and draw-dependent
exhaustion, acquire14pins, allocate rooted shell/entry/list/empty-ID/raw-text/RNG
and cleanup capacity, reserve result, and prepare a candidate cache. Eshkol checks prompt metadata
and context bounds before acquire; native reservation repeats P=1/2, budget
and P+G<=2 checks, and frame_begin must match the reserved P. For G1 the
exact final output length is already1; for G0 it is0. Ordinary constructors also
preallocate Eshkol envelopes before a complete native owner is attached.

Prefill failure preserves old cache/binding/RNG. Its commit replaces cache and
binding after old-cache destruction is preflighted. Manual prefill/decode seals
pending logits; generate/G0 seals empty output; generate/G1 commits only prompt
cache and leaves output pending. G1 sampling stages a candidate/successor, then
runs its full token frame before the output becomes visible. A failure here
aborts/scrubs the append, destroys pending result, retains prompt cache and old
RNG, and raises with no partial output. EOS uses the same full token commit.

`output_prepare` copies final candidate/lengths/RNG into the pending output;
text is then decoded and accepted, before `frame_prepare` checks complete
numerical/cache/text readiness for the final commit. G0 preparation uses current
RNG after reservation; G1 requires sample-ready candidate/successor and cannot
prepare during initial prefill. Prepare, decode-ID copy and accept-text each
advance once-only readiness flags; repeated/out-of-order attempts reject with
invalid-state before writes. Manual result copying likewise precedes prepared=true.
Source-private `(g3t-t1-decode-output! call-entry output-entry)` arity2 validates
exact pending linkage and retained tokenizer, calls `output_copy_decode_ids`,
then `(t1-private-g3-decode-raw-into! tokenizer staging raw)` arity3. The latter
authenticates the actual T1 registry/core: raw mode, empty special/prefix/suffix
collections, V256; requires staging length0/8 and raw lengthG, validates every
ID0..255 **before** its first write, and fills the preallocated bytes. It allocates
no bytevector/list/hash/decode scratch, retains no borrow and enrolls no T1 shell.
Public tokenizer-decode and its allocating semantic decoder remain unchanged.
`output_copy_decode_ids` requires numeric_ready; `output_accept_text` requires
numeric_ready and ids_copied, verifies those already-decoded bytes and sets
text_ready. Final `frame_prepare` requires all three readiness flags.

Before final commit, `call_prepare_end` validates every live pin, no nested A2
view, exact staged transaction/results and all destruction preconditions. The
commit tail publishes native cache/ID/RNG/output, then its trusted Eshkol caller
immediately invokes `call_finish` exactly once and seals rooted Eshkol entries.
`frame_commit` does not itself invoke `call_finish`. No allocation, callback or
recoverable branch may intervene. On its valid prepared postcommit path, finish
returns0 only, drains pins/active tokens and restores idle; invariant defects
fail-stop. A second finish or precommit misuse rejects before mutation.
Non-final G1 prefill uses frame-local cleanup preflight and retains pins. Invalid
unprepared calls reject before writes; impossible tail invariant defects use
M3's fail-stop precedent, never a recoverable error after partial publication.

Abort snapshots error first, ends nested views, aborts candidate append or destroys
uncommitted candidate cache, releases pending typed results, drains pins and
clears native active then Eshkol model10/generator9/call-state. It does not rewind
a successful initial prefill. Cleanup has preallocated ledgers and no allocator.
Released exact owners return success idempotently; all other use is invalid-state.
Wrong kind rejects before reading lifecycle. Busy/borrowed release rejects before
mutation. Tensor/output/RNG release and close clear payload roots but retain
identity tombstones. Accessors allocate independent clones; output-text returns
a fresh one-element list and detached bytevector, surviving all parent releases.

E3's mode/cursor restoration and staged-result publication belong only to its
consumer. Shared pin-end clears pins only, never native/model/outer exclusion or
consumer state. G3's finish/abort still own matching token cleanup; only the
outer `m3-call` clears its guard after the inner consumer handler completes.
G3 stays eval-only, with no shared mode flag or restoration callback.

## Bounded error transport

Do not reuse M3T's mapper for new G3T codes. Native constructors preserve first
failure through partial cleanup; Eshkol snapshots domain/category/code before
any later cleanup call. Domain0 is G3T;1 is unchanged K1 (including A2 cache);
2 is I1;3 is I2. Success is category/code0. Error category/code ranges are K1
1..8/1..16, I1 and I2 each1..6/1..11, interpreted by their existing headers.
Malformed triples map to internal/invariant, never unchecked table access.

G3T categories1..6 are invalid-argument, invalid-state, shape-mismatch,
unsupported, internal, determinism-unavailable. Codes1..13 are identity,
lifecycle/busy/borrow, selector/config, shape/token-range, alias, topology,
allocation, capability, stale-binding, invariant, FP-control, required-draw
exhaustion, readiness. Required-draw exhaustion maps to category2/code12;
explicit FP admission failure to6/11. Never infer determinism-unavailable from
an arbitrary K1 unsupported/provider rejection or invent a K1 invalid-state enum.
Other dtype/device/layout errors retain their K1 E1 mapping. T1 native statuses
1/2/3/4/5 map to G3T(category,code)=(1,1)/(2,2)/(3,4)/(5,10)/(5,7);
unexpected statuses map to5/10. Eshkol T1 semantic preflight errors use these
same bounded meanings without retaining an exception cause.

Details are only source-domain/source-code/source-message/native-category;
source-message is selected from fixed safe strings, not pointers or copied
arbitrary native memory. Raise E1 with the invoking private operation now/public
operation in G3-G and cause#f. Authentic exhausted RNG remains accepted for
creation/greedy/manual/G0; only required categorical G1 rejects before prefill.

## Source composition, inventories and required proof

Proposed sibling tuple is `native/g3t_package_{root.esk,bridge.c,private_renames.txt,public_exports.txt}`;
recipe/policy/native-selection files are `scripts/{build-g3t,g3t-package-policy,g3t-native-inputs}.sh`.
One archive member `g3t_package.o`; installed seven facades/87 boxed exports/93
globals match M3 exactly. All G3T/G3-N/G3-S/A2-cache symbols are LOCAL; reject
unresolved private references. Later G3-G's fifteen names would yield eight
facades/102 boxed exports/108 globals, requiring its own reviewed tuple.

Under the accepted common-source contract, the root loads the M3 root, then
`native/m3_call_adapters.esk` once, then `native/g3t_transport_extension.esk`.
The common source owns the existing 38 checked bindings; `m3-call-state` and
`m3-call` stay canonical in `native/m3_model_extension.esk`. T1's new
raw helper is trusted internal source only. Proposed transport TU
`src/eshkol_transformer/g3t_transport.c` includes `m3_model.c` and replaces its
compile entry once. The accepted common contract replaces the formerly proposed `g3t_f32_integration.c`
with `src/eshkol_transformer/m3_call_f32_integration.c`, including the existing
`m3t_f32_integration.c` once and adding only the fixed14 pin mechanics; it
replaces that compile entry. Neither old proposed G3 pin path becomes a shim. Keep `m3_i64_integration.c` and `t1_i64_shell.c` once; no new
registry copy. Private declarations live in `g3t_transport.h` and the common
`m3_call_pins.h` (superseding proposed `g3t_model_pins.h`), never installed.
Actual `et_g3t_*` spellings and all three signatures remain unchanged. Shared
sources import no G3 registry, cache, decoder, sampler or provider runtime. Actual depfiles must include both new and
included predecessor sources. No whole-archive provider admission is allowed.

The exact selected native object tuple is:

```text
n2/n2_primitives_provider.o
n3k/n3k_primitives_provider.o
a2/a2_attention_provider.o
a2/a2_kv_cache.o
g3n/g3n_primitives_provider.o
g3s/g3s_sampling_provider.o
```

M3 currently selects only the first three provider objects; it does **not**
include the cache object. A bounded future A2 recipe change must emit
`a2_kv_cache.d` with `-MMD/-MF/-MT`; source closure explicitly gains
`native/a2_kv_cache.c` and `include/eshkol_transformer/a2_kv_cache.h`, plus N/S
sources/headers. This does not change A2 ABI/capability semantics.

Recognize the exact G3T tuple before predecessor M3/M3T detection, suppressing
predecessor checks only after authentic G3T tuple admission. Preserve canonical
lexical paths, per-component symlink rejection including C `../` spelling,
unfiltered source/native depfile validation, scrubbed compiler environment and
ordered include roots `internal/p1/lib`, `internal/c1/lib`, `internal/t1/lib`,
`src`. Preserve exact public strings, inherited bridge,46rename rows and facade
closures; freeze actual local symbols/undefined dependencies from the supported
implementation build, not guessed compiler output in this proposal.

Future proof has two separate lanes: normal exact archive plus installed
inherited public M3 smoke/error-identity and private-link/import negatives; and
an explicitly development-only source-composed private Eshkol witness with its
own checked closure, no competing aggregate. Cover every role/shape/ordinal,
cache versus uncached parity, both lengths and all G0/G1/manual paths, same-model
and aggregate reentrancy, mutated/restored parameters/mode, gradient reset/plan
exclusion, copied/dead/wrong-kind identities, active-borrow releases, accessor
survival and zero T1 decode-registry growth. Inject all allocation/pin/view and
cleanup-preflight failures; prove prefix/RNG/gradient/count preservation and
infallible publication tails. Require ASan/UBSan/LSan, supported exact-head CI,
private AOT, and measured1,024/8,192-loop live and cumulative retention. Tombstones
are cumulative; no flat-memory or unmeasured execution claim is made.

## Review and execution record

Independent guard/pinning review verified29 native calls plus3 pin helpers and
required the stable sentinel address and explicit last-logits copy before prepare.
Owner/T1 review required once-only output readiness and an unambiguous exactly-once
postcommit finish. Packaging and guard reviewers both rejected premature private
generation orchestration; it is replaced with single-step transport bindings.
Those corrections are incorporated. Final owner/T1 and packaging cross-reviews
pass for contract submission with no remaining corrections; the guard review
found no other blocker after its specified corrections. Root subsequently
accepted all four, following two separate Astra/high
guard/pinning and package/error reviews and a root ownership/decoder/A2
transaction audit. These source/design reviews provide no runtime evidence.

At reviewed proposal `a7a4590`, documentation checks passed:29 unique native
seams,46rename rows split
38M3T/8M3,4,736 parameter bytes,55 local links across the five changed Markdown
files, accepted G3/N/S/request files byte-identical to HEAD, and `git diff --check`.
Exactly five Markdown files changed in that proposal. The acceptance follow-up
changes four Markdown files and verifies the reviewed seam body is byte-identical
after removing only the binding decision's gradient-lock clarification. Local
links and whitespace pass; accepted G3/N/S contracts remain unchanged. These are
documentation checks; all implementation proof above remains future work. No production file, header, ABI, full build,
full CI, implementation PR or merge is added by this proposal.
