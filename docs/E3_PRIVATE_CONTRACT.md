# E3-DESIGN: exact private evaluation contract proposal

**Proposed, not implemented or frozen.** This package concretizes the accepted
[private-first architecture](E3_EVALUATION_PROPOSAL.md) at main
`27c99f7c9f27a227f0e237f1571ac89d5cdfa225`. Root acceptance and prerequisite merges
must precede implementation. It adds no public evaluator, VERIFIED evidence, A0
schema change or completion claim. The E3 task owns E3-DESIGN, later E3-PRIVATE
orchestration/integration, and independent E3-PRIVATE-R; root owns acceptance.

The following documents form one review unit:

| Contract | Exact responsibility |
|---|---|
| This document | Frame/source identities, storage, native and Eshkol calls, ordered transaction, publication, package generation and gates |
| [BOOL metrics](e3/E3_BOOL_METRICS_CONTRACT.md) | ABI1.0 descriptor/three rows, domains, rounding, error precedence, fenv/errno and independent numerical gate |
| [P1 modes](e3/E3_P1_MODES_CONTRACT.md) | Canonical lexical slots64..68, status0..4, nine-slot authority record, all17 modes and allocation-free tails |
| [Errors](e3/E3_ERROR_LIFETIME_CONTRACT.md) | Root-built156-cell E1 table, same-region classification, first-error fields and raise after guard exit |
| [D2](e3/E3_D2_CONTRACT.md) | Fixed source variant, T1 identity, unary native idle status0..2, fourteen-slot restore record and three fixed staging copies |

The separately owned issue101 shared contract supplies exactly
`native/m3_call_adapters.esk`, `src/eshkol_transformer/m3_call_pins.h` and
`src/eshkol_transformer/m3_call_f32_integration.c`. Consume its accepted actual
`et_g3t_model_pins_internal` and begin/check/end declarations, without aliases or
G3 imports. Until its exact contract merges, that source dependency is pending;
this proposal does not create or redefine it.

## 1. Fixed scope and concrete source files

Only M3 N1/T2/V256/D4/Hq=Hkv2/Dh2/L1/F8 CPU f32, learned position, LayerNorm
0x3727c5ac, exact-erf GELU and tied token/head are admitted. Canonical raw byte T1
with empty specials/prefix/suffix; D2 unshuffled seed#f/window1, packed or unpacked,
current cursor through EOS; maximum8192 batches/16384 active tokens. Exceeding the
bound rejects the whole call; no truncation. Entry EOS or zero total weight is
invalid-state. No dropout, RNG advance, graph, backward, cache or gradient action.

Proposed new files (these names are contract choices, not existing APIs):

| Source | Purpose |
|---|---|
| `native/e3_private_root.esk` | One ordered source aggregate |
| `native/e3_private_extension.esk` | Frame/destination ledger, admission, first-error slots, closed transaction |
| `native/e3_forward_schedule.esk` | Literal21 forward calls followed by four metric calls |
| `native/e3_errors_extension.esk` | Root initialization of the156 immutable errors and fixed classifier |
| `native/e3_p1_modes_extension.esk` | Five exact P1 slot wrappers |
| `internal/e3/lib/e3_d2_identity.esk` | Fixed authenticated T1 byte identity |
| `internal/e3/lib/e3_d2_restore.esk` | D2 record, restore helpers and three fixed consumers |
| `src/eshkol_transformer/e3_frame_internal.h` | Opaque frame declaration and exact private native prototypes below |
| `src/eshkol_transformer/e3_frame.c` | Includes `m3_model.c` once; E3 registry, embedded pins, forward storage, fixed dispatch, commit |
| `native/e3_d2_native.{c,h}` | Single D2 include-based replacement and idle preflight |
| `native/e3_private_bridge.c` | Includes existing M3 boxed bridge once; one private test-driver wrapper |
| `tests/e3_private/driver.esk` | Closed test entry/observers; absent from installed artifacts |

Numeric header/source/scripts and deterministic D2 source generator paths are
fixed in their companion documents. Only canonical P1 module source changes in
place: append slots64..68 and lexical helpers, preserving first64/public provide.
This affects all P1 inheritors' private symbols/fixed allocation. Their manifests
and retention tests require explicit P1/root acceptance; no predecessor binary
identity is promised. Existing D2 sources and its T2 tuple remain byte-identical.

## 2. Source identities and lifetime

`e3-frame-bind!/2(model,destinations)` is a source-private setup operation under
`m3-call 'e3-frame-bind!`. `destinations` is an exact four-slot vector of native
`et_f32_tensor *` values created by the private C test fixture with existing I2
creation APIs, ordered loss/weight/perplexity/accuracy. It is not an ordinary
number vector, I2 carrier clone or generic public tensor wrapper. Only this trusted
fixture boundary supplies these pointers; native admission independently checks
all four against the real I2 registry before dereference. Require distinct live
rank0 CPU/f32 owners, exact4-byte spans, unborrowed/unpinned, no alias with each
other, model parameter/value/gradient storage, frame storage or control records.
Caller owns them through frame unbind; E3 never destroys them. Four pointers are
bound once, never replaced. The source ledger retains the exact destinations
vector so a copied vector cannot be substituted on subsequent use.

Source model admission uses existing `m3t-entry model 'model operation #t`;
its native OWNER is entry slot3, canonical handles slot8, active slot10 and profile
slot11. E3 stores that exact entry in source slot3; native frame_create receives
its slot3 native pointer, never the boxed shell or an arbitrary source vector.

The returned inert Eshkol frame token is an exact identity in
`e3-frame-registry-root=(vector '())`. No token payload authenticates itself.
Lookup exact `eq?` token first, then read its private entry. Setup roots and
canonicalizes all source objects through the existing root write-barrier pattern
before retaining any identity natively or acquiring model tokens. Native frame
stores no raw P1/D2 vector pointer. No frame move/realloc while live.

Exact source entry (vector length16):

| Slot | Type and meaning |
|---:|---|
|0|exact inert frame token|
|1|source phase i64:0 staging,1 idle,2 active,3 restoring,4 dead|
|2|authentic M3 model shell, #f after unbind|
|3|exact M3 source model entry, #f after unbind|
|4|native frame pointer, null after unbind|
|5|exact destination vector, #f after unbind|
|6|P1 opaque mode token, #f after unbind|
|7|canonical D2 restore record14, #f after unbind|
|8|P1 setup box2 `[frame-token,#f]`, #f after bind completes|
|9|reserved#f; batch authority lives only in its region-local cell|
|10|acquisition bitset immediate i64, initially0|
|11|first mapped root-lifetime error or #f|
|12|fixed diagnostic vector4 `[stage,domain,category,code]`, initially zeros|
|13|traversal-started boolean, initially#f|
|14|published-valid boolean, initially#f; set#t after native publish, preserved on later failure|
|15|first-error index i64, -1 on entry;0..155 once set, frozen through cleanup|

Bits0..4 mean native acquisition/model source token/P1 prepared/P1 entered/D2
prepared; bit5 means owned batch. The existing `m3-call` alone owns the outer
boolean; E3 neither adds another guard nor manually clears it. Native acquisition
contains all14 pins or none. Source bits publish after successful steps, with no
callback/fallible allocation between corresponding native/source token stores.

D2 record/root lifetime must not retain a young batch through the process-root
ledger. A call-local preallocated control cell in the batch's owning region,
referenced lexically by its handler, retains the owned batch. Slot9 remains#f
in the root entry; bit5 tracks ownership only. Likewise transient error objects
are never written into root slots. This distinction is part of the contract, not
permission to depend on automatic per-batch graph promotion.

P1 mode token/arrays and D2 restore record allocate once per frame. D2 datasets
used in this private artifact are constructed in the true package/root arena by
the trusted fixture before any invocation/batch region opens. Shell/state/core
identities must already be root-stable at D2 native enrollment. Merely outliving
the invocation is insufficient: storing an enclosing-region shell into a root
record could promote it and break native identity. No younger dataset is admitted
or promoted by this private route; poison/idle tests prove identical native owner
identity before/after prepare. The C runner enters from a fresh root context;
this is a test-fixture lifetime precondition, not a new general D2 open API. Clear all
D2 binding slots on every exit. Source frame records and native frame headers
remain inert tombstones after release, preventing address/identity reuse. They
retain no model/dataset/tensor/runtime references. Setup history is not claimed
flat; repeated calls on one frame allocate no new frame or mode tombstones.

`e3-frame-d2-record-internal` first exact-eq authenticates source token/entry,
then calls fixed native frame_check before returning canonical entry7. This
supplies its required native identity check without a generic carrier/selector or
assuming a copied native pointer authenticates itself. Source phase1→2 marks the
admitted invocation before prepare; phase3 marks restoration; all exits return1.
Source model-slot10 equality is required once acquisition bit1 is set; before that
it must be idle. Native phases are checked independently as specified below.

Source-private named call inventory owned by this package (no installed facade):
`e3-frame-bind!/2`, `e3-evaluate-into!/3(model,dataset,frame)`,
`e3-stage-batch!/2(frame,batch)`, `e3-forward-schedule-internal/1`,
`e3-forward-role!/2(frame,role)`, `e3-ce!/1`, `e3-reduce!/1`, `e3-correct!/1`,
`e3-accumulate!/1`, `e3-finalize!/1`, `e3-cleanup-preflight!/1`, `e3-abort!/1`,
`e3-publish!/1`, `e3-counter-ref/2(frame,selector)`, `e3-frame-unbind!/1`,
and `e3-frame-d2-record-internal/1`. Numeric selectors are closed role0..20 and
counter0..1 only. The abort wrapper composes native preflight/drain, D2/P1 restore
and native finish; it is not a second native cleanup authority. Additional lexical
helpers are implementation-local and inventoried in generated private symbols.
P1/D2 exact wrapper arities/statuses are in the companions. Bind returns canonical
frame token; evaluate returns#t on success and raises mapped E1 after cleanup on
failure; unbind returns#t (dead exact token idempotent), counter returns exact i64.
Unbind authenticates idle token and native destruction preflight, then P1 unbind,
native nonfailing destroy, source root clearing/dead phase. No partially unbound
live frame is returned as a recoverable error.

Exact combined bind sequence and unwind:

1. The private fixture performs setup in true root context before invocation
   regions. Enter shared bind guard; authenticate complete idle M3 model and all
   four genuine destination owners. Allocate the inert source token, entry16,
   D2 record14/snapshot15, P1 setup box and ledger cell; no mode/cursor/output or
   native active token changes. Publish source entry as STAGING only after the
   entire source graph is valid/root-stable; canonicalize/read back identity.
2. Call P1 slot64 with that exact model/setup box. Its complete mode token/nodes/
   saved arrays/ledger graph is allocated before publication. Root context means
   no young shell needs promotion; P1 publishes its ledger then writes its canonical
   rooted token into the rooted setup box using nonallocating stores. There is no
   recoverable publication gap hiding an enrolled token from source cleanup.
3. Native frame_create independently re-admits owner and destinations. Allocate a
   private unregistered candidate,37 genuine I2 scratch tensors and six runtimes;
   initialize metadata/positions/keep/epsilon and all named fields. Never acquire
   pins/model.active or mutate caller destinations during setup. Enroll native
   candidate only after every allocation/admission/initialization succeeds, then
   return pointer; the candidate never escapes earlier.
4. On native setup failure, save its first diagnostic triple, destroy only already
   created scratch tensors/runtimes in reverse order, free its unregistered
   candidate, then restore diagnostic triple and returnNULL. This cleanup uses
   existing nofail-after-preflight owner destruction and terminates on defects;
   it cannot overwrite the reported first failure. I2's retained control shells,
   if any, remain measured predecessor costs, not a claimed allocation rollback.
5. If any step after source staging fails, unbind any successfully published P1
   token, clear source model/destination/D2 references, mark source entry DEAD and
   retain only its inert identity tombstone. No native frame was published on a
   NULL result. If the native constructor succeeded, remaining source writes are
   pointer/phase stores into already-rooted slots: store pointer, mode token,
   clear setup-box reference, mark IDLE last, return original canonical token.
   No allocating boxing/provider/callback exists after native enrollment.
6. Failure before source ledger publication leaves no source authority; partial
   setup allocations are accounted separately. Each later bind is a new explicit
   setup attempt, never implicit retry/replacement of a half-live frame. Dead
   tokens reject evaluation and support idempotent unbind only. The return and
   error paths require pinned AOT allocation/lifetime proofs; declaration review
   alone does not prove the noalloc tail.

## 3. Native layout, authentication and storage

`e3_frame_internal.h` forward declares `typedef struct et_e3_frame_internal
et_e3_frame_internal;`, includes stdint, K1 and I2 headers and extern-C guards.
The complete struct lives only in `e3_frame.c` after including `m3_model.c` and
`m3_call_pins.h`; it is neither serialized nor a public ABI. Exact named fields:

```c
struct et_e3_frame_internal {
  struct et_e3_frame_internal *next;
  uint32_t lifecycle, phase, next_role, stage_plane;
  owner *model;
  et_g3t_model_pins_internal pins;
  et_f32_tensor *forward[21], *ce, *epsilon;
  et_f32_tensor *batch[4], *sum[2][3], *result[4], *destination[4];
  int64_t ids[2], targets[2], positions[2];
  unsigned char keep[4], mask[2];
  int64_t counts[2][2], active, staged_counts[2], published_counts[2];
  uint32_t sum_bank, committed, cleanup_ready, published_valid;
  et_kernel_runtime *runtime[6];
};
```

`owner` is the included M3T private type, never header-visible.
`next` participates in a dedicated E3 registry, not a new M3T record kind. Every
entry scans this registry by address before dereferencing, requires lifecycle1
live (2 dead), then independently calls existing M3T OWNER admission. No arbitrary
self/tag/embedded-pin pointer is accepted. During an acquired call require exact
`model->active==frame` and source model slot10==its authentic source frame entry.
Before acquisition both must be idle. Parameters/identities derive only from
`model->p`/`model->handles`. Begin pins, then publish owner.active; end clears pins
only, retaining active until restoration and any output commit complete.

There are37 genuine owned I2 f32 scratch tensors:21 forward, CE[1,2], epsilon[],
4 batch scalars `[N_b,W_b,mean,C_b]`,6 ping-pong sum scalars `[N,W,C]` per bank,
and4 result scalars. Shapes match section5 and metric schemas exactly. Total
float payload2820 bytes:2752 forward+8 CE+4 epsilon+16 batch+24 sums+16 results.
Four caller outputs add16 bytes but are not frame-owned. `positions={0,1}`,
`keep={1,1,1,1}`, epsilon bits0x3727c5ac are initialized at setup, immutable per
call. IDs/targets/positions and masks are explicitly private fixed native typed
storage (48+6 bytes), not fabricated I1 owners or a public carrier API. Native
counter arrays/cells total72 bytes. C padding/control/shape/runtime memory is
measured separately; payload arithmetic is not a total-retention measurement.

All37 tensors and six explicit provider runtimes are created once at bind and
released at idle unbind; runtime order N3K,N2,A2-attention,L2,L3S,E3-metrics.
No runtime resolve/allocator is permitted after mutation begins. Each role uses
stack `et_f32_scoped_guard_internal` for its scratch operands, validated and ended
synchronously; held model parameter inputs use pin views directly. No saved14
parameter copies, reverse tensors, ordinary per-call I1/I2 borrow shells or copy
plans. Native typed i64/bool views have stable internal shape arrays and fixed
read-only metadata. Destinations are accessed only by the final native helper.

## 4. Exact native calls, statuses and readiness

All declarations below have private C linkage and v1 suffixes; `F` denotes
`void *frame` only in this table, not an actual macro/API. Return `int64_t` status
unless stated otherwise. No call accepts a provider, parameter tensor selector,
owner-kind, pin pointer, callback or output pointer chosen after binding.

| Exact symbol suffix after `et_e3_private_` | Signature/meaning |
|---|---|
|`frame_create_v1`|`void *(void *model, et_f32_tensor *loss, et_f32_tensor *weight, et_f32_tensor *ppl, et_f32_tensor *accuracy)`; NULL on setup failure|
|`frame_check_v1`|`int64_t(F)`; read-only registry/live/OWNER admission; phase0 requires native model idle, phases1..8 require owner.active==frame; no pin acquisition|
|`acquire_v1`|`int64_t(F)`; idle→acquired, zero private sums/counters/readiness, pins then model token|
|`stage_inputs_v1`, `stage_targets_v1`, `stage_mask_v1`|`int64_t(F,const et_kernel_tensor_view_v1 *view)`; exact sequential copies16/16/2 bytes|
|`forward_role_v1`|`int64_t(F,int64_t role)`; only exact next0..20; one fixed role dispatch|
|`ce_v1`, `reduce_v1`, `correct_v1`, `accumulate_v1`|`int64_t(F)`; one fixed dispatch each in this order|
|`finalize_v1`|`int64_t(F)`; nonempty totals→four separate result tensors and staged counters|
|`cleanup_preflight_v1`|`int64_t(F)`; check owned views absent, owner/pins/scratch/result readiness; no drain or output write|
|`drain_v1`|`void(F)`; cleanup-ready only, shared pin-end; leaves model.active and final staging intact|
|`publish_v1`|`int64_t(F)`; drained/finalized and Eshkol restore complete; six-output commit below|
|`finish_v1`|`void(F)`; matched native token clear, private readiness reset, idle; success and abort both use it|
|`counter_ref_v1`|`int64_t(F,int64_t selector)`; idle/published_valid only;0 tokens,1 batches; -1 on error and diagnostics set|
|`frame_destroy_preflight_v1`|`int64_t(F)`; idle, model idle, all37 scratch tensors/runtimes and caller destinations checked before any unbind write|
|`frame_destroy_v1`|`void(F)`; immediately after preflight and source P1 unbind, destroy own tensors/runtimes, clear refs, dead tombstone; impossible failure is terminal|
|`last_error_domain_v1`, `last_error_category_v1`, `last_error_code_v1`|`int64_t(void)`; immediate fixed native diagnostic triple|

The native phase enum is exact:0 IDLE,1 ACQUIRED (including between batches),
2 STAGING,3 FORWARD,4 CE_READY,5 REDUCED,6 CORRECTED,7 FINALIZED,8 DRAINED.
Acquire sets1. First inputs copy in1 starts2/plane1; targets requires2/plane1;
mask requires2/plane2 and sets3/next_role0. Forward role20 sets4. CE requires4
and uses an additional exact next_role sentinel21→22; reduce requires4/22 then5;
correct5→6; accumulate6→1, switching sum_bank only after successful dispatch.
Finalize requires1 with batches>0, sets7. Failure never advances readiness;
cleanup may act from any acquired phase1..7, including unfinished staging/forward/
metrics states. Its preflight requires finalized result validity only when
committed==1; abort cleanup must not require results that were never computed.
A publish-admission failure occurs in8 DRAINED with P1/D2 already restored: its
abort path skips second drain/restoration and performs finish/source token cleanup
only, preserving the first publication error and all previous caller values. `cleanup_ready` sets only after all
checks; no further numerical/staging call is admitted. Drain transitions8 and
preserves whether phase7 had finalized data with a separate readiness bit encoded
in `committed`: use exact values0 no staged result,1 staged,2 committed. It is
reset0 on acquire,1 by finalize,2 by publish; on finish retain2 for counter reads,
and retain previous published validity on abort in `published_valid`;
never lose a previous successful report when a later call fails.

Publication rollback never copies old results: no caller output has changed.
Failure after acquire invokes cleanup/drain/finish; before acquire only acquired
source preparation is canceled. Invalid untrusted entry returns bounded status,
but impossible admitted drain/finish invariants call abort before mutation or
fail-stop after any irreversible store. No recoverable partial cleanup.

E3-local statuses are0 OK,1 INVALID_ARGUMENT,2 INVALID_STATE,3 SHAPE_MISMATCH,
4 UNSUPPORTED,5 INTERNAL. Local codes0 OK,1 IDENTITY,2 PHASE,3 SHAPE,4 PROFILE,
5 ALLOCATION,6 CAPACITY,7 REENTRANCY,8 DESTINATION,9 INVARIANT. Diagnostics domain0
is this pair; domain1 K1,2 I1,3 I2 preserve predecessor enums exactly. Domain4 is
D2 idle0..2 and domain5 P1 modes0..4 when snapshotted by source code, not native
rewrites of their meanings. For these source-only domains, category=status and
code=status, using the fixed companion mapping; native accessors never invent a
new D2/P1 error enumeration. Native provider errors preserve K1 category/code;
shared pin errors preserve I2 category/code. Return is a transport status, never
assumed numerically identical to K1/I2 category. Capture all three immediately
before another native call. Invalid diagnostic triples become internal/invariant.
No message pointer or dynamically allocated native error is retained.
Included M3T domain0 is not E3 domain0: translate OWNER-admit IDENTITY1→E3
IDENTITY1 and LIFECYCLE2→E3 PHASE2 explicitly. Other M3T domain0 helper failures
must be translated by meaning (ALLOCATION5→5, SHAPE4→3, REENTRANCY9→7,
INTERNAL10→9); unknown triples map E3 INTERNAL/INVARIANT. Never forward a raw
M3T code3/4/9/10 as an E3 code. For dependency domains1..3 preserve category/code,
map return status by its A0 semantic category: invalid-argument/dtype-mismatch/
device-mismatch/noncontiguous→1, invalid-state→2, shape-mismatch→3,
unsupported/determinism-unavailable→4, all other failure categories→5. The exact
outward A0 category still comes from domain/category (existing m3t-native-category
mapping for1..3), not this coarse nonzero status. Domains4/5 are source snapshot
translations only and use their companion mappings.

## 5. Closed role and metric transcript

Eshkol owns literal calls0..20 in `e3_forward_schedule.esk`; native role dispatch
performs only that one operation. Input Pn means canonical held pin view indexn.
All unlisted forward outputs are f32[1,2,4]; head outputs f32[1,2,2,2], FU/FG
f32[1,2,8], Z f32[1,2,256]. All request/data schemas are inherited unchanged.

| Role/output | Inputs | Fixed row and request |
|---|---|---|
|0 ET|ids,P10|N3K embedding.forward [1,2,256,4]|
|1 EP|positions,P13|N3K embedding.forward [1,2,2,4]|
|2 X|ET,EP|N3K residual.forward [1,2,4]|
|3 N1|X,P7,P6,epsilon|N2 layer-norm.forward [1,2,4]|
|4 QT;5 KT;6 VT|N1,P2;N1,P0;N1,P3|N3K linear.forward-no-bias [1,2,4,4]|
|7 QH;8 KH;9 VH|QT;KT;VT|N3K heads.split.forward [1,2,2,2]|
|10 AH|QH,KH,VH,positions,positions,keep|A2 causal-attention.forward [1,2,2,2,2,2]|
|11 AT|AH|N3K heads.merge.forward [1,2,2,2]|
|12 AO|AT,P1|N3K linear.forward-no-bias [1,2,4,4]|
|13 R|X,AO|N3K residual.forward [1,2,4]|
|14 N2|R,P9,P8,epsilon|N2 layer-norm.forward [1,2,4]|
|15 FU|N2,P5|N3K linear.forward-no-bias [1,2,4,8]|
|16 FG|FU|N3K gelu.forward [1,2,8]|
|17 FD|FG,P4|N3K linear.forward-no-bias [1,2,8,4]|
|18 Y|R,FD|N3K residual.forward [1,2,4]|
|19 NF|Y,P12,P11,epsilon|N2 layer-norm.forward [1,2,4]|
|20 Z|NF,P10|N3K linear.forward-no-bias [1,2,4,256]|

Then Eshkol calls CE (L2 indexed CE logits/targets→CE), reduce (L3S reduce.bool
CE/mask→N_b,W_b,mean), correct (E3 correct.bool Z/targets/mask→C_b,active),
accumulate (current bank plus batch→other bank plus native counters). Reuse mean
only as required L3S output; never aggregate batch means. EOS calls E3 finalize
current N/W/C→separate four I2 result scalars; copy counts into staged_counts.
No kernel accesses source cursor/modes or caller result storage. Exact capability/operation pairs (no abbreviated aliases at dispatch) are:
`n3k.embedding-forward`/`n3k.embedding.forward`,
`n3k.residual`/`n3k.residual.forward`,
`n3k.linear`/`n3k.linear.forward-no-bias`,
`n3k.head-layout`/`n3k.heads.split.forward` or `n3k.heads.merge.forward`,
`n3k.gelu`/`n3k.gelu.forward`, `kernel.norm`/`layer-norm.forward`,
`kernel.causal-attention`/`causal-attention.forward`,
`kernel.indexed-cross-entropy`/`indexed-cross-entropy.forward` request[1,2,256],
`l3s.masked-objective`/`l3s.masked-objective.reduce.bool` request[1,2]. The numeric
companion fixes the three new E3 strings/request[1,2,256] and complete schemas.
All requests use f32/cpu/deterministic1. L3S outputs bind batch indices0/1/2,
correct writes batch3+active; accumulate uses batch0/1/3, never batch2.

## 6. Transaction and six-output publication

1. Enter existing outer `m3-call 'e3-evaluate-into!` once. Authenticate model,
   frame, fixed profile/destinations; reject model/source/native busy state.
   Allocate any invocation controls and finish error preparation before mutation.
2. Prepare P1 snapshot and D2 record; all admission/profile/remaining-cap checks
   precede mutation. Failed preparation clears only acquired records. Acquire
   native pins and model token; then source model slot10=exact frame entry.
3. P1 enter writes eval to all17 nodes after preflight. Traverse D2 in Eshkol,
   owning one batch/region at a time; three sequential borrowed copies,21 roles,
   CE/reduce/correct/accumulate. Release each exact batch before region close.
4. At EOS reject empty, else finalize. On failure snapshot the first mapped error
   before any cleanup can overwrite diagnostics. End any owned D2 view through
   its existing guard, then release the exact owned batch; entry-live batches
   were rejected and are never released by E3.
5. Native cleanup preflight and D2 restore preflight; P1 restore preflights all17
   before its first write. Drain pins, retain native/source model and outer
   exclusion. Restore D2 saved ordinal/status and all17 modes. Failure path clears
   only its own acquired records; no late allocating public setter/seek.
6. Success calls publish with finalized staging still live; failure skips it.
   Then native finish clears only matching model.active, source clears matching
   model slot10 and invocation bits/roots; enclosing m3-call exits last.

Native publication is one synchronous helper. Registry-admit frame/model and
require drained/full finalized staging. Source code alone establishes both restore
records idle/all17 restored; native code cannot authenticate Eshkol mode/cursor
vectors and must not pretend a caller boolean proves that. This is a closed source
call-order invariant tested at the compiled boundary, not a public native promise.

Open four stack I2 destination scopes in order0..3. Preflight every live rank0
metadata/span/sentinel and all aliases, including staged result scopes/counters
and complete stack controls. Any failure at destination1/2/3/4 ends only scopes
already opened, saves first error, writes none of six caller values. Validate
all four staged result scopes and exact staged counter bounds before the first
write. Copy staged bits/counters to local fixed arrays, end result scopes, and
check each stack guard's self/owner/view still equals the just-opened guard while
it remains open. There is no existing scoped-check API exposing the I2 owner's
private active sentinel. Successful scoped-begin plus distinct owners, no escape/
callback and serialized unchanged guards prove the later end precondition; the
existing end itself checks the owner sentinel. Do not invent a new common helper. No allocating
I2 copy/borrow owner is created. Under serialized immutability there are no
fallible operations after this point: memcpy four4-byte floats in order, store
published_counts[0..1], mark published_valid/committed, end destination scopes
using checked nonrecoverable cleanup, return0. Subsequent matched-token/source
cleanup performs immediate stores only. Test success→failure→read must recover all six
previous values using persistent published_valid, not last-call readiness. Any impossible sentinel failure is
fail-stop, never a reported recoverable six-output partial success.

“Atomic” means recoverable-call byte atomicity under closed serialized execution,
not a hardware-wide atomic write or concurrent-reader guarantee. Prior values
and prior published counters remain byte-identical on every recoverable failure.
Counter access is exact immediate i64, no fallible number boxing after commit.

## 7. First-error representation

The [exact error lifetime contract](e3/E3_ERROR_LIFETIME_CONTRACT.md) binds actual
E1 lexical dispatch and pinned compiler/runtime behavior. Package init constructs
156 immutable E1 shells,13 stages×12 A0 categories. Every cell has operation
`e3-evaluate-into!`, fixed safe stage details, cause#f. The current stage values
are0 admission,1 P1-prepare,2 D2-prepare,3 acquire,4 mode-enter,5 next-batch,
6 staging,7 forward,8 CE,9 reduce,10 correct,11 accumulate,12 finalize/publication.
Cleanup invariant defects are terminal and never become a replacement first error.

Before each catch's acquired work allocate its argument pair in that same region.
Classify with existing `e1-internal-dispatch` predicate/category using that pair;
ordinary public getters allocate and are forbidden here. Clear the pair before
region exit. Store only immediate first-error index and diagnostic fields, then
freeze the index into source entry15 and diagnostic vector12 once, then
select the root table shell into source entry11. Current-stage is a separate
invocation-local immediate; no later failure/cleanup call overwrites the first
index or vector12. Reset entry15=-1/entry11=#f only at next admitted invocation. Never retain/promote the arbitrary
caught graph. Source error category is preserved; original details/cause/object
identity are deliberately normalized, consistent with bounded M3 dependency policy.

On error, finish restoration and token cleanup, return immediate#f normally through
existing m3-call so its normal path clears the outer guard; only then `(raise ...)`
the selected root error outside that macro. Success returns#t, with no throw or
error construction after publication. The compiler allocates an exception header/
message even for a prebuilt error: error-delivery allocation is outside rollback,
counted separately. Runtime exhaustion during that final delivery can lose the
chosen diagnostic; all state/output preservation obligations are already complete.
No noalloc raise or arbitrary promotion reservation API is invented.

## 8. Exact private tuple and generated-manifest method

Choose `scripts/build-e3-private.sh`, `scripts/e3-private-package-policy.sh`,
`scripts/e3-private-native-inputs.sh`, with build-only manifest generator
`scripts/generate-e3-private-manifests.py`. Default artifact
`build/e3-private/libeshkol_transformer_e3_private.a` has one member
`e3_private_package.o`; no facade directory or installation target. Test driver is
explicitly test-only and cannot supply public-path acceptance.

Ordered root loads: unchanged M3 root (its18 transitive inputs), common adapters,
P1 E3 wrappers, E3 errors, existing D2 semantic core, E3 fixed identity, generated D2 dataset,
E3 private extension, E3 D2 restore, E3 forward schedule. The exact build entry `native/e3_private_driver_root.esk` loads
that exact root then fixed `tests/e3_private/driver.esk`. No original D2 dataset
load, D2/T2 root, G3 extension, second package initializer or registry archive.

Private bridge includes M3 bridge once and adds exactly
`void et_e3_test_run_v1(void *scenario, void *corpus_path, void *output)` using
existing initialized boxed argument/output conventions. Two fixture-only localized C helpers in that bridge are exactly
`void *et_e3_test_destination_create_v1(void)` and
`int64_t et_e3_test_destination_destroy_v1(void *tensor)`. They call existing
I2 rank0 creation/destruction, and never run on the evaluation path. Each returns
NULL/nonzero on failure using the fixed fixture status; all partial setup owners
are released. Both are compiled only with ET_E3_TESTING. Corresponding Eshkol
`e3-test-run/2` is defined only by the fixed test driver; it creates/reuses genuine
owners and invokes the closed private production-source seams. Observers and
fault injection are compiled only under `ET_E3_TESTING` and are called internally
by this driver/fixture, never additional global public exports. Normal source
absence and negative-link tests remain mandatory. Inherited M3 wrappers are
retained for reentry/compatibility tests, not called by the evaluation transcript.

Exact exported globals set = sorted contents of merged
`native/m3_package_defined_symbols.txt` union `{et_e3_test_run_v1}`. Existing
public boxed names are unchanged; 38 M3T rename left sides become common checked
names per issue101; eight M3 mappings unchanged; all other predecessor map rows
unchanged. Add exactly `e3-test-run et_e3_private_test_run_cabi_v1`. Export allowlist
= inherited M3 public_exports plus the LF line `et_e3_test_run_v1`. The existing export validator accepts only
et_e1b_public names: add the one exact et_e3_test_run_v1 exception only after this
E3 tuple is admitted, never broaden its regex for other packages. Keep the six E1 inspector globals
already in the predecessor set. No new numeric/frame/D2 idle accessor survives
localization. No arbitrary observer/implementation symbol glob is allowed. This selected
artifact always defines ET_E3_TESTING; optimized and sanitized builds share this
same fixed private test-driver surface. There is no claimed second production E3
aggregate or invented normal tuple. “Absent from normal artifacts” means the
standalone metric and predecessor packages; negative-link them against the test
fixture accessors. Instrumented fault helpers remain localized inside this private
test artifact and their complete names are reviewed from source/evidence.

Ordinary native compile entries in order: `native/data_io.c`,
`native/checkpoint_io.c`, `native/kernel_abi.c`,
`src/eshkol_transformer/m3_i64_integration.c`, `native/t1_i64_shell.c`,
`src/eshkol_transformer/m3_call_f32_integration.c`,
`src/eshkol_transformer/e3_frame.c`, `native/e3_d2_native.c`,
`native/indexed_cross_entropy.c`, `native/l3s_masked_objective_provider.c`,
`native/e3_evaluation_metrics_provider.c`. Fixed provider objects are
`n2/n2_primitives_provider.o`, `n3k/n3k_primitives_provider.o`,
`a2/a2_attention_provider.o`, built once. E1/package bridge translation units
follow inherited builder policy and are included exactly once, not duplicated in
this list. Includes preserve raw transitive closure, including f32_tensor.c,
m3t_f32_integration.c,m3_model.c,m3t_transport.c,d2_native.c,p1_identity.c.
No direct additional f32/i64/M3T/D2 object or completed M3/D2 archive is linked.

Ordered trusted include roots retain `internal/p1/lib`, `internal/c1/lib`,
`internal/t1/lib`, `src`; append `internal/d2/lib`, `internal/e3/lib`, then the exact generated
source directory. These are the only three added roots. The latter contains only the pinned generated D2 source; no
ambient include shadowing is accepted. Builder must admit this exact tuple before
M3 predecessor detection and preserve component symlink/C ../spelling checks,
scrubbed environment and **unfiltered** depfiles. No generic skip-policy option. Add a distinct `e3-private-aggregate` policy,
admitted before M3 detection, selected by driver-root/bridge/rename/export tuple.
Its branch must explicitly supply: source/native manifests; the11 native entries;
the7 include roots; mapped generated-source depfile normalization; the38 checked
rename substitutions and one driver mapping; bridge flags ET_M3T_PACKAGE_BUILD,
ET_E3_TESTING and src include; source-specific `-ffp-contract=off
-fexcess-precision=standard -fno-fast-math -frounding-math` for the strict native
numeric entries; the fixed `e3-private-native-inputs.sh` three-provider object
recipe; all E3/D2/L2/L3S private unresolved-symbol denial and required-local checks;
and evidence/raw depfile capture. Reuse inherited M3/P1/I2 checks explicitly,
including existing M3C assembly rename repair, only for this admitted successor.
There is no existing generic branch that already supports generated Eshkol inputs;
its exact @BUILD@ single-file normalization and path verification are new reviewed
builder work. No regeneration or relaxation of predecessor policy is implied.

Checked-in manifest paths use prefix `native/e3_private_package_` and suffixes
`source_closure.txt`, `native_source_closure.txt`, `native_objects.txt`,
`private_renames.txt`, `public_exports.txt`, `defined_symbols.txt`,
`undefined_symbols.txt`, `archive_members.txt`, `public_strings.txt`; plus
`native/e3_private_manifest_spec.json` binds the exact ordered tuple, toolchain
pin, hashes of every source/generator and accepted frozen manifest, compiler/link flags and original/
replacement/generated D2 hashes. Generated logical path is
`@BUILD@/source/e3_d2_dataset.esk`, with that single explicit root substitution;
raw depfiles are retained alongside normalized comparison, never filtered away.
The spec never includes its own digest. During candidate generation it binds only
already-frozen predecessor manifests plus reviewed source/generator inputs; after
candidate review its manifest-hash map is extended to the accepted E3 manifests.
Manifests contain lists, not a digest of the spec, so this dependency is acyclic.
Evidence records the final spec SHA256 externally; no recursive/self hash.

Generation procedure after implementation approval:

1. Validate full source hashes and D2 one-form variant, then compile the exact
   admitted tuple in a fresh directory. Capture raw Eshkol/native depfiles, command
   records, LLVM IR, object/member list and prelocalization `nm` defined/undefined
   inventories. Actual code-generated private names are measured, not guessed.
2. Generator consumes only that evidence and the checked-in spec. It checks all
   sources within allowed roots/no extra TU/provider/test helper, exact exported
   set equation and rename map, then emits candidate LF/sorted manifests under
   `build/e3-private/candidate-manifests/`. It cannot overwrite checked-in policy
   or approve its own newly observed symbols. Classified local symbols remain
   evidence even though localization removes them from exported inventory.
3. Independent review reconciles every candidate delta against exact source
   declarations/call sites. Root accepts the reviewed candidate files in a focused
   manifest commit. Existing manifests are never regenerated wholesale or relaxed
   to make a build pass. Unknown symbols/dependencies block that build.
4. Rebuild fresh with frozen manifests; `cmp` every normalized manifest, exact
   archive member/global/string sets and raw-depfile-derived closure. Repeat in
   another fresh path for the accepted reproducibility comparison. Negative probes
   reject altered tuple/include/symlink/source/provider/global/undefined entries.

This specifies the generation method and acceptance boundary; no generated
compiler symbol/hash inventory is fabricated in a docs-only phase. Exact source
contracts permit bounded implementation; measured manifests require their own
independent integration review before runtime acceptance.

## 9. Gates, dispositions and handoff

All numerical/P1/D2 gates in companion contracts apply. E3-PRIVATE additionally
requires a real compiled fixed model→D2→21 roles→L2→L3S→metrics→four genuine I2
outputs/two counters test, independent reference, exact mixed17 modes/cursor/
parameter/gradient/RNG preservation, every failure point, no graph/reverse calls,
retained exclusion, malformed/copy/cross-frame inputs and byte-atomic destination
admission failures. Validate root-region lifetime with poisoning and allocation
counters, normal and ASan/UBSan/LSan with leak detection enabled. Measure live and
cumulative frame/I2/D2 controls, caller-region bytes, FDs and payload across
1024/8192 batches and repeated complete calls on the same frame; reconcile growth.
No performance, flat process memory, GPU, mixed precision, determinism beyond
proved fixed rows, public trainer or resume-equivalence claim follows from design.

Independent exact-delta review is pending. This phase runs document/structure checks
and optional declaration-only probes; no production edit, toolchain build, full
numerical CI or runtime acceptance. Independent numerical/lifetime/package review and common-owner/G3 impact
coordination must be recorded before committing/submitting.
