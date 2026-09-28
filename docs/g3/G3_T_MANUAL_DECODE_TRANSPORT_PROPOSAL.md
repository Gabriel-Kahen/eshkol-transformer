# G3-T manual one-token decode transport proposal

Status: **proposal pending independent source review**. This is the first
missing private transport leaf after the isolated G3-M manual P2 composer on
the accepted capacity-two path. It adds no implementation, Eshkol operation,
public facade, package export, repeated append, or generation claim.

## Authority and exact boundary

The accepted [G3-T private contract](G3_T_PRIVATE_CONTRACT.md) assigns call
kind 1 to manual decode, frame kind 2 to a non-null authentic kind-2 input,
an owned last-logits f32 `[1,256]` result, a 21-role position-one append, and
cache length `1→2` without a draw. The accepted G3-C4
[frame begin](G3_C4_MANUAL_FRAME_BEGIN_STEP23B_CONTRACT.md),
[role step](G3_C4_MANUAL_ROLE_STEP_CONTRACT.md),
[A2 ordinal 10](G3_C4_MANUAL_A2_ORDINAL10_CONTRACT.md), and
[frame commit](G3_C4_MANUAL_FRAME_COMMIT_CONTRACT.md) prove the semantic route
for a different owner; they do not implement G3-T.

Source anchors on independently reviewed G3-T P2 head `832a709` (its native
source is byte-identical to `fd11d72`): `g3t_transport.c:567-591` already
admits native call kind 1; `:865-896` reserves kind-3 logits for kinds 0/1;
`:1385-1513` sends only kind 0 to manual `frame_begin` and requires NULL input
for generation decode; `g3t_prefill_roles.inc:213-252` sends only kind 0 to
manual roles, while its T1 helpers already use position one and the committed
cache when `frame.kind==2`; `g3t_transport.c:1519-1595`, `:1721-1818`,
`:1917-2055`, and `:2059-2086` prepare/preflight/commit/finish only manual
kind 0 or generation. The private Eshkol wrappers still admit only call kinds
0/2 and frame-kind-2 input `#f` (`g3t_prefill_sample_extension.esk:134-167`,
`:235-251`); widening them is a **later consumer leaf**, not this native
transport leaf.

Propose `ET_G3T_MANUAL_DECODE_PRIVATE`, depending on the existing manual P1
logits/commit path and owned one-token input feature. It extends only the
existing native `call_acquire`, `frame_begin`, `role_step`, `frame_prepare`,
`call_prepare_end`, `frame_commit`, `call_abort`, and `call_finish` symbols.
No new native symbol or owner kind is needed. Feature-off behavior, P1/P2
manual prefill and P1/G0, P2/G0, P1/G1 generation remain unchanged.

## Admission and exact transcript

Under the already acquired kind-1 call and fourteen pins, exactly one linked
pending kind-3 logits owner must exist; there is no pending generation output
or active frame. A committed length-one A2 cache and current 4,736-byte / 14-
identity parameter binding are required. Native `frame_begin(ctx,input,2)`
requires an authentic live kind-2 input from the same G3-T registry, length
one, with canonical unborrowed CPU i64 dense rank-2 `[1,1]` I1 storage of
eight bytes and an ID in `0..255`. The existing
`g3t-generation-token-input-create` and one-token
`g3t-generation-t1-input-create` supply this owner; no new input API is
proposed. Borrow the typed I1, verify its view and equality with the inline
ID, copy the ID into frame-owned scalar storage, and end the borrow before
using a short A2 read borrow to capture the committed first-position K/V bits
and verify length one and mask `[1,0]`; end that lease before installing
`{kind=2,prompt_length=1,next_ordinal=0}`. The frame has **no
candidate cache**: its A2 append transaction targets the committed cache.
Releasing the input after admission must not alter the transcript. A missing
prefix, stale binding, length-two cache/input, wrong frame kind, borrowed
input or pending-result mismatch rejects without installing a frame. This
does not use generation `P+G` or draw preflight.

`role_step` keeps context authentication then ordinal range `0..20` before
frame-state checks. For kind-1/frame-2 it calls the existing T1 G3-N/N2
`g3t_prefill_plain` for ordinals 0..9 and 11..20 and
`g3t_prefill_attention` at 10, one provider dispatch per ordinal. Role 0
uses the copied ID; role 1 uses learned position 1. Attention uses query
position `[1]`, key positions `[0,1]`, causal bool row `[1,1]`, and the
existing capacity-two A2 append transaction with one new K/V position.
The staged K/V row has f32 shape `[1,2,1,2]`; the effective full cache K/V
views have shape `[1,2,2,2]`. Each failed provider/A2 step keeps the same
ordinal retryable; a failed transaction is aborted and its tail scrubbed.

## Preparation, publication and failures

After exactly 21 roles, `frame_prepare(ctx,logits)` validates the exact pending
logits, fourteen live pins and unchanged committed binding, completed A2
transaction, and unprepared frame. It copies the single row `frame.z[0..255]`
into the owned CPU f32 rank-2 `[1,256]` result (1,024 bytes), then marks the
frame prepared. It must not replace the committed parameter binding. The
result remains pending until commit.

`call_prepare_end` and `frame_commit` repeat the same fallible preflight:
exact pending linkage, ordinal 21, prepared state, unchanged binding, writable
logits with bits equal to `frame.z`, staged layer-zero K/V rank-4
`[1,2,2,2]` (32 bytes each), lengths rank-1 `[1]` (8 bytes, value `2`),
and keep mask rank-2 `[1,2]` (2 bytes, values `[1,1]`). The first K/V
position must match the frame-owned admission snapshot bitwise; the second
must match staged `frame.kh`/`frame.vh` bits. A2's transaction view closes
before mutation.
Preflight must prove commit has no live view and every layer is staged; the
accepted A2 transaction excludes concurrent committed-cache read borrows.
After the last recoverable check, commit the transaction in place, update
`last_logits`, detach the result, mark committed, and immediately finish to
drain pins and active links. Cache identity, parameter binding and RNG remain
unchanged. Any impossible failure after first mutation is fail-stop.

Native entry status precedence follows G3-T: clear error and authenticate
context first; lifecycle/pending/frame checks precede input registry identity,
metadata/busy state, typed I1 view, prefix/binding and transaction work.
Forged input is argument/identity; dead, wrong length or busy input is
state/lifecycle; malformed typed I1 is shape/token-range; inline/I1 mismatch
or stale binding is state/topology. Preserve original I1, F32, A2 or provider
domain/category/code across cleanup. An out-of-range ordinal is
argument/config before frame-state checks. A borrowed pending result may
block abort before any cleanup, allowing borrow release and abort retry.
Other precommit failures abort the transaction and pending result, scrub
position-one K/V, drain pins and retain the length-one cache, binding, RNG
and older detached outputs. No sampler call, RNG advance, scalar numerical
fallback, callback or AD graph belongs to this route.

## Acceptance and consumer gate

Use a closed source manifest extending the reviewed G3-T manual P2 closure,
a feature-gated checker, Q0 Python isolation, and the pinned network-disabled
normal/repeat/ASan+UBSan+LSan aggregate. Production feature-off object and
symbol inventory must remain unchanged. A genuine owned `[1,1]` input and
independent M3T two-token reference must prove all 256 final-logit bits,
both K/V positions, cache length/mask, unchanged RNG and detached ownership
after parent close. Cover a P1 manual and P1/G0 predecessor; reject no
prefix, stale binding, P2/length-two cache, wrong input/frame/ordinal,
borrowed/malformed I1, borrowed pending result, allocation/provider/A2
failures and retry/abort. Preserve P1/P2 manual and P1/G0, P2/G0, P1/G1
regressions. Report real provider failures separately from any synthetic
test-only dispatch cut; seal exact commit/tree, source hashes and logs.

Only after independent native review may a separate G3-T Eshkol wrapper leaf
admit kind-1 acquisition and non-null frame-kind-2 input, followed by a G3-M
manual decode composer. The proposed G3-G public facade and a production
seeded generate schedule remain later gates; this proposal does not install
either.
