# G3-T manual kind-0 P2 transport proposal

Status: **accepted for bounded private implementation** after independent source
review. This document specifies a private dependency for G3-M P2 orchestration;
it does not claim an implemented G3-T operation, public generation, manual
decode, or a new external ABI.

## Existing boundary and proposed feature

The accepted G3-C4 manual contracts admit a two-token prefill transcript,
dispatch 21 ordered roles, and publish its last 256 logits. They describe the
G3-C4 owner. G3-T has a separate native owner and provider implementation:
`g3t_transport.c` currently admits only length one for manual kind 0;
`g3t_prefill_roles.inc` routes T2 only for generation kind 2. G3-T's T2
provider helpers already execute the same accepted semantic route for its
private generation path. G3-T uses N3K/N2/A2, a capacity-two A2 cache and a
4,736-byte parameter snapshot; the G3-C4 contract's capacity-four cache and
snapshot size must not be copied into this implementation.

Source anchors at the proposal base (`5d2df96`): manual `frame_begin` and
single-ID staging at `g3t_transport.c:1373-1394`; P1-only manual dispatch and
generation-only T2 dispatch at `g3t_prefill_roles.inc:219-240`; P1-only
preparation, preflight and publication at `g3t_transport.c:1459-1490`,
`:1647-1719` and `:1847-1871`; existing T2 scratch and provider groups in
`g3t_prefill2_roles.inc`; and the accepted G3-C4 frame-begin, role-step and
frame-commit contracts. The accepted G3-T manual P1 leaf remains the
regression contract.

Propose a source-private `ET_G3T_MANUAL_P2_PREFILL_PRIVATE` feature requiring
the existing manual P1, P2 zero-budget, and authenticated T1 input features.
It extends the existing `frame_begin`, `role_step`, `frame_prepare`,
`call_prepare_end`, `frame_commit`, `call_abort`, and `call_finish` symbols. It
adds no native symbol, Eshkol transport wrapper, package export, or public
facade. With the feature off, P1 behavior and the normal production object
must retain their existing behavior and symbol set.

## Admission and transcript

An active kind-0 call, acquired under the existing M3 guard with fourteen
pins, must have exactly one attached pending kind-3 `[1,256]` CPU f32 logits
owner, no pending output, no active frame and a valid prior cache. A read
borrow on the prior cache may remain outstanding until publication preflight,
matching P1 rollback behavior. `frame_begin`
accepts frame kind 1 and an authentic live G3-T kind-2 input with a canonical
owned CPU i64 tensor of rank two, shape `[1,2]`, dense row-major layout and
16 bytes. Each ID is in 0..255. The native frame must first require the I1 to
be unborrowed, then borrow it, inspect its typed view, copy both IDs to native
owned `prompt_ids[2]`, release the
borrow, and only then install the unpublished capacity-two A2 candidate and
frame `{kind=1, prompt_length=2, next_ordinal=0}`. The validated inline ID
slots must match the canonical I1; a mismatch rejects before frame install.
No pointer, borrow, or T1 shell survives admission. The caller may release the
input immediately afterward without changing the transcript.

The existing `g3t-generation-t1-input-create` copies a same-aggregate sealed
T1 P2 input into such an I1 owner. The current
`g3t-generation-pair-input-create` creates only inline IDs and is therefore
not eligible for this manual P2 route: it has no canonical I1 tensor to
authenticate or borrow. This proposal adds no new owner API. P1 remains
eligible for its accepted input forms; this proposal does not change P1
admission.

No generation `P+G` or draw preflight applies to kind 0. A missing, forged,
wrong-kind or dead owner rejects before mutation; wrong frame kind, repeat
frame, wrong length, non-byte ID, busy/borrowed input or absent pending logits
reject without a candidate. I1 borrow failure propagates its original I1
status. On candidate allocation failure, the call and pending result remain
active for retry or abort, while committed cache, binding and RNG remain
unchanged.

## Exact numerical route

`role_step(ctx, ordinal)` retains its order: active-context authentication,
range check 0..20, then pending-result/frame/next-ordinal checks. For kind-0
P2 it selects the already reviewed `g3t_prefill2_plain` at ordinals 0..9
and 11..20 and `g3t_prefill2_attention` at ordinal 10, one provider dispatch
per ordinal. The helper owns per-role scratch and increments `next_ordinal`
only after success. It consumes the two copied IDs at positions `{0,1}`;
attention stages both real K/V positions in one A2 transaction and uses the
causal rows `{1,0}` and `{1,1}`. Failed dispatch or A2 allocation retains
the same ordinal for retry. Kind-0 P1 keeps its current P1 branch; kind-2
generation keeps its current T2 branch.

## Preparation and atomic publication

After ordinal 20, `frame_prepare` requires the exact pending logits, all
fourteen live parameter pins, a complete length-two A2 transaction and an
unprepared frame. It snapshots the fourteen pin identities and exactly 4,736 value
bytes, then copies only `p2.z[256..511]` (row one, 256 f32 values) into the
owned `[1,256]` logits tensor. It marks the frame prepared only after copy
success. That tensor stays pending until commit; abort destroys it, so a
failed later preflight must never expose it as detached live output.

`call_prepare_end` and `frame_commit` repeat the P2 publication preflight:
exact pending linkage, completed ordinal 21, unchanged pins and binding
snapshot, writable logits rank/shape/bytes with bits equal to the P2 last
row, staged layer-zero K and V tensors each of rank 4, shape `[1,2,2,2]`
and 32 bytes, with both positions bitwise equal to the P2 scratch K/V;
lengths tensor of rank 1, shape `[1]`, 8 bytes and value `2`; keep-mask
tensor of rank 2, shape `[1,2]`, 2 bytes and values `[1,1]`; and a read lease
proving the old cache can be destroyed. The old cache may have
committed length 0, 1 or 2. End every view
and lease before mutation, preserving the first failure status. The first
preflight marks end-ready; commit repeats it and then uses the accepted P1
no-failure tail: commit A2 transaction, destroy the old cache, install the
candidate, binding snapshot and last row, detach the logits owner, mark
committed, then immediately `call_finish` to drain pins and active linkage.
An impossible failure after the first commit mutation aborts the process.
Neither role nor publication samples or advances RNG.

Native status precedence follows the existing G3-T entry order. Each entry
clears last error and authenticates the context first. `frame_begin` then
checks call/pending/frame lifecycle, input registry identity and live state,
input metadata length and busy state, unborrowed I1, typed I1 view and IDs,
then candidate allocation. Forged input is argument/identity; dead, wrong
metadata length or busy input is state/lifecycle. A malformed typed I1 view
is shape/token-range, a non-byte ID is state/lifecycle (matching manual P1),
and inline-versus-owned ID divergence is state/topology. I1 borrow failure
preserves its I1 domain/category/code. `role_step` rejects an out-of-range
ordinal as argument/config before frame-state checks; bad manual lifecycle is
state/lifecycle. F32, A2 and provider failures preserve their first domain,
category and code. Cleanup may not replace the first error;
an impossible cleanup failure is fail-stop. A borrowed pending result may
block `call_abort` before it changes frame/call state, as in P1; the caller
must release that borrow and retry abort. Other precommit failures leave the
prior cache, binding, RNG, old detached logits and pins/active-call semantics
of the current P1 transport intact.

## Evidence and consumer gate

The implementation needs a closed source manifest based on the accepted G3-T
manual P1 closure, a feature-gated source checker, Q0 Python isolation, and
normal/repeat/ASan+UBSan+LSan runs under the pinned network-disabled toolchain.
The production object must exclude test symbols and the feature when off.
Tests must show an authentic T1-backed `[1,2]` I1 input; release after frame
begin; exact 21 T2 dispatches; all 256 last-logit bits against independent
M3T row one; both K/V positions and length-two mask; repeated P2 and P1-to-P2
replacement; retained older detached logits; wrong input/length/borrow/busy
and malformed typed view; ordinal, I1, f32, provider, A2, result-borrow,
old-cache-borrow and binding failures; retry/abort and ownership counts.
The existing P1 and P2/G0 aggregate must pass unchanged as regressions.

Only after this G3-T transport is independently reviewed may the private
G3-M composer add `(g3m-prefill-p2! generator input)`, using the same outer
`m3-call`, acquire/reserve/21 roles/prepare/preflight/commit/finish sequence
as P1, with native frame admission providing the P2 proof. That later leaf
must make its own source closure and normal/repeat/sanitizer witness. This
proposal does not authorize implementing against a guessed P2 wrapper.
