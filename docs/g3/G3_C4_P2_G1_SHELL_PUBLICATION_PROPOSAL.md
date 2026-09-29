# G3-C4 private P2/G1 shell and raw-byte publication proposal

**Contract for independent review; no implementation or package behavior is
accepted.** Base: reviewed native publication commit `1a2b0b6` (tree
`af0a90e`). This leaf would make one source-private Eshkol output shell live
for `(P,G,N)=(2,1,1)` after the existing native joint A2/RNG/output commit.
It does not add a public generation operation, accessor, general loop, or CLI.

## Existing owners and call boundary

`native/g3c4_output_envelope_extension.esk` reserves an exact 12-slot pending
output entry. Slot 3 holds its native output owner, slots 4–6 retain the
generator/model/tokenizer, slot 8 holds a two-element auxiliary vector with
preallocated raw `[1]` and i64-LE staging `[8]` bytevectors, and active call
ledger slot 4 holds the sole pending output. `g3t-output-entry-pending`
authenticates these edges; `g3t-t1-decode-output!` copies the native I1 ID to
staging, runs the same-aggregate raw-V256 T1 decoder into the raw byte, and
calls native `output_accept_text_v1`. That native call checks raw against the
owned I1 **only at acceptance time**. The native owner has no raw-byte field.

At `1a2b0b6`, native `generation_frame_prepare_v1(context,output)`,
`call_prepare_end_v1(context)`, `generation_frame_commit_v1(context,output)`
and `call_finish_v1(context)` implement the one-token joint tail and preserve
the independently live native output. `g3c4-with-call-internal` cannot host
this leaf: it unconditionally calls `call_prepare_end` and `call_finish`, and
`g3c4-call-success!` requires an empty child ledger. A feature-gated sibling
private call scope must use the existing 12-slot call/ledger authority,
`m3-call-state`, native acquire/abort, and registry rollback helpers, then
finish and clear those edges once. It must not invoke the old macro around
its own joint tail or change that macro's feature-off path. The new source
belongs in a non-installed C4 extension loaded after the existing call,
output-envelope, T1-decode and P2/G1 pending extensions. It binds the two
reviewed native generation-frame entrypoints alongside the existing private
call/output functions only when that feature is enabled.

## Precommit transcript and no-failure tail

The coordinator accepts only the canonical active `(kind,budget)=(2,1)`
call and its unique linked P2/G1 pending output after genuine T1-backed P2
prefill and a `READY` position-two token frame. It must not accept an output
already decoded through a separate call: numeric preparation, ID copy,
raw-V256 decode, and native text acceptance execute once *inside the same
coordinator invocation*. Thus native `ids_copied` and `text_ready` transition
from zero to one once, and the caller has no callback or result shell between
acceptance and commit. Repeated coordinator invocation rejects before a
second decode or commit.

Immediately after the native ID copy, capture staging byte zero in a local
scalar `accepted-byte`; validate staging bytes 1–7 are zero before decoding.
Before native frame prepare, reauthenticate the call/output/model/tokenizer
links, policy budget, ledger, exact raw/staging identities and lengths, and
mutable byte contents (`raw[0]=staging[0]=accepted-byte`, staging bytes 1–7
zero), along with the accepted raw-V256 tokenizer. Native text acceptance
proves the raw byte matches the owned I1; the local scalar prevents a later
equal alteration of both bytevectors from masquerading as that accepted ID.
Native frame prepare and commit independently recheck
the I1 candidate, A2 staged geometry, RNG, pins, and pending identity. All
checks and any allocation occur before native commit. The coordinator must
end its own scoped I1/A2 leases before preparing the frame. A live external
I1 borrow or A2 transaction view rejects precommit without publishing; the
lease owner must end it before retry or abort. The Eshkol call guard must not
attempt native abort while such a lease is still held, because native abort
correctly rejects it. This recoverable in-region retry/cleanup behavior needs
an explicit witness; an exception escaping a call scope with a held lease is
an invariant failure, not a partial successful result.

Order the remaining fallible operations as: native frame prepare, call
prepare-end, final Eshkol linkage/raw-byte preflight, then native joint
commit. Before the first native mutation, ordinary exceptions with closed
leases abort the staged frame and pending native output, scrub the one raw
and eight staging bytes through captured original bytevector references,
clear ledger slot 4, and tombstone the Eshkol output and call. The committed
P2 cache/binding and old generator RNG remain. A test-held lease or altered
mutable ledger edge instead rejects within the still-active private call
region: its owner ends the lease or repairs the edge before retry or explicit
abort. No pending output shell escapes.
The final raw preflight has no decoder, provider call, allocation, or mutable
callback after it. A test mutation of either or both bytevectors after native
text acceptance but before this preflight must reject and roll back, rather
than publish a byte that differs from the accepted I1.

Immediately after successful native frame commit, mark the guard committed:
abort is forbidden. Native call finish, Eshkol vector updates and return of
the *existing* output shell form one no-failure tail. An impossible failure
there is fail-stop. Before entering it, preflight the exact edges that
`g3c4-call-success!` will clear. Finish drains the native pins/call once;
the Eshkol tail sets output state `live`, drops output slots 4–6, replaces
slot 8 with its existing raw bytevector (dropping the staging root), clears
call ledger slot 4, then reuses the preflighted `g3c4-call-success!` to clear
generator/model active-call links and tombstone the call; the enclosing
`m3-call` clears its guard on return. The live output keeps native owner slot 3,
one raw byte and its diagnostic profile, with no generator/model/tokenizer
retention. A new source-private live-output validator must require the exact
12-slot shell, native owner and detached slots 4–7, slot-8 raw bytevector
length one, slots 9–10 false and diagnostic profile. Its raw byte is a
private owned value, not native output storage.
The caller obtains this shell once as the coordinator's return; no raw alias
or text accessor is installed by this leaf. A later private/public accessor
must return a fresh decoded value or byte copy, as required by the accepted
[generation contract](../PUBLIC_API_CONTRACT.md), rather than expose mutable
slot-8 storage.

An exact source-private live-output release must authenticate the shell and
native owner, call native `output_release_v1` before changing Eshkol fields,
and then tombstone the Eshkol entry. Native I1-borrow rejection leaves both
live and permits retry; repeated release of the authenticated dead shell is
idempotent. The output survives call finish and generator close. A forged,
wrong-kind, stale, or cross-call shell cannot acquire or release this owner.

## Acceptance and later package seam

The source-private witness must use authentic sealed C4 parameters, owned
T1 P2 prompt, both greedy and categorical G3-S policies, and the genuine
21-role prefill/token-forward path. It checks output `[1]` ID, raw byte,
length one, cache length `3/1110`, successor RNG, P3 same-provider logits/KV
parity, and independent Philox sampling; then tests output retention after
generator close and typed release. Cut allocation, T1/ID-copy/decode/native
text acceptance, altered raw/staging bytes (including equal alteration of
both), forged shell/ledger edges, wrong
order, held I1/A2 leases, frame prepare/end/commit rejection, and native
output-release borrow rejection. Every precommit cut must prove no live
shell, committed P2 cache and old RNG; after any injected lease or linkage
is repaired, prove explicit abort tombstones or a successful fresh retry.
Test the old auto call scope and feature-off tuple
unchanged. Use private source closure, symbol/ABI audit, and supported
normal/repeat/sanitizer Eshkol-plus-native gates; no installed public symbol
or `lib/transformer/generation.esk` change belongs to this leaf.

The accepted public result contract includes an emitted EOS in ID, text,
length and cache length. This leaf fixes one emitted token and validates the
configured EOS ID only through the existing policy; it does not compare the
candidate with EOS or define stop/continuation behavior. Public P2/G1 output
wrapping, fresh text/ID/length/RNG accessors, package installation, G>1,
N>1, no-cache parity, and CLI3 `generate` each need later reviewed gates.
