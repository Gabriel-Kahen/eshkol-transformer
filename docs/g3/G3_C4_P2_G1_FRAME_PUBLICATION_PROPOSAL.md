# G3-C4 private P2/G1 frame and native-output publication proposal

**Contract for independent review; no code or public result is accepted.**
Base: PR #154 head `f82a5ac`. This leaf follows the C4 P2/G1
[pending transcript](G3_C4_P2_G1_PROPOSAL.md). It commits exactly one
sampled append at absolute position two and makes the native output record
independently live. It does not publish an Eshkol output shell, raw text, or an
installed G3/CLI3 result.

## Existing authority and required new boundary

The pending C4 path already owns a committed P2 cache (`length=2`, mask
`1100`), a `READY` token frame with one staged A2 transaction, candidate ID,
position two and successor RNG, and a unique linked output with
`numeric_ready=ids_copied=text_ready=1`, `length=1`, prospective
`cache_length=3`, exact I1 `[1]` ID and recorded successor RNG. Native
`et_g3c4_private_output_accept_text_v1` checks the raw byte against that I1
ID **at acceptance time**; the raw bytevector remains owned by the Eshkol
output entry. Current `call_prepare_end` rejects every pending output,
`call_finish` rejects one, and `output_release` discards only pending output.

Introduce a separate C4-private publication feature that depends on the
pending P2/G1, output ID/text readiness, token-forward and A2 gates. Its
new native-private generation-frame prepare and commit entrypoints take the
active context and exact output identity. They use existing A2/I1/pin APIs;
no provider row, public call ABI, M3T context, or installed symbol is added.
They must not reuse the existing
`et_g3c4_private_token_frame_publish_v1`: that routine currently commits
A2/RNG and sets `budget=0` without the linked output. Under this feature it
must reject whenever a pending output is linked, before any mutation.
Feature-off builds retain all current guards and behavior.

## Recoverable preparation and joint commit

Generation-frame prepare requires `(call_kind,budget)=(2,1)`, the sole
authenticated pending output with `(P,G)=(2,1)`, a live sealed owner and all
14 current pins matching the committed P2 binding. It checks the three
readiness flags and exact output lengths/RNG, an unborrowed I1 `[1]` whose
sole ID equals the frame candidate in `[0,255]`, a `READY` position-two frame,
and the staged A2 view: capacity-four K/V for both
heads, effective length three and mask `1110`. The scoped I1 borrow and A2
view are ended before preparation succeeds. A2 forbids a committed-cache read
borrow while its transaction is live; the already-proven P2 length and recorded
frame position two are checked against that transaction view, without
inventing a simultaneous second lease. Any allocation, borrow, shape,
linkage or binding failure leaves the output pending, A2 append staged,
committed P2 cache/binding, generator RNG and caller-visible bytes unchanged.

Feature-gated `call_prepare_end` repeats this preflight and records a distinct
end-ready state instead of rejecting this exact prepared tuple. Generation
frame commit requires end-ready, repeats the fallible preflight, then performs
one no-failure tail: commit the already-staged A2 transaction, copy the
successor RNG to the generator, detach the output from its parent, mark the
native output live, and mark the frame committed. It preserves output I1 ID,
length one, cache length three, recorded RNG and readiness. No allocation,
provider call, decoding, mutable Eshkol callback or recoverable error follows
the first commit mutation. A2 rejection after the final closed-view preflight
is fail-stop, as in the accepted [G3-T final-publication leaf](G3_T_FINAL_PUBLICATION_LEAF.md)
and C4 manual-frame publication. Internal frame-state encoding must retain
the existing C4 context/output record sizes (6496/6504 and 128 bytes) and
distinguish prepared, end-ready and committed from ordinary `READY`/idle states.
The new live-output validator accepts only a detached parent, `busy=0`,
owned I1 `[1]`, `(P,G,length,cache_length)=(2,1,1,3)`, valid recorded RNG,
and all three readiness flags; the existing dead record remains zeroed.

After commit, `call_abort` rejects; a dedicated `call_finish` path drains pins
and active-call ownership exactly once without fallible work. Before commit,
abort discards the candidate and pending output while retaining committed P2
cache/binding and old RNG; an active I1 borrow or A2 view must reject cleanup
before mutation so the caller can end it and retry. Live native-output
release first proves the I1 owner unborrowed, then destroys it and leaves an
authenticated dead tombstone; repeated release is harmless. The live output
survives call finish and generator close. Pending-reservation cleanup remains
available during early reservation rollback; direct pending release rejects
after frame preparation begins, so it cannot substitute for call abort.

## Acceptance and next interface

A source-private witness must drive real T1-backed P2 prefill, G3-S sampling,
token-forward, numeric/ID/text readiness, frame prepare/end/commit and finish.
Greedy and categorical cases compare the committed length-three A2 mask/K/V
and all 256 next-logit bits against the existing same-provider P3 reference;
an independent G3-S/Philox oracle checks sampled ID and successor RNG.
Before commit, cut allocation/borrow, nested A2 view, malformed or mutated I1
ID/descriptor, wrong/forged output identity, readiness, lengths/RNG, stale
pins, wrong frame order and A2/provider failures. Each cut must preserve
pending/cache/binding/RNG and allow repair/retry or abort. Explicitly prove
standalone `token_frame_publish_v1` rejection with a linked pending output.
After commit, prove cache `3/1110`, exact ID/length/RNG, abort rejection,
one-shot finish, independent native-output lifetime, borrowed-I1 release
rejection, release retry and tombstone. Feature-off P2/G1 remains unpublished;
G0/P1-G1, manual C4, M3T/C2 and installed package behavior remain unchanged.
Run static closure/symbol checks and supported normal/repeat/sanitizer gates.

The C4 policy validates an EOS ID but never compares it on this path; this
leaf fixes `G=1` and claims no EOS-stop or repeated-decode semantics. The
native record has no owned raw byte, and the current
`g3c4-with-call-internal` macro automatically prepares/finishes while its
ledger forbids a pending output. A later reviewed Eshkol-private coordinator
must revalidate the mutable raw bytevector and exact call/output linkage,
invoke the native joint tail without a second prepare/finish, seal the output
shell in a no-failure postcommit tail, and then define accessors/package
behavior. This native leaf alone cannot be called a text result or CLI3
`generate`.
