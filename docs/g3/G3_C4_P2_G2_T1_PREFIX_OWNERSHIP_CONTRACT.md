# Private P2/G2 Eshkol ownership and T1 prefix provenance contract

**Proposed prerequisite; no Eshkol P2/G2 admission or coordinator is
implemented by this document.** This is the source-private, non-EOS first
token of `(P,max G,N)=(2,2,1)` for the existing diagnostic-C4 CPU-f32 model,
owned T1/I1 `[1,2]` prompt, same-aggregate raw V256 tokenizer, and A2
capacity four. The accepted native
[`ET_G3C4_P2_G2_FIRST_FRAME_PRIVATE` first frame](G3_C4_P2_G2_PRIVATE_GENERATION_PROPOSAL.md)
and [`ET_G3C4_P2_G2_PREFIX_COMMIT_PRIVATE` commit](G3_C4_P2_G2_PREFIX_COMMIT_LEAF.md)
already admit and commit the numerical prefix. The accepted
`t1-private-g3-decode-raw-into!` checks the registered raw V256 tokenizer
and converts one little-endian `i64` staging word to one raw byte. Eshkol
currently admits neither a budget-two generator/call/output nor a call that
can remain active after this internal commit. It also has no accepted FFI
carrier for the native first-frame raw pointer arguments.

## Private admission and preallocation

A separate source-private feature gate, off by default, may extend
`g3c4-policy?` only for exact `max_new=2`, `g3c4-call-tuple?` only for
`(kind,budget)=(2,2)`, and `g3c4-output-reserve` only for exact
`(prompt,requested budget)=(2,2)` with `P+G=4`. The corresponding generator,
call and pending-output validators must agree on that same gate and retain
**requested** capacity two in the policy, call and output owner. This does
not change a public config/profile cap, C2 admission, P2/G1 admission,
package export or native/provider ABI. Existing public C4 P2/G1 dispatch
continues to cap `max_new=1`; feature-off symbols and behavior remain exact.
The Eshkol budget-two generator must bind the exact native C4 generator,
same-aggregate baseline T1 registry/core identity, live model in eval mode,
and valid policy/RNG before a P2 call. The owned P2 input, first categorical
draw availability and pending-output capacity must be checked before prefill.

Before the initial prefill commit, allocate the pending output's two-byte raw
and sixteen-byte final staging payloads through the accepted output envelope,
plus distinct private **one-byte raw and eight-byte ID staging payloads** for
the first prefix. Each bytevector also has its Eshkol length header. The
private carriers are fixed owner-record children, never public output fields.
The native pending `I1[2]` output remains zero and unready.
`g3t-output-entry-pending` must recognize the exact feature-gated P2/G2
capacity and original auxiliary identities without treating capacity two as
two emitted tokens. The P2/G1 text/copy-out/public validators continue to
reject this pending owner.

## One active call and authentic decode

Use a sibling source-private P2/G2 call scope, not
`g3c4-with-call-internal` (which always prepares and finishes), the P2/G1
shell scope (which acquires budget one and publishes), or either native
`token_frame_publish_v1`/`generation_frame_commit_v1`. The sibling scope
acquires native `(2,2)` and enrolls one exact call entry linked from model and
generator, one pending output linked from its child ledger, and a protected
record of the original call/output/native, tokenizer, output auxiliary and
prefix raw/staging identities. Its phase is `before-prefix` until the native
prefix commit succeeds, then `prefix-committed`. The scope stays inside one
`m3-call` dynamic region while the prefix remains active; it may not return
an output, detached continuation or active call to another region. A bounded
test-only path may abort inside that region after proving the prefix. A later
second-token composer must run in the *same* region before terminal finish.

The accepted native prompt prefill, `token_frame_begin_last_v1` and
`token_forward_v1` take raw `float[256]`/`int64_t*` pointers. An Eshkol
bytevector passed as `ptr` points to its signed-i64 **length header**, not
its payload. Directly passing the eight-byte ID staging bytevector to the
sampler would overwrite its length, and passing a 1,024-byte bytevector as a
logits buffer would overwrite that header. No such direct binding is
admitted. A separately reviewed **private native carrier bridge** is a
prerequisite: it authenticates an Eshkol bytevector header declaring exactly
eight payload bytes, rejects overlap with context, model, output, I1 and A2
storage, owns raw
`float[256]` prefill/forward scratch and a scalar selected-ID slot, and
calls the accepted native P2 prefill, last-logit sample and token forward in
order. On complete success only, it writes the native sampled `i64` as
little-endian bytes into the staging **payload**, leaving the length header
eight. Failures before that write leave staging unchanged and let the call
scope abort the candidate frame. No public symbol, installed facade, provider
row or exact private bridge name/signature is assumed here; its ABI and
ownership checks need their own source-and-test review before this Eshkol
coordinator can be implemented. `p2g2_prefix_commit_v1` already accepts the
one-byte raw carrier header and needs no signature change.

After the bridge succeeds, check all seven upper payload bytes are zero and
take the low payload byte as the exact `0..255` selected ID. No
caller-provided ID, encoded token or raw byte may replace this carrier. The
first prompt ID equal to EOS has no effect; a **selected** first-token EOS is
rejected before the prefix commit in this nonterminal leaf. Later terminal
`G=1` EOS publication needs a separate contract.

Call `t1-private-g3-decode-raw-into!` with the retained, reauthenticated
same-aggregate tokenizer and this same staging carrier, writing the recorded
one-byte raw carrier. Before native commit, reauthenticate the exact active
call, generator/model/tokenizer registry/core and eval binding, native owner,
output child ledger, both auxiliary identities and both prefix carrier
identities; require lengths `2/16/1/8`, unchanged selected low byte, zero
upper staging bytes, and equal decoded raw byte. Recheck after any callback
or recoverable lease/preflight operation. Native prefix commit then verifies
the byte again against its authentic sampled candidate and staged A2 K/V.
There is no decoder, allocation, provider dispatch, borrow acquisition or
recoverable release after the final Eshkol preflight and before that commit.
Only bounded phase/ledger writes follow a successful native commit; an
impossible failure in its closed mutation tail is fail-stop.

## Rollback, leases and visibility

Before the prefix commit, a caught failure aborts only the genuine staged
frame/output and releases the exact call/model/generator links. It scrubs the
captured original raw/staging buffers, leaves committed P2 cache `2/1100`,
original RNG and binding intact, and permits a fresh authenticated request.
After the prefix commit, the same authenticated rollback invokes native
`call_abort_v1` to discard the pending output, scrubs the captured carriers
and closes the Eshkol call, while preserving native committed K/V at
`3/1110`, binding and successor RNG. Native abort clears the private selected
ID/raw/next-logit continuation, so this is a failed request, not an implicit
second-token continuation. Neither path publishes an output shell, ID, text
or generated length. A new full request must pass normal admission again.

Rollback first verifies the original call/output/native and every child,
auxiliary and model/generator ledger edge. Forged, dead, cross-output or
changed edges cannot authorize release of another owner; an escaped
exception with unrepaired forged edges is fail-stop. Native output-I1 or A2
leases must reject inside the active region *before* abort changes owners;
the test may end the genuine lease and retry rollback. In an escaping guard,
an abort that still rejects is fail-stop. After a successful native abort,
scrub only the captured original buffers, mark the original pending shell
dead, clear its exact child edge, then clear model/generator links and the
call entry. Do not use P2/G1's `COMMITTED`-means-finished path.

## Proving gate and next dependency

Exercise two authentic P2 prompts, greedy and categorical, with native
bridge-produced selected-ID staging, genuine same-registry T1 raw decode,
bitwise P3 next-logit/K/V reference and independent G3-S/Philox ID/RNG
oracle. Check pre/post commit `2/1100` and `3/1110`, zero/unexposed pending
`I1[2]`, and exact rollback cache/RNG states. Cut tokenizer registry/core drift,
model/eval/pin drift, altered or equal-mutated raw **and** staging bytes,
wrong output/ledger/auxiliary identities, sampled `-1/256`, first-token EOS,
held output-I1/A2 leases, malformed/aliased bytevector headers, bridge
prefill/sample/forward failures, decoder failure and native precommit failures;
repair and retry where permitted. Inject an impossible closed-tail failure
and require fail-stop. Run feature-off P2/G1/manual/C2 checks, Q0 exact
checker allowlist/near misses, static closure, and supported pinned normal,
repeat and sanitizer Eshkol plus native gates. Only after this ownership and
provenance contract is reviewed should its code be written. Second-token
forward, EOS, terminal G=1/2 output, public ABI/profile and CLI remain
separate gates.
