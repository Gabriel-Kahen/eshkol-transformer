# G3-C4 fixed private P2/G2 generation proposal

**Proposed for independent contract review; no P2/G2 implementation or public
admission is accepted.** Base: merged #163 `3dd901b`/tree `b5c11cb`.
This is one source-private `(P,max G,N)=(2,2,1)` CPU-f32/V256 request on the
existing fourteen-parameter `diagnostic-c4` model, owned I1 `[1,2]` prompt,
raw-V256 T1 tokenizer and capacity-four A2 cache. It builds on the accepted
[G3 generation semantics](../G3_GENERATION_PROPOSAL.md),
[C4 one-frame publication](G3_C4_P2_G1_FRAME_PUBLICATION_PROPOSAL.md),
[raw shell](G3_C4_P2_G1_SHELL_PUBLICATION_PROPOSAL.md), G3-S/Philox,
[last-logit frame](G3_C4_LAST_LOGIT_FRAME_STEP13A_CONTRACT.md),
[one-token forward](G3_C4_TOKEN_FORWARD_STEP11A_CONTRACT.md), and
[T4 reference](G3_C4_FULL_PREFIX_FORWARD_STEP8A_CONTRACT.md). The fixed
public `diagnostic-c4-p2g1` and C2 profiles retain their current admissions,
owners, names and output accessors. No public config, native ABI signature,
provider row or package export is presumed by this proposal.

## Admission and numerical sequence

Only a separate private feature gate may admit policy `max_new=2`, an active
generation call `(kind,budget)=(2,2)`, and a linked output with requested
capacity two and `P+G=4`. Currently `et_g3c4_valid_generator_policy` caps
`max_new` at one; Eshkol `g3c4-policy?`,
`et_g3c4_valid_call_tuple`, prompt preflight, output
reservation, Eshkol `g3c4-output-reserve`, and the public profile cap the
request independently. Widening one guard alone is forbidden. Authenticate
the exact C4 generator, owned P2 prompt, tokenizer, eval/pin binding and
available first categorical draw before prefill or output mutation. EOS is
Eshkol `#f`/native `-1` (disabled) or one byte ID `0..255`; it never relaxes the
`P+G<=4` admission. Prompt bytes equal to EOS do not stop generation.

Prefill the authentic two prompt IDs and commit length `2`, mask `1100`.
Sample the P2 last-logit `[1,256]` row with accepted G3-S; open the
position-2 frame, run the existing 21-role T1 forward, and prepare its ID,
raw-V256 decoded byte and successor RNG before the first commit. The first
commit appends K/V at position 2, advancing cache to `3/1110`; categorical
consumes exactly one Philox block, greedy none. It is an **internal prefix
commit**, not the existing P2/G1 joint output publication or a visible
partial result. Keep its 256 next-logit words and its selected ID/accepted
raw byte in authenticated private state. A matching emitted EOS terminates
with logical `G=1` only **after** this forward and append; in that branch,
the already-prepared output becomes live in the same no-failure commit tail.
Otherwise the call remains active with one token remaining and no live
output. Native finish/abort must recognize this intermediate committed
prefix without applying the existing P2/G1 `COMMITTED`-means-finished rule.
The current standalone `token_frame_publish_v1` rejects a linked pending
output, while `generation_frame_commit_v1` detaches that output; neither
may be repurposed as this intermediate commit without a reviewed state and
ownership change.

If the first emitted ID is not EOS, take the second sample directly from
those genuine next logits through the accepted `[1,256]` G3-S route. Do not
pad a four-row logits carrier or rerun P2 prefill. Begin a new width-one A2
transaction at committed position 3, perform the same T1 forward and closed
view/precommit checks, then atomically append position 3 and its selected
ID/RNG and publish the prepared sole output in one no-failure tail. Final
cache is `4/1111`, logical `G=2`; a second emitted EOS is
included and ends the request, as does budget exhaustion. Categorical
consumes one block per **emitted** token, including singleton selection and
EOS; greedy consumes zero. A failed attempt consumes no block. An exhausted
state rejects before the first prefill when the first categorical draw is
needed; exhaustion discovered for token two rejects before its frame starts,
retaining the committed first-token prefix and successor RNG. No extra
draw or synthetic token follows EOS. The T4 full-prefix function is a
same-provider logits/K/V reference, not a no-cache generation mode or an
independent sampling oracle; sampled IDs and all four RNG words require an
independent G3-S/Philox oracle.

## Output, ownership and failure cuts

Before initial prefill commit, preallocate and authenticate bounded ID/raw/
staging/result capacity for two emitted bytes and any alternative terminal
length-one representation. The owner must record **requested** budget two
separately from **actual** generated count `1` or `2`; neither a pending
`generated_length=budget` invariant nor a spare second byte may masquerade
as emitted output. On success, the sole independently live native and
Eshkol output has exact logical IDs/raw text of the emitted count, generated
length `G`, cache length `2+G`, final four-word RNG snapshot and no retained
generator/model/tokenizer edge. An unused capacity slot is zeroed and never
exposed; a later accessor must clone only the exact logical count. Existing
P2/G1 validators/copy-out must continue rejecting this distinct private
output. The implementation must prove a no-failure finish/shell tail after
its last A2/RNG/output mutation; no allocation, decoder, provider dispatch,
borrow acquisition or recoverable release may follow that point.

Before the first token commit, failure aborts the staged frame and pending
output, scrubs all raw/staging storage, and leaves committed P2 cache/binding
and original RNG; a valid retry remains possible. **After the first token
commit, any second-sample, second-frame, decode, allocation, lease or
prepublication failure retains committed length `3/1110`, first-token K/V,
binding and advanced RNG, but publishes no native output or Eshkol shell and
exposes no partial ID/text.** It discards
only the second candidate transaction and pending output/staging. A new
full-request retry must re-prefill by the usual authenticated route; no
implicit continuation or rewind of that first committed draw is promised.
All fallible output preparation must precede the terminal commit; an
impossible failure inside its closed no-failure tail is fail-stop under the
accepted G3-T/C4 publication rule. Output remains independently live after
call finish/generator close; exact-owner release is native-first,
borrow-aware, retryable and idempotent. Held I1/A2 leases reject before
abort/release mutation; forged or changed call/output/auxiliary ledger
edges may not release another owner. No first-token P2/G1 shell is ever
published during a two-token request.

## Proving gate and limits

Use two distinct authentic P2 prompts, greedy and nontrivial categorical
temperature/top-k/top-p. Compare token-one next logits and position-2 K/V
with the accepted P3 reference; compare token-two next logits and
position-3 K/V with the accepted T4 full-prefix path for the *same four
IDs*. Verify both A2 masks/lengths and every emitted ID/raw byte. Advance
an independent Philox oracle once/twice, including carry, final consumable
block, singleton and exhausted-second-draw cases. Force first-token EOS,
second-token EOS and no EOS with deterministic fixtures, checking `G=1/2`,
cache `3/4`, exact text/ID counts, and no second frame after first EOS.

Cut each provider, A2, I1 borrow/copy, T1 decode, native/Eshkol allocation,
frame prepare/commit and final-preflight site on both tokens. Assert the
appropriate pre-first or post-first state above, no partial output,
scrubbed unused bytes, then valid recovery/retry. Reject malformed owned
P2 I1 descriptors/storage and IDs `-1/256`, malformed/falsified sampled
IDs/logits/K/V, stale pins, wrong order, held I1/A2 views, forged/dead or
cross-output native/Eshkol owners, equal raw/staging tampering, altered
child ledger and output release under borrow. Prove feature-off symbols and
old P2/G1/manual/C2 paths unchanged; C2 must reject P2/G1 before and
after a C4 call. Require focused static closure and supported pinned
normal/repeat/ASan+UBSan+LSan native plus Eshkol gates before calling this
private leaf implemented.

This proposal adds no public P2/G2 profile, public result adapter, N>1,
P1/P3 request, G>2, cache/no-cache parity, save/reload, general decoding
loop, GPU, mixed precision or CLI3 `generate`. A separately reviewed public
admission/typed-copy/release contract must precede any installed caller.
