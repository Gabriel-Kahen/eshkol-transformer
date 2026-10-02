# Private P2/G2 second-frame carrier contract

Proposed native ownership leaf; no implementation or runtime acceptance yet.
This follows the authentic non-EOS prefix commit and protected T1 prefix
coordinator, and complements the separately gated EOS first-frame carrier.
It admits only the existing diagnostic CPU-f32/V256 P2/G2 request with
committed first-token cache `3/1110`, retained 256 next logits, four successor
RNG words and the original unpublished capacity-two output. The first-EOS
route cannot enter this leaf because it has no committed non-EOS prefix.

## Exact private entry and upstream state

A new source-private `ET_G3C4_P2_G2_SECOND_FRAME_PRIVATE` feature requires
`ET_G3C4_P2_G2_PREFIX_COMMIT_PRIVATE` and the accepted carrier bridge and
last-logit frame/forward features. It adds one symbol:

```c
int64_t et_g3c4_private_p2g2_second_frame_carrier_v1(
    void *context, void *pending_output,
    void *staging_header, int64_t carrier_bytes);
```

The native entry authenticates the same active generator call `(kind,budget)
`(2,2)`, exact sole pending output, live model pins, unchanged prefill binding,
committed non-EOS prefix and idle token frame. The output retains requested
capacity two with numeric, ID-copy and text readiness all zero. Its I1 storage
must remain unused. A real A2 read borrow must establish committed length
three and mask `1110` before a draw or append. Carrier bytes are exactly 16,
with an aligned i64 header declaring eight payload bytes; header/range/full
native owned-storage alias checks use the accepted carrier rules. Eshkol
still must prove bytevector identity, allocator backing extent and original
call/input/output/auxiliary identities in its protected region; no arbitrary
pointer allocation proof follows from this native signature.

The caller passes no logits, selected token, raw byte, seed or successor RNG.
The sampler reads only the committed context's retained prefix next logits,
copying them into private float scratch. The existing G3-S greedy/categorical
route produces the genuine ID and all four speculative successor words.
Greedy draws nothing; categorical consumes exactly one available Philox
block, including a singleton or selected EOS. Exhaustion rejects before A2
transaction begin. No P2 prefill, P3 replay, four-row logits fabrication or
external pointer to retained storage is admitted.

## Frame transition and compatibility

The existing raw `token_frame_begin_last_v1` entry must continue rejecting
post-prefix state, even with this feature on: caller logits cannot authorize
a second draw. A separately private internal begin path may reuse its
numerical schedule only when called by the new bridge with authenticated
retained logits and the expected position-three route. First-frame admission
remains position two. Sampling finishes and committed cache is authenticated
before one width-one A2 transaction at position three begins.

Current prefix validation requires an idle frame and current P2/G2 forward
preflight rejects any committed prefix. Extend these checks only under the
new feature for the authenticated second-frame sampled/READY state at
position three. Record and validate the second-frame route separately from
first-frame state; no general permission for a prefix plus arbitrary frame.
The same accepted 21-role selected-token forward runs at position three and
stages exact T4 next logits, K/V, length four and mask `1111`. The committed
cache and generator RNG remain `3/1110` and the first-token successor until
a future separately reviewed terminal commit. Existing non-EOS prefix commit,
first-EOS carrier, output preparation/copy/text, call finish and public G2
rejection are not generalized into terminal consumers by this leaf.

After real forward, reauthenticate the exact context, pending output,
pins/binding/prefix, second-frame selected ID, READY position three and
carrier header/full aliases. Only then write the authentic selected ID as
eight little-endian payload bytes. The header remains unchanged; no next-logit
pointer, ID output shell, text, logical count or independently live output
is returned. Both selected second EOS and non-EOS are valid speculative
frames; decoding and EOS-aware terminal G2 publication remain later leaves.

## Failure and proof

Pre-frame errors leave all caller carrier bytes and committed prefix/cache/RNG
unchanged. Provider/forward errors may discard only the second transaction;
abort preserves committed first-token K/V, cache `3/1110`, binding and successor
RNG while scrubbing private continuation and pending carriers/output. A held
A2 view or pending I1 lease prevents abort before mutation; release of the
genuine lease permits retry. No implicit resume, RNG rewind or partial output
is published. An error after forward but before the carrier write leaves
that write absent and remains recoverable through the protected scope.

Required source/tests must prove real P2 -> first sample -> first forward ->
non-EOS prefix commit -> retained-logit second sample -> second forward.
Use two P2 prompts and greedy/nontrivial categorical policies, compare second
ID/four RNG words with an independent G3-S/Philox oracle and candidate logits/
K/V bitwise with a real four-token full-prefix forward. Exercise second EOS,
non-EOS, singleton/carry/final available/exhausted second draw, original raw
begin rejection, forged/stale/wrong pending owner, first-EOS/noncommitted/wrong
position, malformed/aliased carriers, provider cuts and header tampering,
unchanged parameters/binding, unready output, held-view abort/release/retry.
Normal/repeat and ASan+UBSan+LSan must pass pinned fe9/f31 gates with exact
feature-off source/symbol boundaries and all relevant predecessors. Native
ownership alone does not prove Eshkol T1 provenance, terminal G1/G2 output,
public admission, persistence, larger profiles or complete Wave 3 generation.
