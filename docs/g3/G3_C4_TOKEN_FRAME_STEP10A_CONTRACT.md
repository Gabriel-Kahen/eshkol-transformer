# G3-C4 Step 10A — private one-token frame and joint publication

Status: isolated implementation candidate. This is not a public generation API.
It extends the accepted Step 9A source closure only.

## Counterexample that bounds this leaf

The accepted Step 8A forward is fixed at `T=4`, and the current C4 A2 cache
capacity is also four. Sampling Step 8A's last logits row produces a fifth token.
Committing Step 8A's four-position transaction together with that sampled token
would publish an emitted token whose K/V is absent from cache, violating A0.
Therefore this leaf does not pretend that Step 8A is a token-forward producer.
It introduces the smaller ownership/commit mechanism and leaves the actual
one-token numerical forward as the next dependency.

## Added source-private boundary

`ET_G3C4_TOKEN_FRAME_PRIVATE` requires the complete Step 9A macro tuple and adds
four source-private calls:

- `token_frame_begin(context, full_logits, speculative_token)` invokes the
  accepted Step 9A sampler, begins one A2 append transaction, then records its
  token and numeric successor RNG in the authenticated active call. The caller
  receives the token only as a speculative candidate for numerical forwarding.
- `token_frame_stage(context, speculative_token, keys, values)` requires the
  exact recorded candidate and stages distinct finite dense CPU-f32 K/V with
  shape `[1,2,1,2]` into layer zero. These bytes must be the numerical forward
  result for that candidate; this leaf authenticates shape and candidate
  identity but does not produce them.
- `token_frame_publish(context, token)` requires the frame ready, commits the A2
  transaction, then performs an infallible serialized tail that copies the
  successor RNG, publishes the matching token, consumes the one-token budget,
  and clears frame state.
- `token_frame_abort(context)` aborts and scrubs a sampled or ready append and
  clears frame state without changing committed cache, generator RNG or budget.

Only an authenticated active generation call `(kind=2,budget=1)` may begin a
frame. One frame owns exactly one opaque A2 transaction. `call_finish` rejects a
pending frame. `call_abort` owns pending-frame cleanup before draining the call.
`call_prepare_end` accepts only a ready frame or an idle cache.

## Publication and failure atomicity

Sampling happens before A2 transaction allocation. The speculative output is
published only after both sampling and transaction begin succeed. Sampler or A2
allocation failure therefore leaves the frame idle and caller output unchanged.
A K/V staging failure aborts and scrubs the transaction while restoring the
first provider error. Candidate mismatch and commit-output alias rejection occur
before mutation and leave the frame available for explicit abort or retry.

A2 commit validates the opaque transaction and all staged layers before it
changes cache lengths. After it succeeds, no allocation, provider dispatch,
callback or recoverable branch occurs before RNG/token/budget publication and
frame reset. External serialization and the accepted active-call guard exclude
observation inside that tail. Generator policy, pins, model parameters and prior
committed cache bytes are otherwise unchanged.

The successor numeric RNG retains exact G3-S bits. Version and seed are checked
against the current generator RNG; both signed counter halves remain valid.
Greedy therefore republishes an unchanged RNG, while categorical publishes its
one-block successor only with the matching committed token and K/V.

## Explicit limits

This leaf does not implement the missing one-token numerical forward, positions,
attention over committed prefix, binding snapshots, prompt replacement, result
or text ownership, EOS behavior, multi-token continuation, public generation,
persistence, packaging or CI registration. Its staged K/V input is source
private and is not evidence that an upstream producer exists. No public token is
emitted by this repository until the actual forward producer and result facade
are composed and tested end to end.
