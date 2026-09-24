# G3-C4 Step 13A — private last-logit frame adapter

Status: isolated implementation candidate. This is not a public generation API.
It extends the accepted Step 12A source closure only.

## Source-grounded boundary

Step 12A returns one exact dense CPU-f32 `[1,256]` row. The accepted G3-S
greedy and categorical operations already consume that shape, while the older
Step 9A sampler boundary selects row three from a complete `[1,4,256]` tensor.
Step 13A adds the source-private
`et_g3c4_private_token_frame_begin_last_v1(context, last_logits, token)` under
`ET_G3C4_LAST_LOGIT_FRAME_PRIVATE`. It samples the supplied row directly and
opens the accepted Step 10A pending A2 append transaction. It does not fabricate
a four-row carrier.

The boundary requires an active generation call with one token remaining, an
idle token frame, the accepted sampler runtime, and the exact Step 12A parameter
binding. It admits the input and output spans before reading either, and rejects
overlap with each other, the context, owner, all 14 pinned parameter payloads,
or any live A2 cache object or backing allocation.

## Atomic behavior

Sampling runs against private logits and RNG copies. The committed cache length
is read through an A2 borrow, the borrow ends, and then the width-one append
transaction begins. Provider rejection or any of the three borrow and five
transaction allocations leaves the cache, RNG, binding, frame, caller output,
and active-call budget unchanged. After transaction creation, installing the
sampled token, successor RNG, position, frame state, and caller output is an
infallible publication tail. Step 10A still owns abort and joint cache/RNG/token
commit; Step 11A still owns one-token numerical staging.

## Evidence

The focused test feeds the real Step 12A P3 row into this adapter, executes the
accepted Step 11A continuation, publishes through Step 10A, and proves its next
logits bit-equal to row three of the accepted T4 reference for the sampled
token. Greedy and categorical results match the legacy Step 9A/10A begin path
for equivalent test input. The test covers one sampler dispatch cut, all eight
A2 allocation cuts, stale binding, non-finite logits, exhausted and wrong call
states, and both input and output aliases against every owned region. Normal,
repeated, ASan, UBSan, and leak-detection runs use the same test.

## Explicit limits and next dependency

Tokenizer identity and eval admission are a separate Eshkol call-entry
authority concern. T1 is an Eshkol aggregate whose baseline raw-V256 fingerprint
is `sha256:eshkol-byte-tokenizer-v1:aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704`;
no raw tokenizer pointer belongs in this C boundary. A later leaf must retain the
exact T1 shell in the generator, recheck exact tokenizer identity and model eval
mode before every native call, and bridge authenticated prompt ownership to the
borrowed native token span.

This leaf adds no public generation loop, prompt/result owner, text decoder,
EOS policy, persistence, package surface, or exported public symbol. Public
generation remains blocked until tokenizer/eval admission, prompt ownership,
and result/text ownership are complete. Public generation remains blocked by
those ownership and admission gaps.
