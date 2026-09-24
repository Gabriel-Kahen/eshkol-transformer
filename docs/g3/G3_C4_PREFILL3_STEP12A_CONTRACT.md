# G3-C4 Step 12A — private three-token numerical prefill

Status: isolated implementation candidate. This is not a public generation API.
It extends the accepted Step 11A source closure only.

## Source-grounded dependency closure

The accepted providers close a real `T=3` transformer schedule without fabricated
K/V state. G3-C4 owns token and position embedding, layer norm, linear, head
split/merge, GELU, and cached causal-attention rows. N2 owns residual addition.
A2 owns the unpublished candidate cache and its append transaction. The exact
21-role schedule uses shapes `[1,3,4]`, heads `[1,2,3,2]`, and cached attention
`[1,2,2,3,4,2]`.

## Added source-private boundary

`ET_G3C4_PREFILL3_PRIVATE` requires Step 11A and adds
`et_g3c4_private_prefill3_v1(context, token_ids, last_logits)`. The call accepts
one borrowed native `int64_t[3]` and returns one borrowed 256-float last-position
logit row. It creates an unpublished A2 cache, computes and stages all three K/V
rows, attends with the exact causal mask, completes the full model schedule, and
commits the candidate transaction. Only then does an infallible tail replace the
generator's old cache, native parameter snapshot, prompt-token copy, and output.

The snapshot binds the 14 accepted native tensor identities and their exact
4,768 parameter bytes. Continuation rejects a missing or stale snapshot before
sampling or opening an A2 transaction. A later successful prefill may replace a
stale snapshot because all candidate work remains unpublished until completion.

## Numerical and atomicity evidence

The focused test runs real accepted providers and authenticates all 21
capability, operation, and shape rows at the dispatch boundary. Its P3
last-position logits are bit-exact with row two of the accepted T4 full-prefix
path. A subsequent accepted Step 10A/11A token frame appends token four and is
bit-exact with row three of that same T4 reference. The committed cache advances
from exact keep mask `1110` and length three to length four without changing RNG
during prefill or numerical forward.

Each of the 21 provider dispatch sites and all 11 A2 candidate-cache,
transaction, and transaction-view allocation sites has an injected failure cut.
K1 runtime-discovery allocation also fails independently. Every failure retains
the prior cache, 14 identities, 4,768 snapshot bytes, prompt-token copy, RNG, and
caller output, while preserving the first error. Parameter mutation after call
acquisition proves stale continuation rejection; exact restoration permits the
same frame to begin. A source-private A2 storage query rejects output overlap
with the committed cache handle or any backing allocation; parameter-output
aliases are likewise rejected before numerical work, preserving the prior
binding and owned bytes.
Normal, repeated, ASan, UBSan, and leak-detection runs use the same focused test.

## Explicit limits

This is a native parameter snapshot, not the complete public binding required
by the accepted roadmap. The fixed owner topology is implicit, and no T1
tokenizer identity/fingerprint, tokenizer evaluation check, public prompt
authority, or result owner exists in this leaf. The returned `[256]` last-logit
row also cannot feed the accepted Step 9A/10A `[1024]` sampler transport; a
future exact adapter is required instead of a fabricated T4 carrier.

The leaf adds no public generation loop, text decoding, EOS behavior,
persistence, package surface, or exported symbol. Its borrowed native token
array is not public prompt ingestion. Step 11A remains independently compilable
with its original layout and behavior when the new feature macro is absent.
