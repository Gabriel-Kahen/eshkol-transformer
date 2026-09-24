# G3-C4 Step 11A — private one-token numerical forward

Status: isolated implementation candidate. This is not a public generation API.
It extends the accepted Step 10A source closure only.

## Dependency audit

No single accepted provider owns the complete `T=1` schedule. The accepted K1
rows nevertheless form an exact closed composition:

| Provider | Rows used by this leaf |
| --- | --- |
| G3-N | token embedding `[1,1,256,4]`; linear `[1,1,4,4]`, `[1,1,4,8]`, `[1,1,8,4]`, and `[1,1,4,256]`; layer norm and residual `[1,1,4]`; head split/merge `[1,1,2,2]` |
| N2 | GELU `[1,1,8]` |
| G3-C4 | position embedding `[1,1,4,4]`; cached causal attention `[1,2,2,1,4,2]` |

A2 supplies the candidate transaction view after exact K/V staging. Its
effective length exists only after staging, while the position is needed to
compute those K/V. The accepted API closes that ordering without a new cache
surface: `token_frame_begin` borrows the committed layer-zero length before it
begins the append transaction and stores that position in the private frame.
The forward later requires the transaction view's effective length to equal
that stored position plus one.

## Added source-private boundary

`ET_G3C4_TOKEN_FORWARD_PRIVATE` requires Step 10A and adds
`et_g3c4_private_token_forward_v1(context, speculative_token, logits)`.
The call accepts only the exact sampled candidate in an authenticated
`(kind=2,budget=1)` frame. It executes the 21-role transformer schedule, stages
the numerical key/value outputs through the Step 10A boundary, attends over the
full candidate cache, and copies 256 next-token logits only after every role
succeeds. The frame remains ready and unpublished for the existing joint
cache/token/RNG/budget publication tail.

The schedule preserves the fixed 14-pin mapping: token and position embeddings,
pre-attention norm, Q/K/V projections, head layout, cached attention, attention
projection and residual, pre-FFN norm, up projection, GELU, down projection and
residual, final norm, and tied token-embedding output head. G3-N and N2 runtimes
are discovered per call; the already admitted G3-C4 runtime remains owned by the
private generator route layer.

## Numerical and atomicity evidence

The focused test runs real accepted providers and interposes only the K1
dispatch boundary to authenticate all 21 capability, operation, and shape rows.
Four consecutive private token frames are bit-exact against all four rows of the
accepted G3-C4 `T=4` full-prefix provider for the same token sequence. Separate
empty-cache and length-three cases prove position, mask, staged K/V publication,
committed-prefix preservation, determinism, and cache sensitivity.

Each of the 21 dispatch sites has an injected failure cut. All eight new A2
allocation cuts across the committed-position borrow and subsequent append
transaction are exercised before the frame can publish its speculative token.
K1 runtime-discovery allocation and A2 transaction-view allocation are also cut.
In every such case,
the first error survives cleanup, the pending transaction is aborted and
scrubbed, committed cache and generator RNG remain byte-identical, and caller
logits remain unchanged. Candidate mismatch and context/output alias rejection
occur before mutation and leave the sampled frame available for retry or abort.
Normal, repeated, ASan, UBSan, and leak-detection runs use the same test.

## Explicit limits

This leaf owns one numerical token forward only. It adds no prompt ingestion,
binding snapshot, initial prefill policy, result or text object, EOS behavior,
multi-token public loop, persistence, package surface, exported generation
symbol, or fabricated K/V input. Step 10A remains independently compilable with
its original layout and behavior when the new feature macro is absent. Public
generation and its end-to-end ownership contract remain pending.
