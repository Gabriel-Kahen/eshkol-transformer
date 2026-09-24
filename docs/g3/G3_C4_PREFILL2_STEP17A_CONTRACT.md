# G3-C4 Step 17A — private two-token prefill

Status: isolated implementation candidate. This is not a public generation API.
It extends the accepted Step 16A source closure only.

## Boundary

`ET_G3C4_PREFILL2_PRIVATE` requires Step 16A and adds
`et_g3c4_private_prefill2_v1(context, token_ids, last_logits)`. The input is one
borrowed, aligned native array containing exactly two byte-token IDs. The output
is one borrowed 256-float row written only after successful numerical execution
and cache publication.

The boundary accepts active manual-prefill calls and generation calls with
budget zero or one. It rejects decode calls, non-byte IDs, invalid spans,
input/output overlap, and aliases with the context, model owner, pinned
parameters, or any A2 cache storage. It consumes no RNG and adds no prompt-owner
or Eshkol surface.

## Numerical schedule

The exact 21-role `T=2` schedule uses the accepted provider rows. N3K owns token
embedding, residual, linear, head-layout, and GELU rows. N2 owns layer norm.
G3-C4 owns learned position embedding and full-capacity cached causal attention.
Positions `{0,1}` attend through masks `{1,0,0,0}` and `{1,1,0,0}`. No padded
token or fabricated K/V row is admitted.

The focused witness compares the returned row bit-for-bit with row one of the
accepted T4 full-prefix path. It captures both exact K/V rows presented to
attention and proves that the committed A2 cache contains those values at
length two with keep mask `{1,1,0,0}`.

## Atomic replacement

The call creates an unpublished capacity-four cache, computes both K/V rows,
begins a two-position A2 transaction, stages layer zero, opens the full-capacity
view, runs attention and the remaining roles, authenticates all 14 pins,
snapshots their identities and exact 4,768 value bytes, and commits the
candidate transaction. Only after runtime cleanup and successful old-cache
destruction does an infallible tail publish the new cache, binding snapshot,
canonical prompt tuple `{token_ids[0],token_ids[1],0}`, binding-ready bit, and
last-position logits.

Every dispatch, runtime-discovery, and A2 allocation failure preserves the old
cache, binding, RNG, caller output, and active-call state. Cleanup preserves the
first error. Normal, repeated, ASan, UBSan, and leak-detection runs exercise all
21 provider cuts, both runtime-discovery cuts, all 11 A2 allocation cuts, owned
aliases, invalid IDs, and invalid call kind.

## Limits

This leaf proves P2 numerical prefill only. It adds no Step 15 I1 input-owner
binding, pre-acquire `P+G` admission, public result/text/EOS behavior, generation
loop, package export, or public symbol. Those remain separate dependency-ordered
leaves.
