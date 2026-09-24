# G3-C4 Step 16A — private one-token prefill

Status: isolated implementation candidate. This is not a public generation API.
It extends the accepted Step 12A source closure only.

## Boundary

`ET_G3C4_PREFILL1_PRIVATE` requires Step 12A and adds
`et_g3c4_private_prefill1_v1(context, token_id, last_logits)`. The input is one
scalar byte-token ID. The output is one borrowed 256-float row written only
after successful numerical execution and cache publication.

The boundary accepts active manual-prefill calls and generation calls with
budget zero or one. It rejects decode calls, non-byte IDs, invalid output spans,
and output aliases with the context, model owner, pinned parameters, or any A2
cache storage. It consumes no RNG and adds no prompt-owner or Eshkol surface.

## Numerical schedule

The exact 21-role `T=1` schedule is the accepted Step 11A composition: G3-N
owns token embedding, residual, layer norm, linear, and head-layout rows; N2
owns exact-erf GELU; G3-C4 owns position embedding and full-capacity cached
causal attention. Query position zero attends to the one staged cache row in a
capacity-four cache. No padded token or fabricated K/V row is admitted.

The focused witness compares the returned row bit-for-bit with row zero of the
accepted T4 full-prefix path. It also captures the exact K/V presented to
attention and proves that the committed A2 cache contains those values at
length one with keep mask `{1,0,0,0}`.

## Atomic replacement

The call creates an unpublished capacity-four cache, computes K/V, begins a
one-position A2 transaction, stages layer zero, opens the full-capacity view,
runs attention and the remaining roles, authenticates all 14 pins, snapshots
their identities and exact 4,768 value bytes, and commits the candidate
transaction. Only after runtime cleanup and successful old-cache destruction
does an infallible tail publish the new cache, binding snapshot, canonical
prompt tuple `{token_id,0,0}`, binding-ready bit, and caller logits.

Every dispatch, runtime-discovery, and A2 allocation failure preserves the old
cache, binding, RNG, caller output, and active-call state. Cleanup preserves the
first error. Normal, repeated, ASan, UBSan, and leak-detection runs exercise all
21 provider cuts, both runtime-discovery cuts, all 11 A2 allocation cuts, output
aliases, invalid IDs, and invalid call kind.

## Limits

This leaf proves P1 numerical prefill only. It adds no P2 execution, I1 input
owner binding, pre-acquire `P+G` admission, public result/text/EOS behavior,
generation loop, package export, or public symbol. Those remain separate
dependency-ordered leaves.
